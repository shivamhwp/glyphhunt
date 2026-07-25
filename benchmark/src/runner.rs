//! Spawn one CLI against one video, in an isolated directory, and mine the
//! event stream for everything the model did.
//!
//! Each run gets its own working directory containing nothing but `clip.mp4`
//! and the prompt, so runs can never see each other's extracted frames.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::config::{
    Harness, ModelCfg, PromptMode, CODEX_HOME_SUFFIX, IDLE_TIMEOUT_SECS, RUN_TIMEOUT_SECS,
};

#[derive(Debug, Clone, Serialize, Default)]
pub struct Behavior {
    pub turns: u32,
    pub tool_calls: u32,
    pub tool_breakdown: BTreeMap<String, u32>,
    pub shell_commands: u32,
    pub ffmpeg_invocations: u32,
    pub python_invocations: u32,
    pub images_read: u32,
    /// Sampling rates the model asked ffmpeg for, e.g. `fps=2`.
    pub chosen_fps: Vec<String>,
    pub used_crop: bool,
    pub used_scale: bool,
    pub used_frame_diff: bool,
    pub used_contrast_stretch: bool,
    /// Highest `-frames`/`select` count we can attribute to a single command.
    pub max_frames_extracted: u64,
    /// Commands showing the agent left its working directory and reached the
    /// benchmark's own source. Any entry invalidates the run's score.
    pub integrity_violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cost_usd: Option<f64>,
    pub api_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunOutput {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    /// Killed for emitting nothing at all, rather than for taking too long.
    pub stalled: bool,
    pub wall_ms: u128,
    pub final_message: String,
    pub answer_json_present: bool,
    pub raw_answer: Option<String>,
    pub behavior: Behavior,
    pub usage: Usage,
    pub workdir_listing: Vec<String>,
}

fn scan_command(cmd: &str, b: &mut Behavior) {
    b.shell_commands += 1;
    let lower = cmd.to_lowercase();

    for m in crate::config::INTEGRITY_MARKERS {
        if lower.contains(m) && b.integrity_violations.len() < 20 {
            let snippet: String = cmd.chars().take(240).collect();
            b.integrity_violations.push(format!("[{m}] {snippet}"));
            break;
        }
    }
    if lower.contains("ffmpeg") || lower.contains("ffprobe") {
        b.ffmpeg_invocations += 1;
    }
    if lower.contains("python") || lower.contains("numpy") {
        b.python_invocations += 1;
    }
    if lower.contains("crop") {
        b.used_crop = true;
    }
    if lower.contains("scale=") || lower.contains("-vf scale") {
        b.used_scale = true;
    }
    // Signals that the model reasoned about the hiding mechanism rather than
    // just eyeballing frames.
    if lower.contains("tblend") || lower.contains("absdiff") || lower.contains("difference")
        || lower.contains("np.diff") || lower.contains("frame_diff")
    {
        b.used_frame_diff = true;
    }
    if lower.contains("eq=contrast") || lower.contains("normalize") || lower.contains("histeq")
        || lower.contains("autolevel") || lower.contains("curves")
    {
        b.used_contrast_stretch = true;
    }
    // `fps=N` / `-r N`
    for token in lower.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '=')) {
        if let Some(v) = token.strip_prefix("fps=") {
            if !v.is_empty() && b.chosen_fps.len() < 24 {
                b.chosen_fps.push(v.to_string());
            }
        }
    }
}

fn parse_claude_event(v: &serde_json::Value, out: &mut RunOutput) {
    match v.get("type").and_then(|t| t.as_str()) {
        Some("assistant") => {
            let Some(content) = v.pointer("/message/content").and_then(|c| c.as_array()) else {
                return;
            };
            for c in content {
                if c.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    out.behavior.tool_calls += 1;
                    *out.behavior.tool_breakdown.entry(name.to_string()).or_insert(0) += 1;
                    if name == "Bash" {
                        if let Some(cmd) = c.pointer("/input/command").and_then(|s| s.as_str()) {
                            scan_command(cmd, &mut out.behavior);
                        }
                    }
                    if name == "Read" {
                        if let Some(p) = c.pointer("/input/file_path").and_then(|s| s.as_str()) {
                            let p = p.to_lowercase();
                            if p.ends_with(".png") || p.ends_with(".jpg") || p.ends_with(".jpeg")
                                || p.ends_with(".webp")
                            {
                                out.behavior.images_read += 1;
                            }
                        }
                    }
                } else if c.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = c.get("text").and_then(|s| s.as_str()) {
                        out.final_message = t.to_string();
                    }
                }
            }
        }
        Some("result") => {
            out.usage.cost_usd = v.get("total_cost_usd").and_then(|c| c.as_f64());
            out.usage.api_duration_ms = v.get("duration_api_ms").and_then(|c| c.as_u64());
            out.behavior.turns = v.get("num_turns").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
            if let Some(u) = v.get("usage") {
                out.usage.input_tokens = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.usage.cached_input_tokens =
                    u.get("cache_read_input_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.usage.output_tokens =
                    u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
            }
            if let Some(r) = v.get("result").and_then(|s| s.as_str()) {
                if !r.is_empty() {
                    out.final_message = r.to_string();
                }
            }
        }
        _ => {}
    }
}

fn parse_codex_event(v: &serde_json::Value, out: &mut RunOutput) {
    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "item.completed" | "item.started" => {
            let Some(item) = v.get("item") else { return };
            let itype = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
            // Only count each item once, at completion.
            if ty == "item.started" {
                return;
            }
            match itype {
                "command_execution" => {
                    out.behavior.tool_calls += 1;
                    *out.behavior.tool_breakdown.entry("shell".into()).or_insert(0) += 1;
                    if let Some(cmd) = item.get("command").and_then(|s| s.as_str()) {
                        scan_command(cmd, &mut out.behavior);
                    }
                }
                "agent_message" => {
                    if let Some(t) = item.get("text").and_then(|s| s.as_str()) {
                        out.final_message = t.to_string();
                    }
                }
                other => {
                    if !other.is_empty() && other != "reasoning" {
                        out.behavior.tool_calls += 1;
                        *out.behavior.tool_breakdown.entry(other.to_string()).or_insert(0) += 1;
                    }
                    if other == "view_image" {
                        out.behavior.images_read += 1;
                    }
                }
            }
        }
        // Codex reports quota exhaustion as an `error` event, not as an agent
        // message, so without this the record's final message is empty and the
        // run looks like an ordinary crash -- counting a billing limit against
        // the model.
        "error" | "turn.failed" => {
            let msg = v
                .get("message")
                .and_then(|m| m.as_str())
                .or_else(|| v.pointer("/error/message").and_then(|m| m.as_str()));
            if let Some(m) = msg {
                out.final_message = m.to_string();
            }
        }
        "turn.completed" => {
            out.behavior.turns += 1;
            if let Some(u) = v.get("usage") {
                out.usage.input_tokens += u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.usage.cached_input_tokens +=
                    u.get("cached_input_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.usage.output_tokens +=
                    u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
                out.usage.reasoning_tokens +=
                    u.get("reasoning_output_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
            }
        }
        _ => {}
    }
}

/// Copy the video into the run dir. On APFS `cp -c` clones it, so 100+ runs
/// don't cost 100x the disk.
async fn place_video(src: &Path, dst: &Path) -> Result<()> {
    let ok = Command::new("cp")
        .arg("-c")
        .arg(src)
        .arg(dst)
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        tokio::fs::copy(src, dst).await.context("copying clip into run dir")?;
    }
    Ok(())
}

/// Everything worth keeping from a finished run. Agents routinely extract
/// hundreds of MB of frames; across a full grid that is tens of GB of
/// reconstructible scratch, so drop it and keep the evidence.
const KEEP: &[&str] = &["answer.json", "events.jsonl", "stderr.log"];

pub async fn cleanup(run_dir: &Path) -> Result<u64> {
    let mut freed = 0u64;
    let mut rd = tokio::fs::read_dir(run_dir).await?;
    while let Ok(Some(e)) = rd.next_entry().await {
        let name = e.file_name().to_string_lossy().to_string();
        if KEEP.contains(&name.as_str()) {
            continue;
        }
        let md = e.metadata().await?;
        if md.is_dir() {
            freed += dir_size(&e.path()).await;
            let _ = tokio::fs::remove_dir_all(e.path()).await;
        } else {
            freed += md.len();
            let _ = tokio::fs::remove_file(e.path()).await;
        }
    }
    Ok(freed)
}

async fn dir_size(p: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(mut rd) = tokio::fs::read_dir(&d).await else { continue };
        while let Ok(Some(e)) = rd.next_entry().await {
            match e.metadata().await {
                Ok(m) if m.is_dir() => stack.push(e.path()),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

pub struct RunSpec {
    pub run_dir: PathBuf,
    pub video: PathBuf,
    pub prompt_path: PathBuf,
    pub cfg: ModelCfg,
    pub mode: PromptMode,
}

pub async fn execute(spec: &RunSpec) -> Result<RunOutput> {
    tokio::fs::create_dir_all(&spec.run_dir).await?;
    place_video(&spec.video, &spec.run_dir.join("clip.mp4")).await?;
    let prompt = tokio::fs::read_to_string(&spec.prompt_path).await?;

    let home = std::env::var("HOME").unwrap_or_default();
    let mut cmd = match spec.cfg.harness {
        Harness::Claude => {
            let mut c = Command::new("claude");
            c.arg("-p")
                .arg(&prompt)
                .arg("--model")
                .arg(spec.cfg.model)
                .arg("--output-format")
                .arg("stream-json")
                .arg("--verbose")
                .arg("--dangerously-skip-permissions");
            c
        }
        Harness::Codex => {
            let mut c = Command::new("codex");
            c.env("CODEX_HOME", format!("{home}/{CODEX_HOME_SUFFIX}"))
                .env("CMUX_CODEX_HOOKS_DISABLED", "1")
                .arg("exec")
                .arg("--json")
                .arg("-m")
                .arg(spec.cfg.model)
                .arg("--skip-git-repo-check")
                .arg("--dangerously-bypass-approvals-and-sandbox");
            if let Some(e) = spec.cfg.effort {
                c.arg("-c").arg(format!("model_reasoning_effort={e}"));
            }
            c.arg(&prompt);
            c
        }
    };

    cmd.current_dir(&spec.run_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        // Put the agent in its own process group. Killing only the agent
        // leaves the ffmpeg/python children it spawned alive, and those
        // inherit the stdout pipe -- so the collector blocks on a pipe that
        // never closes and a run can overshoot its ceiling by many minutes.
        // With a group we can signal the whole tree.
        .process_group(0);

    let started = Instant::now();
    let mut child = cmd.spawn().context("spawning CLI")?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let events_path = spec.run_dir.join("events.jsonl");
    let harness = spec.cfg.harness;
    let last_event = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tick = std::sync::Arc::new(std::time::Instant::now());

    let stamp = last_event.clone();
    let clock = tick.clone();
    let collector = tokio::spawn(async move {
        let mut out = RunOutput {
            exit_code: None,
            timed_out: false,
            stalled: false,
            wall_ms: 0,
            final_message: String::new(),
            answer_json_present: false,
            raw_answer: None,
            behavior: Behavior::default(),
            usage: Usage::default(),
            workdir_listing: Vec::new(),
        };
        let mut raw = String::new();
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            stamp.store(clock.elapsed().as_secs(), std::sync::atomic::Ordering::Relaxed);
            raw.push_str(&line);
            raw.push('\n');
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                match harness {
                    Harness::Claude => parse_claude_event(&v, &mut out),
                    Harness::Codex => parse_codex_event(&v, &mut out),
                }
            }
        }
        let _ = tokio::fs::write(&events_path, raw).await;
        out
    });

    let err_path = spec.run_dir.join("stderr.log");
    let errs = tokio::spawn(async move {
        let mut buf = String::new();
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(l)) = lines.next_line().await {
            buf.push_str(&l);
            buf.push('\n');
        }
        let _ = tokio::fs::write(&err_path, buf).await;
    });

    // Watchdog: kill the process group once the event stream goes quiet.
    let pid = child.id();
    let watch_stamp = last_event.clone();
    let watch_clock = tick.clone();
    let stalled_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stalled_set = stalled_flag.clone();
    let watchdog = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            let last = watch_stamp.load(std::sync::atomic::Ordering::Relaxed);
            if watch_clock.elapsed().as_secs().saturating_sub(last) >= IDLE_TIMEOUT_SECS {
                stalled_set.store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(p) = pid {
                    unsafe {
                        libc::kill(-(p as i32), libc::SIGKILL);
                    }
                }
                return;
            }
        }
    });

    let timed_out = tokio::time::timeout(
        std::time::Duration::from_secs(RUN_TIMEOUT_SECS),
        child.wait(),
    )
    .await
    .is_err();
    watchdog.abort();
    let stalled = stalled_flag.load(std::sync::atomic::Ordering::Relaxed);

    let exit_code = if timed_out {
        // Negative pid signals the entire process group.
        if let Some(pid) = child.id() {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        let _ = child.start_kill();
        None
    } else {
        child.try_wait().ok().flatten().and_then(|s| s.code())
    };

    let mut out = collector.await?;
    let _ = errs.await;
    out.exit_code = exit_code;
    out.timed_out = timed_out || stalled;
    out.stalled = stalled;
    out.wall_ms = started.elapsed().as_millis();

    // Prefer the file the prompt asked for; fall back to the final message.
    let ans_path = spec.run_dir.join("answer.json");
    if let Ok(s) = tokio::fs::read_to_string(&ans_path).await {
        out.answer_json_present = true;
        out.raw_answer = Some(s);
    } else if !out.final_message.is_empty() {
        out.raw_answer = Some(out.final_message.clone());
    }

    if let Ok(mut rd) = tokio::fs::read_dir(&spec.run_dir).await {
        while let Ok(Some(e)) = rd.next_entry().await {
            out.workdir_listing.push(e.file_name().to_string_lossy().to_string());
        }
        out.workdir_listing.sort();
        out.workdir_listing.truncate(200);
    }

    Ok(out)
}
