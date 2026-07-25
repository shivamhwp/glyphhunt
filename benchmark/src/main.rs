//! glyphhunt -- can a model find a word hidden across a 30-second video?
//!
//! Two passes by design. The accuracy pass runs wide and parallel, because
//! correctness does not care about CPU contention. The latency pass re-runs a
//! subset strictly one at a time, because wall-clock numbers taken while
//! eight agents hammer ffmpeg are not measurements of anything.

mod config;
mod record;
mod report;
mod runner;
mod score;
mod truth;
mod tui;

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use config::{ModelCfg, PromptMode, GRID, LEVELS, RUN_ROOT};
use record::{Outcome, RunRecord};
use runner::RunSpec;
use truth::GroundTruth;

#[derive(Parser)]
#[command(name = "glyphhunt", about = "Hidden-text-in-video benchmark")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the grid.
    Run {
        #[arg(long, default_value = "3")]
        trials: u32,
        /// First trial number to emit. Lets a re-run of a lost cell keep its
        /// real trial index (e.g. --trial-base 2 --trials 1) instead of
        /// writing a duplicate trial 1.
        #[arg(long, default_value = "1")]
        trial_base: u32,
        #[arg(long, default_value = "8")]
        concurrency: usize,
        #[arg(long, default_value = "1,2,3")]
        levels: String,
        #[arg(long, default_value = "blind,hinted")]
        modes: String,
        /// Restrict to these model labels (comma separated).
        #[arg(long)]
        only: Option<String>,
        /// Label recorded on each row; use `latency` for the sequential pass.
        #[arg(long, default_value = "accuracy")]
        pass: String,
        /// Skip the TUI and stream plain lines (useful when backgrounded).
        #[arg(long)]
        plain: bool,
        /// Keep each run's extracted frames instead of reclaiming the space.
        #[arg(long)]
        keep_workdirs: bool,
        /// Skip cells already present in results/runs.jsonl. Lets a grid be
        /// restarted onto a fixed binary without repeating finished work.
        /// Cells lost to plan quota are NOT skipped -- those need re-running.
        #[arg(long)]
        resume: bool,
    },
    /// Re-print the report from an existing results file.
    Report {
        #[arg(default_value = "results/runs.jsonl")]
        path: PathBuf,
    },
}

fn load_avg() -> f32 {
    sysinfo::System::load_average().one as f32
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Report { path } => {
            let recs = report::load(&path)?;
            report::print(&recs);
            Ok(())
        }
        Cmd::Run { trials, trial_base, concurrency, levels, modes, only, pass, plain, keep_workdirs, resume } => {
            run(trials, trial_base, concurrency, levels, modes, only, pass, plain, keep_workdirs, resume).await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run(
    trials: u32,
    trial_base: u32,
    concurrency: usize,
    levels: String,
    modes: String,
    only: Option<String>,
    pass: String,
    plain: bool,
    keep_workdirs: bool,
    resume: bool,
) -> Result<()> {
    let root = std::env::current_dir()?;
    let gt = Arc::new(GroundTruth::load(&root.join("ground_truth.json"))?);

    let want_levels: Vec<u8> = levels
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|l| LEVELS.contains(l))
        .collect();
    let want_modes: Vec<PromptMode> = modes
        .split(',')
        .filter_map(|s| match s.trim() {
            "blind" => Some(PromptMode::Blind),
            "hinted" => Some(PromptMode::Hinted),
            _ => None,
        })
        .collect();
    let want_models: Vec<&ModelCfg> = GRID
        .iter()
        .filter(|m| {
            only.as_ref()
                .map(|o| o.split(',').any(|s| s.trim() == m.label))
                .unwrap_or(true)
        })
        .collect();

    let results_dir = root.join("results");
    std::fs::create_dir_all(&results_dir)?;
    let runs_path = results_dir.join("runs.jsonl");

    // Cells already recorded, so a resumed grid does not repeat them. A cell
    // that died on plan quota is deliberately left out of this set: it holds
    // no result and must be run again.
    let mut done: std::collections::HashSet<(String, u8, String, u32)> = Default::default();
    if resume {
        let txt = std::fs::read_to_string(&runs_path).unwrap_or_default();
        for line in txt.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let tail = v.get("final_message_tail").and_then(|t| t.as_str()).unwrap_or("").to_lowercase();
            let quota = v.get("outcome").and_then(|o| o.as_str()) == Some("Crashed")
                && ["session limit", "usage limit", "rate limit", "resets "]
                    .iter()
                    .any(|m| tail.contains(m));
            if quota {
                continue;
            }
            let (Some(m), Some(l), Some(md), Some(t)) = (
                v.get("model").and_then(|x| x.as_str()),
                v.get("level").and_then(|x| x.as_u64()),
                v.get("mode").and_then(|x| x.as_str()),
                v.get("trial").and_then(|x| x.as_u64()),
            ) else { continue };
            done.insert((m.to_string(), l as u8, md.to_string(), t as u32));
        }
    }

    // Build the whole work list up front so the progress denominator is real.
    let mut jobs = Vec::new();
    for trial in trial_base..(trial_base + trials) {
        for lvl in &want_levels {
            for mode in &want_modes {
                for m in &want_models {
                    let key = (
                        m.label.to_string(),
                        *lvl,
                        format!("{mode:?}"),
                        trial,
                    );
                    if done.contains(&key) {
                        continue;
                    }
                    jobs.push(((*m).clone(), *lvl, *mode, trial));
                }
            }
        }
    }
    if jobs.is_empty() {
        anyhow::bail!("no jobs -- check --levels/--modes/--only");
    }

    let sink = Arc::new(Mutex::new(
        std::fs::OpenOptions::new().create(true).append(true).open(&runs_path)?,
    ));

    let state: tui::State = Arc::new(Mutex::new(tui::Shared {
        total: jobs.len(),
        ..Default::default()
    }));

    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
    let inflight = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for (cfg, level, mode, trial) in jobs {
        let sem = sem.clone();
        let state = state.clone();
        let gt = gt.clone();
        let sink = sink.clone();
        let root = root.clone();
        let inflight = inflight.clone();
        let pass = pass.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let started = Instant::now();
            let run_id = format!(
                "{}-L{}-{}-t{}-{}",
                cfg.label.replace(['/', '.'], "_"),
                level,
                mode.short(),
                trial,
                chrono::Utc::now().format("%H%M%S%3f")
            );

            {
                let mut s = state.lock().unwrap();
                s.active.push(tui::Active {
                    label: cfg.label.to_string(),
                    level,
                    mode: mode.short(),
                    started,
                });
            }
            let concurrent = inflight.fetch_add(1, Ordering::SeqCst) + 1;
            let load = load_avg();

            let spec = RunSpec {
                run_dir: std::path::Path::new(RUN_ROOT).join(&run_id),
                video: root.join("videos").join(format!("level{level}.mp4")),
                prompt_path: root.join("prompts").join(mode.file()),
                cfg: cfg.clone(),
                mode,
            };

            let out = runner::execute(&spec).await;
            inflight.fetch_sub(1, Ordering::SeqCst);
            if !keep_workdirs {
                let _ = runner::cleanup(&spec.run_dir).await;
            }

            let base = |outcome, wall, msg: String| RunRecord {
                run_id: run_id.clone(),
                model: cfg.label.to_string(),
                harness: cfg.harness,
                model_id: cfg.model.to_string(),
                effort: cfg.effort.map(|s| s.to_string()),
                level,
                mode,
                trial,
                pass: pass.clone(),
                seed: gt.seed,
                started_at: chrono::Utc::now().to_rfc3339(),
                outcome,
                integrity_violation: false,
                exit_code: None,
                wall_ms: wall,
                load_avg_1m: load,
                concurrent_runs: concurrent,
                score: Default::default(),
                behavior: Default::default(),
                usage: Default::default(),
                raw_answer: Some(msg.clone()),
                answer_json_present: false,
                final_message_tail: msg,
                workdir_listing: vec![],
            };

            let rec = match out {
                Ok(o) => {
                    let parsed = o.raw_answer.as_deref().and_then(|s| {
                        serde_json::from_str::<score::Answer>(s)
                            .ok()
                            .or_else(|| score::extract_json(s))
                    });
                    let sc = gt
                        .level(level)
                        .ok()
                        .map(|lt| score::score(parsed.as_ref(), &gt, lt))
                        .unwrap_or_default();
                    let cheated = !o.behavior.integrity_violations.is_empty();
                    let outcome = if cheated {
                        Outcome::Cheated
                    } else if o.timed_out {
                        Outcome::TimedOut
                    } else if o.exit_code.unwrap_or(0) != 0 {
                        Outcome::Crashed
                    } else if parsed.is_none() {
                        Outcome::Unparseable
                    } else {
                        Outcome::Scored
                    };
                    // Zero the score outright rather than trusting a caller to
                    // filter on the flag later.
                    let sc = if cheated { Default::default() } else { sc };
                    let tail: String = {
                        let c: Vec<char> = o.final_message.chars().collect();
                        c[c.len().saturating_sub(600)..].iter().collect()
                    };
                    RunRecord {
                        integrity_violation: cheated,
                        exit_code: o.exit_code,
                        wall_ms: o.wall_ms,
                        outcome,
                        score: sc,
                        behavior: o.behavior,
                        usage: o.usage,
                        raw_answer: o.raw_answer,
                        answer_json_present: o.answer_json_present,
                        final_message_tail: tail,
                        workdir_listing: o.workdir_listing,
                        ..base(Outcome::Scored, 0, String::new())
                    }
                }
                Err(e) => base(
                    Outcome::Crashed,
                    started.elapsed().as_millis(),
                    format!("harness error: {e}"),
                ),
            };

            if let Ok(line) = serde_json::to_string(&rec) {
                let mut f = sink.lock().unwrap();
                let _ = writeln!(f, "{line}");
            }
            if plain {
                println!(
                    "{:<14} L{} {:<5} t{}  {:?}  chars {:>2}/17  frames {:>2}  exact {}  {}s",
                    rec.model, rec.level, rec.mode.short(), rec.trial, rec.outcome,
                    rec.score.chars_correct, rec.score.frames_correct,
                    rec.score.word_exact, rec.wall_ms / 1000
                );
            }

            let mut s = state.lock().unwrap();
            if let Some(p) = s
                .active
                .iter()
                .position(|a| a.label == cfg.label && a.level == level && a.mode == mode.short())
            {
                s.active.remove(p);
            }
            s.done.push(rec);
        }));
    }

    if plain {
        for h in handles {
            let _ = h.await;
        }
    } else {
        let ui = state.clone();
        let render = tokio::task::spawn_blocking(move || -> Result<()> {
            enable_raw_mode()?;
            let mut so = std::io::stdout();
            execute!(so, EnterAlternateScreen)?;
            let mut term = Terminal::new(CrosstermBackend::new(so))?;
            loop {
                tui::draw(&mut term, &ui)?;
                if event::poll(std::time::Duration::from_millis(400))? {
                    if let Event::Key(k) = event::read()? {
                        if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                            break;
                        }
                    }
                }
                if ui.lock().unwrap().finished {
                    tui::draw(&mut term, &ui)?;
                    std::thread::sleep(std::time::Duration::from_millis(700));
                    break;
                }
            }
            disable_raw_mode()?;
            execute!(term.backend_mut(), LeaveAlternateScreen)?;
            term.show_cursor()?;
            Ok(())
        });

        for h in handles {
            let _ = h.await;
        }
        state.lock().unwrap().finished = true;
        let _ = render.await;
    }

    let recs = report::load(&runs_path)?;
    report::print(&recs);
    println!("\nraw results: {}", runs_path.display());
    Ok(())
}
