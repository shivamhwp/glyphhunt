//! Final terminal report: leaderboard, level breakdown, hint value,
//! per-typeface difficulty, and what the agents actually did.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::record::Outcome;

/// Deserialising view of RunRecord -- the writing side owns the full struct.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub model: String,
    pub level: u8,
    pub mode: String,
    pub pass: String,
    pub outcome: Outcome,
    pub wall_ms: u128,
    pub concurrent_runs: usize,
    pub score: ScoreView,
    pub behavior: BehaviorView,
    pub usage: UsageView,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ScoreView {
    #[serde(default)]
    pub parsed: bool,
    #[serde(default)]
    pub word_exact: bool,
    #[serde(default)]
    pub chars_correct: usize,
    #[serde(default)]
    pub frames_correct: usize,
    #[serde(default)]
    pub spatial_correct: usize,
    #[serde(default)]
    pub levenshtein_norm: f64,
    #[serde(default)]
    pub positions: Vec<PositionView>,
    #[serde(default)]
    pub word_normalized: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PositionView {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub font: String,
    #[serde(default)]
    pub char_ok: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BehaviorView {
    #[serde(default)]
    pub shell_commands: u32,
    #[serde(default)]
    pub ffmpeg_invocations: u32,
    #[serde(default)]
    pub images_read: u32,
    #[serde(default)]
    pub python_invocations: u32,
    #[serde(default)]
    pub used_frame_diff: bool,
    #[serde(default)]
    pub used_contrast_stretch: bool,
    #[serde(default)]
    pub used_crop: bool,
    #[serde(default)]
    pub turns: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct UsageView {
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

impl Row {
    /// Normalised outcome, matching what the site reports.
    ///
    /// A run is recorded as TimedOut whenever its process was still alive at
    /// the wall-clock limit, even if it had already written a complete
    /// answer. Counting that as a failure conflates "produced nothing before
    /// the deadline" with "answered, but slowly" and inflates the failure
    /// rate. Judge on whether an answer parsed; keep the timeout separate.
    pub fn effective(&self) -> Outcome {
        if self.outcome == Outcome::Cheated {
            return Outcome::Cheated;
        }
        if self.score.parsed {
            return Outcome::Scored;
        }
        self.outcome
    }
    pub fn hit_wall(&self) -> bool {
        self.outcome == Outcome::TimedOut
    }
}

pub fn load(path: &Path) -> Result<Vec<Row>> {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    Ok(s.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Row>(l).ok())
        .collect())
}

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const R: &str = "\x1b[0m";
const CY: &str = "\x1b[36m";

fn tint(frac: f64) -> &'static str {
    if frac >= 0.85 {
        "\x1b[32m"
    } else if frac >= 0.5 {
        "\x1b[33m"
    } else if frac > 0.0 {
        "\x1b[91m"
    } else {
        "\x1b[2m"
    }
}

fn bar(frac: f64, width: usize) -> String {
    let n = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(n), "·".repeat(width - n))
}

#[derive(Default, Clone)]
struct Agg {
    runs: usize,
    exact: usize,
    chars: f64,
    frames: f64,
    spatial: f64,
    lev: f64,
    wall: f64,
    cost: f64,
    out_tok: u64,
    shell: f64,
    ffmpeg: f64,
    imgs: f64,
    diffs: usize,
    fails: usize,
    walls: usize,
}

fn agg_of<'a, I: Iterator<Item = &'a Row>>(rows: I) -> Agg {
    let mut a = Agg::default();
    for r in rows {
        a.runs += 1;
        a.exact += r.score.word_exact as usize;
        a.chars += r.score.chars_correct as f64;
        a.frames += r.score.frames_correct as f64;
        a.spatial += r.score.spatial_correct as f64;
        a.lev += r.score.levenshtein_norm;
        a.wall += r.wall_ms as f64;
        a.cost += r.usage.cost_usd.unwrap_or(0.0);
        a.out_tok += r.usage.output_tokens;
        a.shell += r.behavior.shell_commands as f64;
        a.ffmpeg += r.behavior.ffmpeg_invocations as f64;
        a.imgs += r.behavior.images_read as f64;
        a.diffs += r.behavior.used_frame_diff as usize;
        a.fails += (r.effective() != Outcome::Scored) as usize;
        a.walls += r.hit_wall() as usize;
    }
    a
}

pub fn print(rows: &[Row]) {
    if rows.is_empty() {
        println!("no results yet");
        return;
    }

    let models: Vec<String> = {
        let mut v: Vec<String> = rows.iter().map(|r| r.model.clone()).collect();
        v.sort();
        v.dedup();
        v
    };

    // ---- leaderboard -------------------------------------------------
    println!("\n{BOLD}  GLYPHHUNT - can a model find 'theolovesobsidian' hidden in 30s of video{R}");
    println!("{DIM}  {} runs | 17 glyphs, 17 typefaces, 1800 frames{R}\n", rows.len());

    let mut ranked: Vec<(String, Agg)> = models
        .iter()
        .map(|m| (m.clone(), agg_of(rows.iter().filter(|r| &r.model == m))))
        .collect();
    ranked.sort_by(|a, b| {
        (b.1.exact as f64 / b.1.runs.max(1) as f64, b.1.chars / b.1.runs.max(1) as f64)
            .partial_cmp(&(
                a.1.exact as f64 / a.1.runs.max(1) as f64,
                a.1.chars / a.1.runs.max(1) as f64,
            ))
            .unwrap()
    });

    println!("{CY}{BOLD}  #  model           solved    chars/17   accuracy          frame  spatial   avg time    cost{R}");
    println!("{DIM}  ─────────────────────────────────────────────────────────────────────────────────────────────{R}");
    for (i, (m, a)) in ranked.iter().enumerate() {
        let n = a.runs.max(1) as f64;
        let cf = a.chars / n / 17.0;
        println!(
            "  {:<2} {BOLD}{:<14}{R} {}{:>3}/{:<3}{R}  {}{:>5.1}{R}     {}{}{R}  {:>5.1}  {:>6.1}   {:>6.0}s  {:>6}",
            i + 1,
            m,
            tint(a.exact as f64 / n),
            a.exact,
            a.runs,
            tint(cf),
            a.chars / n,
            tint(cf),
            bar(cf, 14),
            a.frames / n,
            a.spatial / n,
            a.wall / n / 1000.0,
            if a.cost > 0.0 { format!("${:.2}", a.cost) } else { "-".into() },
        );
    }

    // ---- per level ---------------------------------------------------
    let mut levels: Vec<u8> = rows.iter().map(|r| r.level).collect();
    levels.sort();
    levels.dedup();

    println!("\n{BOLD}  BY LEVEL{R}  {DIM}(solved / runs, mean chars){R}");
    let names = ["", "L1 font+OCR", "L2 temporal", "L3 needle"];
    print!("\n  {:<14}", "model");
    for l in &levels {
        print!("{CY}{:<22}{R}", names.get(*l as usize).copied().unwrap_or("?"));
    }
    println!();
    println!("{DIM}  ────────────────────────────────────────────────────────────────────────{R}");
    for (m, _) in &ranked {
        print!("  {BOLD}{:<14}{R}", m);
        for l in &levels {
            let a = agg_of(rows.iter().filter(|r| &r.model == m && r.level == *l));
            let n = a.runs.max(1) as f64;
            let cf = a.chars / n / 17.0;
            print!(
                "{}{:>2}/{:<2} {:>5.1}/17 {}{R}  ",
                tint(cf),
                a.exact,
                a.runs,
                a.chars / n,
                bar(cf, 6)
            );
        }
        println!();
    }

    // ---- hint value --------------------------------------------------
    println!("\n{BOLD}  WHAT THE HINT IS WORTH{R}  {DIM}(mean chars, blind -> hinted){R}\n");
    for (m, _) in &ranked {
        let b = agg_of(rows.iter().filter(|r| &r.model == m && r.mode == "Blind"));
        let h = agg_of(rows.iter().filter(|r| &r.model == m && r.mode == "Hinted"));
        if b.runs == 0 && h.runs == 0 {
            continue;
        }
        let (bm, hm) = (b.chars / b.runs.max(1) as f64, h.chars / h.runs.max(1) as f64);
        // A delta against a mode with no runs is not a delta. Mid-grid one
        // side is often still empty, and printing e.g. -17.0 would read as a
        // finding rather than missing data.
        if b.runs == 0 || h.runs == 0 {
            println!(
                "  {BOLD}{:<14}{R} {}{:>5}{R} -> {}{:>5}{R}   {DIM}n/a{R}  {DIM}(blind {} / hinted {} runs){R}",
                m,
                DIM,
                if b.runs == 0 { "-".into() } else { format!("{bm:.1}") },
                DIM,
                if h.runs == 0 { "-".into() } else { format!("{hm:.1}") },
                b.runs,
                h.runs
            );
            continue;
        }
        let d = hm - bm;
        println!(
            "  {BOLD}{:<14}{R} {}{:>5.1}{R} -> {}{:>5.1}{R}   {}{:+.1}{R}",
            m,
            tint(bm / 17.0),
            bm,
            tint(hm / 17.0),
            hm,
            if d > 0.5 { "\x1b[32m" } else if d < -0.5 { "\x1b[91m" } else { DIM },
            d
        );
    }

    // ---- per typeface ------------------------------------------------
    let mut per_font: BTreeMap<(usize, String), (usize, usize)> = BTreeMap::new();
    for r in rows {
        for p in &r.score.positions {
            let e = per_font.entry((p.index, p.font.clone())).or_insert((0, 0));
            e.1 += 1;
            e.0 += p.char_ok as usize;
        }
    }
    if !per_font.is_empty() {
        println!("\n{BOLD}  HARDEST TYPEFACES{R}  {DIM}(share of runs that read this glyph correctly){R}\n");
        let mut v: Vec<_> = per_font.into_iter().collect();
        v.sort_by(|a, b| {
            let fa = a.1 .0 as f64 / a.1 .1.max(1) as f64;
            let fb = b.1 .0 as f64 / b.1 .1.max(1) as f64;
            fa.partial_cmp(&fb).unwrap()
        });
        for ((idx, font), (ok, tot)) in v {
            let f = ok as f64 / tot.max(1) as f64;
            println!(
                "  [{:>2}] {:<22} {}{} {:>5.0}%{R}  {DIM}{}/{}{R}",
                idx,
                font,
                tint(f),
                bar(f, 18),
                f * 100.0,
                ok,
                tot
            );
        }
    }

    // ---- agent behaviour ---------------------------------------------
    println!("\n{BOLD}  HOW THEY WORKED{R}  {DIM}(means per run){R}\n");
    println!("{CY}  model           shell  ffmpeg  images  turns   out-tok   diffed?  30m  failed{R}");
    println!("{DIM}  ──────────────────────────────────────────────────────────────────────{R}");
    for (m, a) in &ranked {
        let n = a.runs.max(1) as f64;
        let turns = agg_of(rows.iter().filter(|r| &r.model == m))
            .runs
            .max(1);
        let t: f64 = rows
            .iter()
            .filter(|r| &r.model == m)
            .map(|r| r.behavior.turns as f64)
            .sum::<f64>()
            / turns as f64;
        println!(
            "  {BOLD}{:<14}{R} {:>5.1}  {:>6.1}  {:>6.1}  {:>5.1}  {:>8.0}  {:>7}  {:>3}  {:>6}",
            m,
            a.shell / n,
            a.ffmpeg / n,
            a.imgs / n,
            t,
            a.out_tok as f64 / n,
            format!("{}/{}", a.diffs, a.runs),
            a.walls,
            a.fails
        );
    }

    // Latency is only meaningful from the sequential pass; say so rather
    // than quietly mixing contended numbers into the headline table.
    let seq: Vec<&Row> = rows.iter().filter(|r| r.pass == "latency").collect();
    if !seq.is_empty() {
        println!("\n{BOLD}  LATENCY{R}  {DIM}(sequential pass only, no CPU contention){R}\n");
        for (m, _) in &ranked {
            let a = agg_of(seq.iter().copied().filter(|r| &r.model == m));
            if a.runs == 0 {
                continue;
            }
            println!(
                "  {BOLD}{:<14}{R} {:>7.0}s  {DIM}over {} runs{R}",
                m,
                a.wall / a.runs as f64 / 1000.0,
                a.runs
            );
        }
    } else {
        let maxc = rows.iter().map(|r| r.concurrent_runs).max().unwrap_or(1);
        println!(
            "\n{DIM}  Latency omitted: all rows are from the parallel accuracy pass \
             (up to {maxc} concurrent).\n  Run with --pass latency --concurrency 1 for clean timings.{R}"
        );
    }
    println!();
}
