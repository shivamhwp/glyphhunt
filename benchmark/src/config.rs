//! The benchmark grid: which models, at which effort, on which levels.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Harness {
    Claude,
    Codex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCfg {
    /// Short name used in the leaderboard.
    pub label: &'static str,
    pub harness: Harness,
    /// Value passed to `--model`.
    pub model: &'static str,
    /// Codex only: `model_reasoning_effort`. Claude has no equivalent flag.
    pub effort: Option<&'static str>,
}

/// `gpt-5.6` and `gpt-5.5-codex` are not available on this account -- probing
/// them returns 400 invalid_request_error -- so the 5.6 line is `gpt-5.6-sol`.
pub const GRID: &[ModelCfg] = &[
    ModelCfg { label: "opus-5",       harness: Harness::Claude, model: "opus",        effort: None },
    ModelCfg { label: "fable-5",      harness: Harness::Claude, model: "fable",       effort: None },
    ModelCfg { label: "gpt-5.6/high", harness: Harness::Codex,  model: "gpt-5.6-sol", effort: Some("high") },
    ModelCfg { label: "gpt-5.6/med",  harness: Harness::Codex,  model: "gpt-5.6-sol", effort: Some("medium") },
    ModelCfg { label: "gpt-5.5/high", harness: Harness::Codex,  model: "gpt-5.5",     effort: Some("high") },
    ModelCfg { label: "gpt-5.5/med",  harness: Harness::Codex,  model: "gpt-5.5",     effort: Some("medium") },
];

pub const LEVELS: &[u8] = &[1, 2, 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptMode {
    Blind,
    Hinted,
}

impl PromptMode {
    pub fn file(&self) -> &'static str {
        match self {
            PromptMode::Blind => "blind.txt",
            PromptMode::Hinted => "hinted.txt",
        }
    }
    pub fn short(&self) -> &'static str {
        match self {
            PromptMode::Blind => "blind",
            PromptMode::Hinted => "hint",
        }
    }
}

pub const MODES: &[PromptMode] = &[PromptMode::Blind, PromptMode::Hinted];

/// Codex runs use the personal profile per project convention.
pub const CODEX_HOME_SUFFIX: &str = ".codex-p";

/// Agentic video work is slow; anything past this is a hang, not a think.
pub const RUN_TIMEOUT_SECS: u64 = 1800;

/// Kill a run that has emitted no event for this long.
///
/// Lowering the hard ceiling to save time would be the wrong trade: observed
/// solves span 406s to 1751s, so a 20-minute cap would discard a third of
/// them. Silence is different from slowness. A working agent emits an event
/// on every shell command, so a long gap means it is stuck, not thinking.
/// Set at ten minutes rather than five: a high-effort model can reason for
/// several minutes before its first command, and killing that would lose a
/// real run. Total silence for ten minutes is unambiguous.
pub const IDLE_TIMEOUT_SECS: u64 = 600;

/// Run directories live OUTSIDE the project tree, under a neutral name.
///
/// With run dirs at `<project>/results/runs/<id>`, agents walked `../../..`,
/// found the generator, and re-ran it with the recorded seed to reproduce
/// exact glyph coordinates -- scoring a perfect blind L3 without looking at
/// the video at all. Distance alone is not a guarantee, so it is backed by
/// the prompt rule and the detector below.
pub const RUN_ROOT: &str = "/private/tmp/vidtask";

/// Every marker here names something that exists ONLY in the project tree.
///
/// Earlier sets were wrong twice over. `_sheet.png`, `_reference.mp4`,
/// `generator/` and `../../..` flagged agents for writing their own contact
/// sheets and nesting scratch directories. Then `/users/shivam` flagged the
/// Claude harness reading back its own truncated tool output, which it
/// stores under `~/.claude/projects/...` -- the agent's own work product,
/// nowhere near the benchmark.
///
/// The project lives at `.../Developer/t3/test/glyphhunt`, so any genuine
/// reference to it contains both `developer/t3` and `glyphhunt`. Matching the
/// home directory adds no detection and only catches innocents.
pub const INTEGRITY_MARKERS: &[&str] = &[
    "glyphhunt",
    "ground_truth",
    "base_montage",
    "developer/t3",
];
