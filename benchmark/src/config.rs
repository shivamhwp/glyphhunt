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

/// Run directories live OUTSIDE the project tree, under a neutral name.
///
/// With run dirs at `<project>/results/runs/<id>`, agents walked `../../..`,
/// found the generator, and re-ran it with the recorded seed to reproduce
/// exact glyph coordinates -- scoring a perfect blind L3 without looking at
/// the video at all. Distance alone is not a guarantee, so it is backed by
/// the prompt rule and the detector below.
pub const RUN_ROOT: &str = "/private/tmp/vidtask";

/// Markers that can only appear in an agent's shell history if it left its
/// working directory and found the benchmark's own source or answers.
///
/// Deliberately narrow. Matching on bare filenames like `verify.py` or
/// `render.py` would flag honest runs -- agents routinely write scratch
/// scripts by those names -- and matching the target word would flag every
/// correct answer. These are artifacts that exist only in the project tree,
/// plus the escape sequence that reaches it from a run directory.
/// Every marker here names something that exists ONLY in the project tree.
///
/// Earlier versions also matched `_sheet.png`, `_reference.mp4`, `generator/`
/// and `../../..`. All four were wrong. An agent tiling candidate frames
/// naturally writes something like `suspect_sheet.png`, and one was flagged
/// for exactly that despite never leaving its directory. And now that run
/// directories live under `/private/tmp/vidtask`, a short relative path
/// cannot reach the project at all -- `../../..` lands in `/private`, so it
/// signals nothing while flagging agents that merely nest their scratch
/// directories. Anything that genuinely reaches the project has to name it.
pub const INTEGRITY_MARKERS: &[&str] = &[
    "glyphhunt",
    "ground_truth",
    "base_montage",
    "/users/shivam",
    "developer/t3",
];
