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
