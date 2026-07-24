//! One row per run, appended to results/runs.jsonl.

use serde::{Deserialize, Serialize};

use crate::config::{Harness, PromptMode};
use crate::runner::{Behavior, Usage};
use crate::score::Score;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// Answer parsed and scored.
    Scored,
    /// Process finished but produced nothing we could read as an answer.
    Unparseable,
    /// Hit the wall-clock limit.
    TimedOut,
    /// Non-zero exit.
    Crashed,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    pub run_id: String,
    pub model: String,
    pub harness: Harness,
    pub model_id: String,
    pub effort: Option<String>,
    pub level: u8,
    pub mode: PromptMode,
    pub trial: u32,
    /// Which measurement pass this belongs to -- latency numbers are only
    /// trustworthy from the sequential pass.
    pub pass: String,
    pub seed: u64,
    pub started_at: String,
    pub outcome: Outcome,
    pub exit_code: Option<i32>,
    pub wall_ms: u128,
    /// Machine load sampled during the run, so a latency number taken under
    /// contention can be identified rather than silently trusted.
    pub load_avg_1m: f32,
    pub concurrent_runs: usize,
    pub score: Score,
    pub behavior: Behavior,
    pub usage: Usage,
    pub raw_answer: Option<String>,
    pub answer_json_present: bool,
    pub final_message_tail: String,
    pub workdir_listing: Vec<String>,
}

impl RunRecord {
    pub fn cell(&self) -> String {
        format!("{}|L{}|{}|t{}", self.model, self.level, self.mode.short(), self.trial)
    }
}
