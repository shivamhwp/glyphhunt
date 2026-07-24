//! Ground truth emitted by the Python generator.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TargetGlyph {
    pub index: usize,
    pub char: String,
    pub font: String,
    pub frame: i64,
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
    #[serde(default)]
    pub recovered_contrast: Option<f64>,
}

impl TargetGlyph {
    /// Generator records the glyph's top-left corner; models are asked for
    /// the centre, so compare against the centre.
    pub fn center(&self) -> (f64, f64) {
        (self.x as f64 + self.w as f64 / 2.0, self.y as f64 + self.h as f64 / 2.0)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LevelTruth {
    pub video: String,
    pub description: String,
    pub final_delta: Option<f64>,
    pub targets: Vec<TargetGlyph>,
    #[serde(default)]
    pub decoys: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GroundTruth {
    pub target: String,
    pub seed: u64,
    pub fps: f64,
    pub width: f64,
    pub height: f64,
    pub frames: i64,
    pub levels: HashMap<String, LevelTruth>,
}

impl GroundTruth {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("reading ground truth {}", path.display()))?;
        Ok(serde_json::from_str(&s)?)
    }

    pub fn level(&self, lvl: u8) -> Result<&LevelTruth> {
        self.levels
            .get(&lvl.to_string())
            .with_context(|| format!("ground truth has no level {lvl}"))
    }

    /// Diagonal in pixels -- spatial tolerance is expressed as a fraction of it.
    pub fn diagonal(&self) -> f64 {
        (self.width.powi(2) + self.height.powi(2)).sqrt()
    }
}
