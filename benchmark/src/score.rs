//! Turn a model's answer.json into the full set of numbers we track.
//!
//! Word, temporal and spatial accuracy are kept as three independent scores
//! rather than folded together -- a model that reads the word but cannot say
//! where it saw it is telling us something different from one that localises
//! glyphs it cannot read.

use serde::{Deserialize, Serialize};

use crate::truth::{GroundTruth, LevelTruth};

/// A frame guess counts as correct within this many frames of truth.
pub const FRAME_TOLERANCE: i64 = 2;
/// A position guess counts as correct within this fraction of the diagonal.
pub const SPATIAL_TOLERANCE_FRAC: f64 = 0.05;

#[derive(Debug, Clone, Deserialize)]
pub struct AnswerLetter {
    #[serde(default)]
    pub c: String,
    #[serde(default)]
    pub frame: Option<i64>,
    #[serde(default)]
    pub x: Option<f64>,
    #[serde(default)]
    pub y: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    #[serde(default)]
    pub word: String,
    #[serde(default)]
    pub letters: Vec<AnswerLetter>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct PositionScore {
    pub index: usize,
    pub expected: String,
    pub got: Option<String>,
    pub font: String,
    pub char_ok: bool,
    pub frame_ok: bool,
    pub spatial_ok: bool,
    pub frame_delta: Option<i64>,
    pub pixel_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Score {
    pub parsed: bool,
    pub word_exact: bool,
    pub word_normalized: String,
    pub levenshtein_norm: f64,
    pub chars_correct: usize,
    pub chars_total: usize,
    pub first_error_index: Option<usize>,
    pub frames_correct: usize,
    pub spatial_correct: usize,
    pub median_frame_delta: Option<f64>,
    pub median_pixel_delta: Option<f64>,
    pub positions: Vec<PositionScore>,
}

/// Lowercase, strip anything that isn't a-z. Models like to answer
/// "theo loves obsidian" or wrap the word in quotes.
pub fn normalize(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_ascii_lowercase()).collect()
}

pub fn score(answer: Option<&Answer>, gt: &GroundTruth, lvl: &LevelTruth) -> Score {
    let mut sc = Score { chars_total: gt.target.len(), ..Default::default() };
    let Some(ans) = answer else { return sc };
    sc.parsed = true;

    let target: Vec<char> = gt.target.chars().collect();

    // Prefer the letters array as the source of truth for the word: it is
    // what carries localisation, and a model whose `word` field disagrees
    // with its own letters has not really solved it. Fall back to `word`
    // when the array is missing or the wrong length.
    let from_letters: String = ans.letters.iter().map(|l| normalize(&l.c)).collect();
    let from_word = normalize(&ans.word);
    sc.word_normalized = if from_letters.chars().count() == target.len() {
        from_letters.clone()
    } else if !from_word.is_empty() {
        from_word.clone()
    } else {
        from_letters.clone()
    };

    sc.word_exact = sc.word_normalized == gt.target;
    let maxlen = sc.word_normalized.chars().count().max(target.len()).max(1);
    sc.levenshtein_norm =
        1.0 - (strsim::levenshtein(&sc.word_normalized, &gt.target) as f64 / maxlen as f64);

    // Per-position scoring is index-aligned: the task defines position by
    // temporal order, so the Nth reported letter is compared to the Nth target.
    let mut frame_deltas = Vec::new();
    let mut pixel_deltas = Vec::new();
    let tol_px = gt.diagonal() * SPATIAL_TOLERANCE_FRAC;

    for (i, want) in target.iter().enumerate() {
        let tg = lvl.targets.iter().find(|t| t.index == i);
        let mut ps = PositionScore {
            index: i,
            expected: want.to_string(),
            font: tg.map(|t| t.font.clone()).unwrap_or_default(),
            ..Default::default()
        };

        if let Some(al) = ans.letters.get(i) {
            let got = normalize(&al.c);
            ps.got = Some(got.clone());
            ps.char_ok = got.chars().next() == Some(*want);

            if let (Some(tg), Some(f)) = (tg, al.frame) {
                let d = (f - tg.frame).abs();
                ps.frame_delta = Some(d);
                ps.frame_ok = d <= FRAME_TOLERANCE;
                frame_deltas.push(d as f64);
            }
            if let (Some(tg), Some(x), Some(y)) = (tg, al.x, al.y) {
                let (cx, cy) = tg.center();
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                ps.pixel_delta = Some(d);
                // Spatial credit requires being on the right frame too --
                // the right coordinates on the wrong frame is a coincidence.
                ps.spatial_ok = d <= tol_px && ps.frame_ok;
                pixel_deltas.push(d);
            }
        }

        if ps.char_ok {
            sc.chars_correct += 1;
        } else if sc.first_error_index.is_none() {
            sc.first_error_index = Some(i);
        }
        if ps.frame_ok {
            sc.frames_correct += 1;
        }
        if ps.spatial_ok {
            sc.spatial_correct += 1;
        }
        sc.positions.push(ps);
    }

    sc.median_frame_delta = median(&mut frame_deltas);
    sc.median_pixel_delta = median(&mut pixel_deltas);
    sc
}

fn median(v: &mut Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    Some(if n % 2 == 0 { (v[n / 2 - 1] + v[n / 2]) / 2.0 } else { v[n / 2] })
}

/// Pull the answer out of stdout when the model never wrote answer.json.
pub fn extract_json(text: &str) -> Option<Answer> {
    // Walk every balanced {...} span, newest first, and take the first one
    // that parses as an Answer with letters or a word.
    let bytes: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut stack = Vec::new();
    for (i, ch) in bytes.iter().enumerate() {
        match ch {
            '{' => stack.push(i),
            '}' => {
                if let Some(start) = stack.pop() {
                    spans.push((start, i));
                }
            }
            _ => {}
        }
    }
    for (s, e) in spans.into_iter().rev() {
        let frag: String = bytes[s..=e].iter().collect();
        if let Ok(a) = serde_json::from_str::<Answer>(&frag) {
            if !a.letters.is_empty() || !a.word.is_empty() {
                return Some(a);
            }
        }
    }
    None
}
