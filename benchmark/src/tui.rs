//! Live leaderboard while the grid runs.

use std::collections::BTreeMap;
use std::io::Stdout;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Row, Table},
    Terminal,
};

use crate::record::{Outcome, RunRecord};

#[derive(Debug, Clone)]
pub struct Active {
    pub label: String,
    pub level: u8,
    pub mode: &'static str,
    /// Held as a start instant, not a duration, so the display ticks without
    /// anything having to push updates into it.
    pub started: std::time::Instant,
}

#[derive(Default)]
pub struct Shared {
    pub done: Vec<RunRecord>,
    pub active: Vec<Active>,
    pub total: usize,
    pub finished: bool,
}

pub type State = Arc<Mutex<Shared>>;

#[derive(Default, Clone)]
struct Agg {
    runs: usize,
    exact: usize,
    chars: f64,
    frames: f64,
    spatial: f64,
    wall_ms: f64,
    cost: f64,
    best_chars: usize,
    failures: usize,
}

fn aggregate(done: &[RunRecord]) -> BTreeMap<String, Agg> {
    let mut m: BTreeMap<String, Agg> = BTreeMap::new();
    for r in done {
        let a = m.entry(r.model.clone()).or_default();
        a.runs += 1;
        if r.score.word_exact {
            a.exact += 1;
        }
        a.chars += r.score.chars_correct as f64;
        a.frames += r.score.frames_correct as f64;
        a.spatial += r.score.spatial_correct as f64;
        a.wall_ms += r.wall_ms as f64;
        a.cost += r.usage.cost_usd.unwrap_or(0.0);
        a.best_chars = a.best_chars.max(r.score.chars_correct);
        if r.outcome != Outcome::Scored {
            a.failures += 1;
        }
    }
    m
}

fn score_color(frac: f64) -> Color {
    if frac >= 0.85 {
        Color::Green
    } else if frac >= 0.5 {
        Color::Yellow
    } else if frac > 0.0 {
        Color::LightRed
    } else {
        Color::DarkGray
    }
}

pub fn draw(term: &mut Terminal<CrosstermBackend<Stdout>>, state: &State) -> Result<()> {
    let snap = { state.lock().unwrap().clone_parts() };
    let (done, active, total) = snap;
    let agg = aggregate(&done);

    let mut rows: Vec<(String, Agg)> = agg.into_iter().collect();
    rows.sort_by(|a, b| {
        let ka = (a.1.exact as f64, a.1.chars / a.1.runs.max(1) as f64);
        let kb = (b.1.exact as f64, b.1.chars / b.1.runs.max(1) as f64);
        kb.partial_cmp(&ka).unwrap()
    });

    term.draw(|f| {
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(9),
        ])
        .split(f.area());

        let pct = if total == 0 { 0.0 } else { done.len() as f64 / total as f64 };
        f.render_widget(
            Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(format!(
                    " glyphhunt - {}/{} runs - {} active ",
                    done.len(),
                    total,
                    active.len()
                )))
                .gauge_style(Style::default().fg(Color::Cyan))
                .ratio(pct.clamp(0.0, 1.0))
                .label(format!("{:.0}%", pct * 100.0)),
            chunks[0],
        );

        let header = Row::new(vec![
            "#", "model", "runs", "exact", "chars", "frame", "spatial", "best", "avg t", "cost", "fail",
        ])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan));

        let body: Vec<Row> = rows
            .iter()
            .enumerate()
            .map(|(i, (name, a))| {
                let n = a.runs.max(1) as f64;
                let cf = a.chars / n / 17.0;
                Row::new(vec![
                    Cell::from(format!("{}", i + 1)),
                    Cell::from(name.clone()).style(Style::default().add_modifier(Modifier::BOLD)),
                    Cell::from(format!("{}", a.runs)),
                    Cell::from(format!("{}/{}", a.exact, a.runs))
                        .style(Style::default().fg(score_color(a.exact as f64 / n))),
                    Cell::from(format!("{:.1}/17", a.chars / n))
                        .style(Style::default().fg(score_color(cf))),
                    Cell::from(format!("{:.1}", a.frames / n))
                        .style(Style::default().fg(score_color(a.frames / n / 17.0))),
                    Cell::from(format!("{:.1}", a.spatial / n))
                        .style(Style::default().fg(score_color(a.spatial / n / 17.0))),
                    Cell::from(format!("{}", a.best_chars)),
                    Cell::from(format!("{:.0}s", a.wall_ms / n / 1000.0)),
                    Cell::from(if a.cost > 0.0 { format!("${:.2}", a.cost) } else { "-".into() }),
                    Cell::from(format!("{}", a.failures)).style(Style::default().fg(
                        if a.failures > 0 { Color::Red } else { Color::DarkGray },
                    )),
                ])
            })
            .collect();

        f.render_widget(
            Table::new(
                body,
                [
                    Constraint::Length(3),
                    Constraint::Length(14),
                    Constraint::Length(5),
                    Constraint::Length(8),
                    Constraint::Length(9),
                    Constraint::Length(7),
                    Constraint::Length(8),
                    Constraint::Length(5),
                    Constraint::Length(7),
                    Constraint::Length(7),
                    Constraint::Length(5),
                ],
            )
            .header(header)
            .block(Block::default().borders(Borders::ALL).title(" leaderboard ")),
            chunks[1],
        );

        let mut lines: Vec<Line> = active
            .iter()
            .map(|a| {
                Line::from(vec![
                    Span::styled(format!("  {:<14}", a.label), Style::default().fg(Color::White)),
                    Span::styled(
                        format!("L{} {:<6}", a.level, a.mode),
                        Style::default().fg(Color::Magenta),
                    ),
                    Span::styled(
                        format!("{:>4}s", a.started.elapsed().as_secs()),
                        Style::default().fg(Color::DarkGray),
                    ),
                ])
            })
            .collect();
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "  (idle)",
                Style::default().fg(Color::DarkGray),
            )));
        }
        f.render_widget(
            ratatui::widgets::Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(" in flight ")),
            chunks[2],
        );
    })?;
    Ok(())
}

impl Shared {
    fn clone_parts(&self) -> (Vec<RunRecord>, Vec<Active>, usize) {
        (self.done.clone(), self.active.clone(), self.total)
    }
}
