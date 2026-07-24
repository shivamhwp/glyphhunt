export const pct = (n: number, d: number) => (d > 0 ? n / d : 0);

export function bar(frac: number, width = 14) {
  const on = Math.round(Math.max(0, Math.min(1, frac)) * width);
  const cls = frac >= 0.85 ? 'on' : frac >= 0.5 ? 'on mid' : 'on low';
  return { on: '█'.repeat(on), off: '·'.repeat(width - on), cls };
}

/** Verdict for a group of runs. `— NO DATA` is distinct from `✗ UNSOLVED`. */
export function verdict(a: any) {
  if (!a || a.valid === 0) return { cls: 'mute', text: '— NO DATA' };
  if (a.exact >= a.valid) return { cls: 'pass', text: '✓ SOLVED' };
  if (a.exact > 0 || a.chars > 0) return { cls: 'warn', text: '~ PARTIAL' };
  return { cls: 'fail', text: '✗ UNSOLVED' };
}

/** Per-run outcome badge. */
export function runBadge(r: any) {
  if (r.cheated) return { cls: 'fail', text: '✗ INVALID' };
  if (r.exact) return { cls: 'pass', text: '✓ SOLVED' };
  // A run that answered is judged on its answer even if the process was still
  // alive at the wall-clock limit; `timed_out` stays visible separately.
  if (r.outcome !== 'Scored') return { cls: 'mute', text: '— TIMED OUT' };
  return r.chars > 0 ? { cls: 'warn', text: '~ PARTIAL' } : { cls: 'fail', text: '✗ MISSED' };
}

export const money = (n: number) => (n > 0 ? `$${n.toFixed(2)}` : '—');
export const secs = (n: number) => `${Math.round(n)}s`;
