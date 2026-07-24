#!/bin/sh
# Re-run cells that were lost to Claude plan quota rather than to the task.
#
# Those runs died on "You've hit your session limit", which says nothing about
# the model, so they are excluded from scoring and re-run here to keep trial
# counts even across the grid. --trial-base preserves the real trial index so
# a replacement for trial 2 is recorded as trial 2, not a second trial 1.
#
# Run this only once the main grid has drained, so the two do not compete for
# the same quota and CPU.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN=./benchmark/target/release/glyphhunt

if pgrep -f "glyphhunt run" >/dev/null 2>&1; then
    echo "main grid still running -- wait for it to finish first" >&2
    exit 1
fi

# Which cells are missing, derived from the results rather than hardcoded.
python3 - <<'PY' > /tmp/glyphhunt_lost.txt
import json, pathlib
root = pathlib.Path(".")
rows = [json.loads(l) for l in (root / "results/runs.jsonl").read_text().splitlines() if l.strip()]
markers = ("session limit", "usage limit", "rate limit", "resets ")
lost = [
    r for r in rows
    if r["outcome"] == "Crashed"
    and any(m in (r.get("final_message_tail") or "").lower() for m in markers)
]
for r in lost:
    print(f"{r['model']}\t{r['level']}\t{r['mode'].lower()}\t{r['trial']}")
PY

count=$(wc -l < /tmp/glyphhunt_lost.txt | tr -d ' ')
if [ "$count" = "0" ]; then
    echo "nothing lost to quota -- nothing to re-run"
    exit 0
fi
echo "re-running $count cell(s) lost to quota, one at a time:"
cat /tmp/glyphhunt_lost.txt

while IFS="$(printf '\t')" read -r model level mode trial; do
    [ -z "$model" ] && continue
    echo "--> $model L$level $mode t$trial"
    # Sequential: quota is the constraint here, not CPU.
    $BIN run --only "$model" --levels "$level" --modes "$mode" \
        --trials 1 --trial-base "$trial" --concurrency 1 \
        --pass accuracy --plain || echo "   (failed again -- likely still rate limited)"
done < /tmp/glyphhunt_lost.txt

echo "done. republish with: sh scripts/publish.sh"
