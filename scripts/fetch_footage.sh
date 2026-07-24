#!/bin/sh
# Pull a 2-minute section from each of Theo's latest videos.
# We only need ~3 min of source for a 6x-sped 30s montage, so full
# downloads would be wasted bandwidth.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IDS="$ROOT/assets/ids.txt"
OUT="$ROOT/assets/raw"
mkdir -p "$OUT"

ok=0
fail=0
while IFS= read -r id; do
    [ -z "$id" ] && continue
    if [ -f "$OUT/$id.mp4" ]; then
        echo "skip  $id (already present)"
        ok=$((ok + 1))
        continue
    fi
    echo "fetch $id"
    if yt-dlp \
        -f "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]" \
        --download-sections "*300-420" \
        --force-keyframes-at-cuts \
        --no-warnings --no-progress \
        --merge-output-format mp4 \
        -o "$OUT/$id.%(ext)s" \
        "https://www.youtube.com/watch?v=$id" >/dev/null 2>"$OUT/$id.err"; then
        ok=$((ok + 1))
        rm -f "$OUT/$id.err"
    else
        fail=$((fail + 1))
        echo "  FAILED: $(tail -3 "$OUT/$id.err" | tr '\n' ' ')"
    fi
done < "$IDS"

echo "---"
echo "ok=$ok fail=$fail"
ls -la "$OUT"
