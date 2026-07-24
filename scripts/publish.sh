#!/bin/sh
# Regenerate site data from the current results, rebuild, and deploy.
# Safe to run repeatedly while the grid is still going -- the page shows a
# RUNNING badge and its own progress count until all 108 rows land.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/build_site_data.py
cd site
npm run build --silent
wrangler pages deploy dist \
    --project-name glyphhunt \
    --branch main \
    --commit-dirty=true 2>&1 | tail -3

echo "live: https://glyphhunt.pages.dev"
