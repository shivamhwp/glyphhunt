"""Build all three benchmark videos plus their ground truth.

Usage:  .venv/bin/python generator/main.py [--seed 42] [--levels 1,2,3]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import LEVELS
from fonts import FONT_ASSIGNMENT, TARGET
from montage import build as build_montage
from render import SURVIVAL_FLOOR, measure_survival, render_level

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUT = ROOT / "videos"
MAX_SURVIVAL_ATTEMPTS = 5


def strip(g):
    """Drop the non-serialisable render scratch before writing ground truth."""
    return {k: v for k, v in g.items() if k not in ("font_path", "mask", "pt")}


def build_level(level, shots, seed):
    """Render a level, escalating contrast until every glyph survives encode."""
    OUT.mkdir(exist_ok=True)
    video = OUT / f"level{level}.mp4"
    ref = ASSETS / f"level{level}_reference.mp4"
    scale = 1.0
    base = ASSETS / "base_montage.mp4"

    for attempt in range(1, MAX_SURVIVAL_ATTEMPTS + 1):
        targets, decoys, delta = render_level(level, base, shots, video, seed, scale)

        if not LEVELS[level]["subperceptual"]:
            for t in targets:
                t.pop("mask", None)
            print(f"  L{level}: rendered ({LEVELS[level]['note']})")
            return video, targets, decoys, None

        # The reference is glyph-free but otherwise byte-for-byte the same
        # pipeline, so it only has to be built once.
        if not ref.exists():
            render_level(level, base, shots, ref, seed, scale, embed_glyphs=False)

        surv = measure_survival(video, ref, targets)
        vals = [surv.get(t["frame"], 0.0) for t in targets]
        weakest = min(vals)
        print(f"  L{level} attempt {attempt}: delta={delta:.1f} "
              f"recovered contrast min={weakest:.2f} mean={np.mean(vals):.2f} "
              f"(floor {SURVIVAL_FLOOR})")

        if weakest >= SURVIVAL_FLOOR:
            for t in targets:
                t["recovered_contrast"] = round(surv.get(t["frame"], 0.0), 3)
                t.pop("mask", None)
            return video, targets, decoys, delta

        scale *= 1.6

    print(f"  L{level}: WARNING -- could not get all glyphs above the "
          f"survival floor after {MAX_SURVIVAL_ATTEMPTS} attempts")
    for t in targets:
        t["recovered_contrast"] = round(surv.get(t["frame"], 0.0), 3)
        t.pop("mask", None)
    return video, targets, decoys, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", default="1,2,3")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    shots_path = ASSETS / "shots.json"
    if not shots_path.exists():
        print("building base montage...")
        build_montage(ASSETS / "raw", ASSETS / "base_montage.mp4",
                      shots_path, seed=args.seed)
    shots = json.loads(shots_path.read_text())["shots"]
    print(f"base montage: {len(shots)} shots")

    truth = {"target": TARGET, "seed": args.seed, "fps": 60,
             "width": 1920, "height": 1080, "frames": 1800,
             "fonts": [{"index": p, "char": c, "font": n}
                       for p, c, _, n in FONT_ASSIGNMENT],
             "levels": {}}

    for lvl in levels:
        video, targets, decoys, delta = build_level(lvl, shots, args.seed)
        truth["levels"][str(lvl)] = {
            "video": video.name,
            "description": LEVELS[lvl]["note"],
            "config": {k: v for k, v in LEVELS[lvl].items() if k != "note"},
            "final_delta": delta,
            "targets": [strip(t) for t in targets],
            "decoys": [strip(d) for d in decoys],
        }

    (ROOT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"\nground truth -> ground_truth.json")
    for lvl in levels:
        t = truth["levels"][str(lvl)]
        print(f"  L{lvl}: {len(t['targets'])} targets, {len(t['decoys'])} decoys")


if __name__ == "__main__":
    main()
