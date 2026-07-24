"""Build the chaotic base montage the glyphs get hidden in.

Takes the downloaded Theo clips and cuts them into a 30s 1080p60 reel
with ~0.3s shots at 6x speed. The rapid cutting is deliberate: it means
consecutive frames differ enormously, which defeats naive frame-diffing
as a way to spot the hidden glyphs. Shot boundaries are recorded so the
glyph placer can keep every glyph mid-shot, preserving the intended
solution path (detect shots -> diff within a shot).
"""

import json
import random
import subprocess
from pathlib import Path

FPS = 60
DURATION = 30.0
W, H = 1920, 1080
SPEEDUP = 6.0
SHOT_OUT = 0.30           # each shot lasts this long in the output
SHOT_SRC = SHOT_OUT * SPEEDUP  # ...so it consumes this much source


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def build(raw_dir, out_path, shots_path, seed=42):
    rng = random.Random(seed)
    clips = sorted(Path(raw_dir).glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no source clips in {raw_dir}")

    durations = {c: probe_duration(c) for c in clips}
    n_shots = int(DURATION / SHOT_OUT)

    segs, shots, t = [], [], 0.0
    prev_clip = None
    used = {c: [] for c in clips}  # source windows already spent, per clip
    for i in range(n_shots):
        # Never cut from a clip to itself: back-to-back shots from the same
        # static screen-share look near-identical, which would hand the model
        # an easy frame-diff. Consecutive shots must jump sources.
        pool = [c for c in clips if c is not prev_clip] or clips
        clip = rng.choice(pool)
        max_start = max(0.0, durations[clip] - SHOT_SRC - 0.5)
        # Pick a window that isn't near one already used from this clip, so we
        # don't replay the same few seconds of footage over and over.
        start = rng.uniform(0.0, max_start)
        for _ in range(40):
            if all(abs(start - u) > SHOT_SRC * 2 for u in used[clip]):
                break
            start = rng.uniform(0.0, max_start)
        used[clip].append(start)
        prev_clip = clip
        segs.append((clip, start))
        shots.append({
            "index": i,
            "src": clip.name,
            "src_start": round(start, 3),
            "out_start_frame": int(round(t * FPS)),
            "out_end_frame": int(round((t + SHOT_OUT) * FPS)) - 1,
        })
        t += SHOT_OUT

    # One ffmpeg invocation: trim each source window, speed it up, scale to
    # a common size, then concat. Doing it in a single graph avoids writing
    # ~100 intermediate files.
    inputs, filters, labels = [], [], []
    for i, (clip, start) in enumerate(segs):
        inputs += ["-ss", f"{start:.3f}", "-t", f"{SHOT_SRC:.3f}", "-i", str(clip)]
        filters.append(
            f"[{i}:v]setpts=PTS/{SPEEDUP},fps={FPS},"
            f"scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setsar=1[v{i}]"
        )
        labels.append(f"[v{i}]")
    filters.append(f"{''.join(labels)}concat=n={len(segs)}:v=1:a=0[out]")

    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + inputs +
           ["-filter_complex", ";".join(filters), "-map", "[out]",
            "-t", str(DURATION),
            # near-lossless: the sub-perceptual layer added later must survive
            "-c:v", "libx264", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", str(out_path)])
    subprocess.run(cmd, check=True)

    Path(shots_path).write_text(json.dumps(
        {"fps": FPS, "width": W, "height": H, "duration": DURATION,
         "speedup": SPEEDUP, "seed": seed, "shots": shots}, indent=2))
    return shots


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    shots = build(root / "assets/raw",
                  root / "assets/base_montage.mp4",
                  root / "assets/shots.json",
                  seed=int(sys.argv[1]) if len(sys.argv) > 1 else 42)
    print(f"built {len(shots)} shots -> assets/base_montage.mp4")
