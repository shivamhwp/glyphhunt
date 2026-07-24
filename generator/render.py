"""Stream the montage through numpy, stamp glyphs in, re-encode, verify.

The sub-perceptual level is the delicate one. An 8/255 luma bump on a
single frame is exactly the kind of signal H.264 throws away, so after
encoding we decode the result back and measure how much of each glyph
actually survived. Any glyph whose recovered contrast falls under the
detection floor gets its delta raised and the whole level is re-rendered.
Without this loop L3 would be unsolvable for reasons that have nothing
to do with model capability.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from embed import (FPS, H, L3_BASE_DELTA, LEVELS, N_FRAMES, W, composite,
                   pick_position, plan, render_glyph)
from fonts import TARGET

ROOT = Path(__file__).resolve().parent.parent
FRAME_BYTES = W * H * 3

# A glyph counts as "survivable" if, after encode, the mean luma inside the
# glyph mask differs from its immediate surround by at least this much.
SURVIVAL_FLOOR = 2.2


def decode_stream(path):
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 4)
    while True:
        buf = p.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
    p.stdout.close()
    p.wait()


def encode_stream(path, keyframes):
    """Encoder that forces an I-frame on every glyph-bearing frame."""
    expr = "+".join(f"eq(n,{f})" for f in sorted(keyframes)) or "0"
    return subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "12",
         "-x264-params", f"scenecut=0:keyint=60:min-keyint=1",
         "-force_key_frames", f"expr:{expr}",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, bufsize=FRAME_BYTES * 4)


def render_level(level, base, shots, out_path, seed, delta_scale=1.0,
                 embed_glyphs=True):
    cfg = LEVELS[level]
    rng = np.random.default_rng(seed * 100 + level)
    targets, decoys = plan(level, shots, rng)

    # Pre-render every mask once.
    for g in targets + decoys:
        g["mask"], g["pt"] = render_glyph(g["char"], g["font_path"], cfg["cap_height"])

    by_frame = {}
    for g in targets:
        by_frame.setdefault(g["frame"], []).append(("target", g))
    for g in decoys:
        by_frame.setdefault(g["frame"], []).append(("decoy", g))

    hold = cfg["hold"]
    delta = L3_BASE_DELTA * delta_scale if cfg["subperceptual"] else None

    # Only the sub-perceptual level needs forced keyframes. There, a glyph
    # riding on a P-frame gets quantised away; on an I-frame it survives.
    # Forcing them on L1/L2 would just bloat the files for no benefit.
    kf = set()
    if cfg["subperceptual"]:
        for f in by_frame:
            kf.update(range(f, min(f + hold, N_FRAMES)))

    enc = encode_stream(out_path, kf)
    active = []  # (kind, glyph, frames_remaining)
    prev = None
    for n, frame in enumerate(decode_stream(base)):
        clean = frame.copy() if by_frame.get(n) else None
        for kind, g in by_frame.get(n, []):
            gh, gw = g["mask"].shape
            x, y = pick_position(frame, gw, gh, rng,
                                 adversarial=cfg["subperceptual"], prev=prev)
            g.update(x=int(x), y=int(y), w=int(gw), h=int(gh))
            active.append([kind, g, hold])
        for item in active:
            g = item[1]
            # The reference pass walks the identical rng path and produces the
            # identical encode, but skips the paint -- so diffing the two
            # isolates exactly the signal that survived compression.
            if embed_glyphs:
                composite(frame, g["mask"], g["x"], g["y"],
                          alpha=cfg["alpha"], delta=delta)
            item[2] -= 1
        active = [a for a in active if a[2] > 0]
        # Motion is measured against the clean previous frame, never against
        # one we have already painted on.
        prev = clean if clean is not None else frame
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()

    return targets, decoys, delta


def measure_survival(path, ref_path, glyphs):
    """How much of each glyph actually survived H.264, in 0-255 luma units.

    Diffs the embedded encode against the glyph-free reference encode and
    averages the difference over the glyph's own ink pixels. Comparing a
    glyph's bounding box against its surroundings does not work: thin
    letterforms like `i` and `l` are mostly background inside their own box,
    and a busy backdrop swamps the signal in both directions.
    """
    want = {g["frame"]: g for g in glyphs}
    results = {}
    ref = decode_stream(ref_path)
    for n, frame in enumerate(decode_stream(path)):
        try:
            rframe = next(ref)
        except StopIteration:
            break
        g = want.get(n)
        if not g:
            continue
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        a = frame[y:y + h, x:x + w].astype(np.float32).mean(axis=2)
        b = rframe[y:y + h, x:x + w].astype(np.float32).mean(axis=2)
        mask = g["mask"]
        if mask.shape != a.shape:
            mask = mask[:a.shape[0], :a.shape[1]]
        ink = mask > 0.5
        if not ink.any():
            results[n] = 0.0
            continue
        results[n] = float(np.abs(a - b)[ink].mean())
    return results
