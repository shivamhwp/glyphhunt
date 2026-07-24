"""Embed the target glyphs into the montage and emit exact ground truth.

Three levels, three different skills:

  L1  font/OCR robustness -- glyphs held ~0.5s, large, opaque, but sharing
      the screen with dense decoy text walls in the same mixed-font style.
  L2  temporal assembly -- 5-frame appearances, medium, 70% alpha, spread
      across shots so the word only exists as a time-ordered sequence.
  L3  needle detection -- ONE frame each, sub-perceptual luma delta,
      adversarially placed on the highest-variance patch of that frame,
      plus ~50 decoy flashes with identical statistics. Only the reading
      "theo loves obsidian" means anything, so the model has to use
      semantics to pick its 17 out of the noise.

Frames are streamed through ffmpeg rather than dumped to disk: 1800
1080p PNGs would be several GB per level.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fonts import FONT_ASSIGNMENT, TARGET

W, H, FPS = 1920, 1080, 60
N_FRAMES = 1800

# No synthetic decoy text walls: the source footage is already saturated with
# real text -- terminals, code editors, benchmark tables, docs -- which is a
# far better distractor than anything we could paint on top.
LEVELS = {
    1: dict(hold=30, cap_height=78, alpha=1.00, subperceptual=False,
            decoys=0,
            note="held ~0.5s, large, opaque, amid the footage's own text"),
    2: dict(hold=5, cap_height=34, alpha=0.70, subperceptual=False,
            decoys=12,
            note="5 frames, medium, 70% alpha, split across shots"),
    3: dict(hold=1, cap_height=30, alpha=None, subperceptual=True,
            decoys=50,
            note="1 frame, sub-perceptual luma delta, adversarial placement"),
}

# Starting luma delta for L3, in 0-255 units. Raised automatically until the
# glyph survives H.264 quantisation -- see verify_survival().
L3_BASE_DELTA = 8.0


def render_glyph(letter, font_path, cap_height):
    """Render one glyph to a tight alpha mask, normalised by cap height."""
    # Binary-search a point size that yields the requested ink height.
    lo, hi = 4, 400
    for _ in range(24):
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bb = f.getbbox(letter)
        if bb is None or (bb[3] - bb[1]) < cap_height:
            lo = mid + 1
        else:
            hi = mid
    font = ImageFont.truetype(font_path, lo)
    bb = font.getbbox(letter)
    w, h = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
    img = Image.new("L", (w + 8, h + 8), 0)
    ImageDraw.Draw(img).text((4 - bb[0], 4 - bb[1]), letter, font=font, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0, lo


def variance_map(frame, patch=64):
    """Coarse local-variance map; used to hide glyphs in the busiest areas."""
    g = frame.astype(np.float32).mean(axis=2)
    ph, pw = g.shape[0] // patch, g.shape[1] // patch
    g = g[:ph * patch, :pw * patch].reshape(ph, patch, pw, patch)
    return g.std(axis=(1, 3))


def motion_map(frame, prev, patch=64):
    """Coarse per-patch temporal motion between two consecutive frames."""
    d = np.abs(frame.astype(np.float32) - prev.astype(np.float32)).mean(axis=2)
    ph, pw = d.shape[0] // patch, d.shape[1] // patch
    d = d[:ph * patch, :pw * patch].reshape(ph, patch, pw, patch)
    return d.mean(axis=(1, 3))


def pick_position(frame, gw, gh, rng, adversarial, prev=None):
    """Choose where a glyph goes.

    Adversarial placement targets patches that are visually busy but
    temporally still: high spatial variance hides the glyph from anything
    that just looks at the frame, while low motion means frame differencing
    can still recover it. Picking purely on spatial variance -- the obvious
    reading of "adversarial" -- lands the glyph in the highest-motion region
    of the frame, where its signal is buried under scene churn and no
    technique recovers it. That makes the level unsolvable rather than hard.
    """
    if not adversarial:
        return (int(rng.integers(40, W - gw - 40)), int(rng.integers(40, H - gh - 40)))

    vm = variance_map(frame)
    mm = motion_map(frame, prev) if prev is not None else np.zeros_like(vm)
    score = vm / (1.0 + 3.0 * mm)

    flat = score.flatten()
    k = max(1, int(flat.size * 0.08))
    top = np.argpartition(flat, -k)[-k:]
    choice = top[rng.integers(0, len(top))]
    py, px = divmod(int(choice), score.shape[1])
    # Jitter stays inside the chosen patch so we don't drift into a noisy one.
    x = np.clip(px * 64 + rng.integers(-8, 8), 20, W - gw - 20)
    y = np.clip(py * 64 + rng.integers(-8, 8), 20, H - gh - 20)
    return int(x), int(y)


def composite(frame, mask, x, y, alpha=None, delta=None):
    """Alpha-blend a white glyph, or add a sub-perceptual luma bump."""
    gh, gw = mask.shape
    region = frame[y:y + gh, x:x + gw].astype(np.float32)
    if region.shape[:2] != mask.shape:
        return frame
    if delta is not None:
        # Push luma away from the local mean so the glyph survives on both
        # bright and dark backgrounds.
        sign = -1.0 if region.mean() > 128 else 1.0
        region += (mask * delta * sign)[:, :, None]
    else:
        region = region * (1 - (mask * alpha)[:, :, None]) + \
                 255.0 * (mask * alpha)[:, :, None]
    frame[y:y + gh, x:x + gw] = np.clip(region, 0, 255).astype(np.uint8)
    return frame


def plan(level, shots, rng):
    """Decide frame + identity for every target and decoy glyph."""
    cfg = LEVELS[level]
    hold = cfg["hold"]

    # For short holds, keep glyphs strictly mid-shot -- never within 3 frames
    # of a cut. That is the deliberate escape hatch: inside a single shot,
    # consecutive frames are similar enough that differencing can surface the
    # glyph, so a model that segments shots first has a real way in.
    #
    # A long hold (L1) is longer than a shot, so it cannot avoid crossing a
    # cut. That is fine: those glyphs are plainly visible anyway and the
    # escape hatch is irrelevant to them.
    safe = []
    for s in shots:
        a, b = s["out_start_frame"] + 3, s["out_end_frame"] - 3 - hold
        if b > a:
            safe.append((a, b, s["index"]))

    if len(safe) < 17:
        span = shots[-1]["out_end_frame"]
        safe = [(f, f, -1) for f in range(30, span - hold - 30, 6)]
    if len(safe) < 17:
        raise SystemExit(f"not enough placement windows for hold={hold}")

    # Spread the 17 targets evenly through the timeline, in order.
    chunks = np.array_split(np.arange(len(safe)), 17)
    targets = []
    for i, (pos, letter, fpath, fname) in enumerate(FONT_ASSIGNMENT):
        a, b, shot = safe[chunks[i][rng.integers(0, len(chunks[i]))]]
        targets.append(dict(index=pos, char=letter, font=fname,
                            font_path=fpath, shot=shot,
                            frame=int(rng.integers(a, b + 1))))
    targets.sort(key=lambda t: t["frame"])
    for rank, t in enumerate(targets):
        t["temporal_rank"] = rank
    # Chunks partition the shot list in time order, so letter i always lands
    # in a later shot than letter i-1 and reading by frame spells the word.
    # If that ever stops holding the benchmark is silently wrong, so assert.
    assert [t["char"] for t in targets] == list(TARGET), \
        "temporal order does not spell the target"
    assert len({t["frame"] for t in targets}) == 17, "two targets share a frame"

    # Decoys: same size/alpha/font distribution, but random letters that
    # spell nothing. Font identity is therefore never a tell.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    decoys = []
    for _ in range(cfg["decoys"]):
        a, b, shot = safe[rng.integers(0, len(safe))]
        pos = int(rng.integers(0, 17))
        decoys.append(dict(char=alphabet[rng.integers(0, 26)],
                           font=FONT_ASSIGNMENT[pos][3],
                           font_path=FONT_ASSIGNMENT[pos][2],
                           shot=shot, frame=int(rng.integers(a, b + 1))))
    return targets, decoys
