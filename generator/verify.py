"""Human-checkable proof that the glyphs are where ground truth says.

Produces, per level, a contact sheet of all 17 target crops: the frame as a
model sees it, and the same crop amplified against the clean reference. If
the amplified column spells the target word, the ground truth is honest.

Also writes a solution video with every target boxed and labelled.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080
FB = W * H * 3
LABEL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)


def frame_at(path, n):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})",
         "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout[:FB], dtype=np.uint8).reshape(H, W, 3).astype(np.float32)


def contact_sheet(level, gt, pad=34):
    lvl = gt["levels"][str(level)]
    video = ROOT / "videos" / lvl["video"]
    ref = ROOT / "assets" / f"level{level}_reference.mp4"
    has_ref = ref.exists()

    cell, rows = 128, 2 if has_ref else 1
    sheet = Image.new("RGB", (cell * 17, cell * rows + 46), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), f"Level {level} - {lvl['description']}", font=BOLD, fill="black")

    for i, t in enumerate(sorted(lvl["targets"], key=lambda t: t["frame"])):
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
        raw = frame_at(video, t["frame"])[sl]

        im = Image.fromarray(raw.astype(np.uint8)).resize((cell, cell), Image.LANCZOS)
        sheet.paste(im, (i * cell, 30))
        d.text((i * cell + 3, 30 + cell - 15), f"{t['char']} f{t['frame']}",
               font=LABEL, fill="yellow")

        if has_ref:
            diff = np.abs(raw - frame_at(ref, t["frame"])[sl]).mean(axis=2)
            diff = (diff / max(diff.max(), 1e-6) * 255).astype(np.uint8)
            am = Image.fromarray(np.dstack([diff] * 3)).resize((cell, cell), Image.LANCZOS)
            sheet.paste(am, (i * cell, 30 + cell))
            d.text((i * cell + 3, 30 + 2 * cell - 15), t["font"][:16], font=LABEL, fill="cyan")

    out = ROOT / "verification" / f"level{level}_sheet.png"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    return out


def solution_video(level, gt):
    """Re-encode the level with every target boxed, for eyeball verification."""
    lvl = gt["levels"][str(level)]
    src = ROOT / "videos" / lvl["video"]
    out = ROOT / "verification" / f"level{level}_solution.mp4"
    out.parent.mkdir(exist_ok=True)

    draws = []
    for t in lvl["targets"]:
        # Hold the box on screen well past the glyph so it is visible at speed.
        a, b = t["frame"], t["frame"] + 20
        x, y, w, h = t["x"] - 10, t["y"] - 10, t["w"] + 20, t["h"] + 20
        # This ffmpeg is built without libfreetype, so drawtext is unavailable.
        # Boxes carry the localisation; the contact sheet carries the labels.
        en = f"between(n\\,{a}\\,{b})"
        draws.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:color=red@1.0:t=3:enable='{en}'")
        draws.append(f"drawbox=x={x - 4}:y={y - 4}:w={w + 8}:h={h + 8}"
                     f":color=yellow@0.9:t=1:enable='{en}'")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", ",".join(draws), "-c:v", "libx264", "-crf", "18", str(out)],
        check=True)
    return out


if __name__ == "__main__":
    gt = json.loads((ROOT / "ground_truth.json").read_text())
    levels = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "3"])]
    for lvl in levels:
        print("sheet   ", contact_sheet(lvl, gt))
        print("solution", solution_video(lvl, gt))
