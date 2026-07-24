"""Rasterise the 17 target glyphs in their real typefaces for the website.

Zapfino, Papyrus, Herculanum and friends are macOS system faces -- naming
them in CSS would fall back to something generic for almost every visitor,
which would defeat the point of a page about typeface difficulty. So each
glyph is rendered here from the actual font file and shipped as an alpha
PNG. The page uses it as a CSS mask filled with `currentColor`, so a single
asset works in both light and dark themes.
"""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generator"))

from fonts import FONT_ASSIGNMENT  # noqa: E402

CAP = 140      # rendered ink height in px; scaled down by CSS
PAD = 10


def render(letter, font_path, cap=CAP):
    # Binary-search the point size that lands the requested ink height, the
    # same normalisation the generator uses -- point size means very
    # different things across these faces.
    lo, hi = 4, 700
    for _ in range(24):
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bb = f.getbbox(letter)
        if bb is None or (bb[3] - bb[1]) < cap:
            lo = mid + 1
        else:
            hi = mid
    font = ImageFont.truetype(font_path, lo)
    bb = font.getbbox(letter)
    w, h = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])

    img = Image.new("L", (w + PAD * 2, h + PAD * 2), 0)
    ImageDraw.Draw(img).text((PAD - bb[0], PAD - bb[1]), letter, font=font, fill=255)

    # Alpha-only PNG: white pixels, transparency carries the shape.
    rgba = Image.new("RGBA", img.size, (255, 255, 255, 0))
    rgba.putalpha(img)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", optimize=True)
    return {
        "data": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "w": rgba.width,
        "h": rgba.height,
        "aspect": round(rgba.width / rgba.height, 4),
    }


def main():
    out = {}
    total = 0
    for index, letter, path, name in FONT_ASSIGNMENT:
        g = render(letter, path)
        out[str(index)] = {"index": index, "char": letter, "font": name, **g}
        total += len(g["data"])
    dest = ROOT / "site" / "src" / "data" / "glyphs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"rendered {len(out)} glyphs, {total // 1024} KB of data URIs -> {dest.name}")


if __name__ == "__main__":
    main()
