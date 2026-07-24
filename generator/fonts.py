"""The 17 typefaces, one per glyph position in the target word.

Chosen so that no two positions share a face and repeated letters
(o appears 3x, e/s/i twice each) look nothing alike. A few are
caps-only faces (Copperplate, Herculanum, Academy Engraved) which
render lowercase as small caps -- still legible as the letter, and
the inconsistency is part of the difficulty.
"""

TARGET = "theolovesobsidian"

SYS = "/System/Library/Fonts"
SUP = f"{SYS}/Supplemental"

# (position, letter, font file, human-readable name)
FONT_ASSIGNMENT = [
    (0,  "t", f"{SUP}/Zapfino.ttf",                 "Zapfino"),
    (1,  "h", f"{SUP}/Papyrus.ttc",                 "Papyrus"),
    (2,  "e", f"{SUP}/Herculanum.ttf",              "Herculanum"),
    (3,  "o", f"{SUP}/Chalkduster.ttf",             "Chalkduster"),
    (4,  "l", f"{SUP}/Bodoni 72.ttc",               "Bodoni 72"),
    (5,  "o", f"{SUP}/Didot.ttc",                   "Didot"),
    (6,  "v", f"{SUP}/Copperplate.ttc",             "Copperplate"),
    (7,  "e", f"{SUP}/Luminari.ttf",                "Luminari"),
    (8,  "s", f"{SUP}/Trattatello.ttf",             "Trattatello"),
    (9,  "o", f"{SUP}/SnellRoundhand.ttc",          "Snell Roundhand"),
    (10, "b", f"{SUP}/Brush Script.ttf",            "Brush Script"),
    (11, "s", f"{SYS}/MarkerFelt.ttc",              "Marker Felt"),
    (12, "i", f"{SYS}/Noteworthy.ttc",              "Noteworthy"),
    (13, "d", f"{SUP}/AmericanTypewriter.ttc",      "American Typewriter"),
    (14, "i", f"{SUP}/Impact.ttf",                  "Impact"),
    (15, "a", f"{SUP}/Phosphate.ttc",               "Phosphate"),
    (16, "n", f"{SUP}/Rockwell.ttc",                "Rockwell"),
]

# Decoy glyphs draw from a disjoint pool so font identity can never be
# used as a shortcut to separate targets from noise... except it must
# NOT be disjoint, or font becomes the tell. Decoys reuse the same 17.
DECOY_FONTS = [f for _, _, f, _ in FONT_ASSIGNMENT]

assert len(FONT_ASSIGNMENT) == 17
assert "".join(l for _, l, _, _ in FONT_ASSIGNMENT) == TARGET
assert len({f for _, _, f, _ in FONT_ASSIGNMENT}) == 17, "fonts must be distinct"
