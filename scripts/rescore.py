"""Recover runs that the integrity detector flagged by mistake.

The detector runs inside the harness and zeroes a flagged run's score on the
spot. When a marker turns out to be wrong -- `_sheet.png` matched an agent's
own `suspect_sheet.png` -- that verdict is baked into rows already written,
and the grid keeps producing more until it is restarted.

Every record keeps the offending commands and the raw answer, so the verdict
can be re-derived and the score recomputed offline instead of re-running
hours of work. This mirrors benchmark/src/score.rs exactly; if that file
changes, change this too.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Must match config.rs INTEGRITY_MARKERS.
MARKERS = ("glyphhunt", "ground_truth", "base_montage", "/users/shivam", "developer/t3")

FRAME_TOLERANCE = 2
SPATIAL_TOLERANCE_FRAC = 0.05


def normalize(s):
    return "".join(c for c in str(s).lower() if "a" <= c <= "z")


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_answer(raw):
    """Parse the answer object out of answer.json contents or a final message."""
    if not raw:
        return None
    try:
        o = json.loads(raw)
        if isinstance(o, dict) and ("letters" in o or "word" in o):
            return o
    except json.JSONDecodeError:
        pass
    # Fall back to the last balanced {...} that parses, as the Rust side does.
    starts, spans = [], []
    for i, ch in enumerate(raw):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            spans.append((starts.pop(), i))
    for s, e in reversed(spans):
        try:
            o = json.loads(raw[s : e + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and (o.get("letters") or o.get("word")):
            return o
    return None


def score(answer, truth, level):
    """Recompute a score. Mirrors score.rs."""
    target = truth["target"]
    lvl = truth["levels"][str(level)]
    by_index = {t["index"]: t for t in lvl["targets"]}
    diag = (truth["width"] ** 2 + truth["height"] ** 2) ** 0.5
    tol_px = diag * SPATIAL_TOLERANCE_FRAC

    out = {
        "parsed": False, "word_exact": False, "word_normalized": "",
        "levenshtein_norm": 0.0, "chars_correct": 0, "chars_total": len(target),
        "first_error_index": None, "frames_correct": 0, "spatial_correct": 0,
        "positions": [],
    }
    if not answer:
        return out
    out["parsed"] = True

    letters = answer.get("letters") or []
    from_letters = "".join(normalize(l.get("c", "")) for l in letters)
    from_word = normalize(answer.get("word", ""))
    if len(from_letters) == len(target):
        word = from_letters
    elif from_word:
        word = from_word
    else:
        word = from_letters

    out["word_normalized"] = word
    out["word_exact"] = word == target
    maxlen = max(len(word), len(target), 1)
    out["levenshtein_norm"] = round(1.0 - levenshtein(word, target) / maxlen, 4)

    for i, want in enumerate(target):
        tg = by_index.get(i)
        ps = {"index": i, "expected": want, "got": None,
              "font": (tg or {}).get("font", ""), "char_ok": False,
              "frame_ok": False, "spatial_ok": False,
              "frame_delta": None, "pixel_delta": None}
        if i < len(letters):
            al = letters[i]
            got = normalize(al.get("c", ""))
            ps["got"] = got
            ps["char_ok"] = bool(got) and got[0] == want
            f = al.get("frame")
            if tg and isinstance(f, (int, float)):
                d = abs(int(f) - tg["frame"])
                ps["frame_delta"] = d
                ps["frame_ok"] = d <= FRAME_TOLERANCE
            x, y = al.get("x"), al.get("y")
            if tg and isinstance(x, (int, float)) and isinstance(y, (int, float)):
                cx = tg["x"] + tg["w"] / 2
                cy = tg["y"] + tg["h"] / 2
                d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                ps["pixel_delta"] = round(d, 2)
                # Right coordinates on the wrong frame is a coincidence.
                ps["spatial_ok"] = d <= tol_px and ps["frame_ok"]
        if ps["char_ok"]:
            out["chars_correct"] += 1
        elif out["first_error_index"] is None:
            out["first_error_index"] = i
        out["frames_correct"] += ps["frame_ok"]
        out["spatial_correct"] += ps["spatial_ok"]
        out["positions"].append(ps)
    return out


def real_violation(cmds):
    """Did the command actually reach the project tree?"""
    return any(m in (c or "").lower() for c in cmds for m in MARKERS)


def repair(rows, truth):
    """Return (rows, n_repaired) with mis-flagged runs re-scored."""
    n = 0
    for r in rows:
        if not r.get("integrity_violation"):
            continue
        cmds = (r.get("behavior") or {}).get("integrity_violations") or []
        if real_violation(cmds):
            continue  # genuinely reached the project tree
        # Flagged by a marker we have since retired: restore the run.
        r["integrity_violation"] = False
        r["falsely_flagged"] = True
        r["score"] = score(extract_answer(r.get("raw_answer")), truth, r["level"])
        if r["score"]["parsed"]:
            r["outcome"] = "Scored"
        else:
            # Leaving "Cheated" here would contradict the flag we just cleared.
            # Cheated took precedence over every other outcome in the harness,
            # so the original reason is unrecoverable -- infer it from how the
            # process ended.
            r["outcome"] = "Crashed" if r.get("exit_code") not in (0, None) else "Unparseable"
        n += 1
    return rows, n


if __name__ == "__main__":
    truth = json.loads((ROOT / "ground_truth.json").read_text())
    rows = [json.loads(l) for l in (ROOT / "results/runs.jsonl").read_text().splitlines() if l.strip()]
    rows, n = repair(rows, truth)
    print(f"re-scored {n} falsely flagged run(s) of {len(rows)}")
    for r in rows:
        if r.get("falsely_flagged"):
            s = r["score"]
            print(f"  {r['model']} L{r['level']} {r['mode']} t{r['trial']}: "
                  f"chars {s['chars_correct']}/17 frames {s['frames_correct']} "
                  f"exact {s['word_exact']} word {s['word_normalized']!r}")
