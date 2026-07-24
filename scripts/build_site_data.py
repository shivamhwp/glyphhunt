"""Aggregate results/runs.jsonl into the JSON the site renders.

Runs flagged as integrity violations are counted and shown, never scored --
they say something about the model but nothing about the task.
"""

import hashlib
import json
import platform
import subprocess
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "src" / "data" / "results.json"

LEVEL_NAMES = {
    1: ("L1", "font / OCR robustness", "held ~0.5s, 78px, opaque, amid the footage's own text"),
    2: ("L2", "temporal assembly", "5 frames, 34px, 70% alpha, split across shots, 12 decoys"),
    3: ("L3", "needle detection", "1 frame, sub-perceptual luma delta, adversarial placement, 50 decoys"),
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 2) if xs else 0.0


def sh(cmd, default="unknown"):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else default
    except Exception:
        return default


def device_info():
    """Machine and toolchain the grid actually ran on."""
    mem = sh("sysctl -n hw.memsize", "0")
    try:
        mem_gb = f"{int(mem) / 1024**3:.0f} GB"
    except ValueError:
        mem_gb = "unknown"
    return {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "arch": platform.machine(),
        "cores_physical": sh("sysctl -n hw.physicalcpu", "?"),
        "cores_logical": sh("sysctl -n hw.logicalcpu", "?"),
        "memory": mem_gb,
        "os": f"macOS {platform.mac_ver()[0]}",
        "kernel": platform.release(),
        "ffmpeg": sh("ffmpeg -version | head -1 | awk '{print $3}'"),
        "python": platform.python_version(),
        "rust": sh("cargo --version | awk '{print $2}'"),
        "node": sh("node --version"),
        "codex_cli": sh("codex --version | awk '{print $2}'"),
        "claude_cli": sh("claude --version | awk '{print $1}'"),
        "concurrency": 8,
        "timeout_s": 1800,
        "codex_profile": "codex-p",
        "run_root": "/private/tmp/vidtask",
    }


def load_rows():
    p = ROOT / "results" / "runs.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


RATE_LIMIT_MARKERS = ("session limit", "usage limit", "rate limit", "resets ")


def rate_limited(r):
    """Did this run die because the account ran out of quota?

    A plan limit is a fact about the subscription, not about the model. These
    runs are excluded from every score rather than counted as failures, and
    re-run once quota returns.
    """
    if r["outcome"] != "Crashed":
        return False
    tail = (r.get("final_message_tail") or "").lower()
    return any(m in tail for m in RATE_LIMIT_MARKERS)


def excluded(r):
    """Runs that say nothing about model ability."""
    return bool(r.get("integrity_violation")) or rate_limited(r)


def effective_outcome(r):
    """Normalise the recorded outcome.

    The harness labels a run TimedOut whenever the process was still alive at
    the wall-clock limit -- even if it had already written a complete
    answer.json. That conflates "produced nothing before the deadline" with
    "answered, but slowly", and counting the second as a failure would inflate
    the failure rate. If an answer parsed, the run gets judged on that answer;
    the timeout remains visible as its own flag.
    """
    if r.get("integrity_violation"):
        return "Cheated"
    if rate_limited(r):
        return "RateLimited"
    if (r.get("score") or {}).get("parsed"):
        return "Scored"
    return r["outcome"]


def agg(rows):
    """Aggregate a set of runs. Cheated runs are excluded from every score."""
    valid = [r for r in rows if not excluded(r)]
    n = len(valid)
    return {
        "runs": len(rows),
        "valid": n,
        "cheated": sum(1 for r in rows if r.get("integrity_violation")),
        "rate_limited": sum(1 for r in rows if rate_limited(r)),
        "failed": sum(1 for r in valid if effective_outcome(r) != "Scored"),
        "timed_out": sum(1 for r in valid if r["outcome"] == "TimedOut"),
        "exact": sum(1 for r in valid if r["score"].get("word_exact")),
        "chars": mean([r["score"].get("chars_correct") for r in valid]),
        "best_chars": max([r["score"].get("chars_correct", 0) for r in valid], default=0),
        "frames": mean([r["score"].get("frames_correct") for r in valid]),
        "spatial": mean([r["score"].get("spatial_correct") for r in valid]),
        "lev": mean([r["score"].get("levenshtein_norm") for r in valid]),
        "wall_s": mean([r["wall_ms"] / 1000 for r in valid]),
        "cost": round(sum(r["usage"].get("cost_usd") or 0 for r in valid), 4),
        "out_tokens": mean([r["usage"].get("output_tokens") for r in valid]),
        "shell": mean([r["behavior"].get("shell_commands") for r in valid]),
        "ffmpeg": mean([r["behavior"].get("ffmpeg_invocations") for r in valid]),
        "python": mean([r["behavior"].get("python_invocations") for r in valid]),
        "images": mean([r["behavior"].get("images_read") for r in valid]),
        "diffed": sum(1 for r in valid if r["behavior"].get("used_frame_diff")),
    }


def main():
    rows = load_rows()
    models = sorted({r["model"] for r in rows})
    levels = sorted({r["level"] for r in rows})

    leaderboard = sorted(
        ({"model": m, **agg([r for r in rows if r["model"] == m])} for m in models),
        key=lambda a: (-(a["exact"] / max(a["valid"], 1)), -a["chars"]),
    )

    by_level = {
        str(l): {m: agg([r for r in rows if r["model"] == m and r["level"] == l]) for m in models}
        for l in levels
    }
    by_mode = {
        mode: {m: agg([r for r in rows if r["model"] == m and r["mode"] == mode]) for m in models}
        for mode in ("Blind", "Hinted")
    }

    # Per-typeface: which glyph positions survive, across all valid runs.
    fonts = {}
    for r in rows:
        if excluded(r):
            continue
        for p in r["score"].get("positions") or []:
            k = (p["index"], p.get("font") or "?")
            e = fonts.setdefault(k, {"ok": 0, "total": 0, "expected": p.get("expected", "")})
            e["total"] += 1
            e["ok"] += 1 if p.get("char_ok") else 0
    typefaces = sorted(
        (
            {"index": i, "font": f, "char": v["expected"], "ok": v["ok"], "total": v["total"],
             "pct": round(100 * v["ok"] / max(v["total"], 1), 1)}
            for (i, f), v in fonts.items()
        ),
        key=lambda d: d["pct"],
    )

    commitment = ""
    cp = ROOT / "COMMITMENT.txt"
    if cp.exists():
        commitment = cp.read_text().split("\n")[0].strip()

    gt = ROOT / "ground_truth.json"
    complete = len(rows) >= 108
    truth = json.loads(gt.read_text()) if gt.exists() else {}

    data = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%b %d, %Y, %I:%M %p UTC"),
            "target": truth.get("target", "theolovesobsidian"),
            "seed_committed": commitment,
            # The seed stays withheld until the grid finishes, so a reader
            # cannot regenerate the video and back out the answers mid-run.
            "seed": truth.get("seed") if complete else None,
            "machine": f"{platform.machine()} · macOS {platform.mac_ver()[0]}",
            "device": device_info(),
            "total_runs": len(rows),
            "planned_runs": 108,
            "complete": complete,
            "models": len(models),
            "cheated": sum(1 for r in rows if r.get("integrity_violation")),
        "rate_limited": sum(1 for r in rows if rate_limited(r)),
            "video": {"duration_s": 30, "fps": 60, "frames": 1800, "res": "1920x1080",
                      "glyphs": 17, "typefaces": 17},
        },
        "levels": [
            {"level": l, "short": LEVEL_NAMES[l][0], "name": LEVEL_NAMES[l][1],
             "detail": LEVEL_NAMES[l][2]}
            for l in sorted(LEVEL_NAMES)
        ],
        "overall": agg(rows),
        "leaderboard": leaderboard,
        "by_level": by_level,
        "by_mode": by_mode,
        "typefaces": typefaces,
        "runs": [
            {
                "model": r["model"], "level": r["level"], "mode": r["mode"],
                "trial": r["trial"], "outcome": effective_outcome(r),
                "timed_out": r["outcome"] == "TimedOut",
                "cheated": bool(r.get("integrity_violation")),
                "rate_limited": rate_limited(r),
                "exact": bool(r["score"].get("word_exact")),
                "chars": r["score"].get("chars_correct", 0),
                "frames": r["score"].get("frames_correct", 0),
                "spatial": r["score"].get("spatial_correct", 0),
                "answer": (r["score"].get("word_normalized") or "")[:40],
                "wall_s": round(r["wall_ms"] / 1000),
                "shell": r["behavior"].get("shell_commands", 0),
                "images": r["behavior"].get("images_read", 0),
                "diffed": bool(r["behavior"].get("used_frame_diff")),
            }
            for r in sorted(rows, key=lambda r: (r["model"], r["level"], r["mode"], r["trial"]))
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"{len(rows)} runs -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
