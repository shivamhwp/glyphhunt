# glyphhunt

A benchmark for one narrow question: **can an agentic coding model find a word
hidden inside a video?**

The word is `theolovesobsidian`. Each of its 17 letters is drawn in a
different typeface and planted at a different moment in a 30-second, 1080p60
montage (1800 frames) cut from [@t3dotgg](https://youtube.com/@t3dotgg)'s
recent uploads at 6x speed. Reading the letters in temporal order spells the
word.

Models are handed nothing but a path to `clip.mp4` and a shell. No frames are
pre-extracted, no tools are provided. How to look at a video is part of the
test.

## The three levels

Each level tests a different skill, so scores are not comparable across them.

| | what it tests | how the glyphs are hidden |
|---|---|---|
| **L1** | font / OCR robustness | held ~0.5s, 78px, opaque, competing with the dense real text in the footage (terminals, code, benchmark tables) |
| **L2** | temporal assembly | 5 frames, 34px, 70% alpha, scattered across shots — the word exists only as a time-ordered sequence, plus 12 decoy glyphs |
| **L3** | needle detection | **one frame each** (16ms), sub-perceptual luma delta, adversarially placed, plus 50 decoy flashes |

### Why L3 is hard but not impossible

L3 defeats each obvious strategy:

- **Sampling at 1–2 fps** misses a single-frame flash ~97% of the time.
- **Looking at the whole frame** fails — the glyph survives encoding at only
  ~6/255 luma contrast and is invisible to the eye.
- **Naive frame differencing** drowns: at 6x speed with ~0.3s cuts,
  consecutive frames already differ enormously.
- **Plain OCR** finds the footage's own text, not the glyph.

But it stays solvable. Glyphs are placed mid-shot, never on a cut, in patches
chosen for **high spatial variance but low temporal motion** — visually busy
enough to hide in, still enough that *local* windowed differencing recovers
them. Measured local SNR is 2.75–10.5 across all 17 glyphs, so the intended
path (segment shots → difference within a shot → crop → read → use semantics
to separate targets from decoys) genuinely works.

Placing glyphs purely on spatial variance — the obvious reading of
"adversarial" — puts them in the highest-motion regions, where nothing
recovers them. That version was built first and measured at 5/17 findable;
the motion-aware placement fixed it to 17/17.

### Encoding survival

An 8/255 luma bump on one frame is exactly what H.264 discards. The generator
therefore forces a keyframe on every glyph-bearing frame, encodes at CRF 12,
then **decodes the result back and diffs it against an identically-encoded
glyph-free reference**, measuring recovered contrast over each glyph's own ink
pixels. If any glyph falls below the detection floor the delta is raised and
the level re-renders. Without that loop L3 would be unsolvable for reasons
having nothing to do with model capability.

## The grid

Six configurations × 3 levels × 2 prompt modes (blind / hinted) × 3 trials.

`gpt-5.6` and `gpt-5.5-codex` return `400 invalid_request_error` on this
account, so the 5.6 line is `gpt-5.6-sol`.

| harness | model | effort |
|---|---|---|
| Claude Code | `opus` (Opus 5) | — |
| Claude Code | `fable` (Fable 5) | — |
| Codex | `gpt-5.6-sol` | high, medium |
| Codex | `gpt-5.5` | high, medium |

**Blind** tells the model only that something is hidden. **Hinted** gives the
length, the different-typeface rule, the temporal-order rule, and warns about
faintness and decoys.

## Scoring

Word accuracy, temporal localisation and spatial localisation are kept
separate — a model that reads the word but cannot say where it saw it is
telling us something different from one that localises glyphs it cannot read.

- **word** — exact match, normalised Levenshtein, per-position correctness
- **temporal** — reported frame within ±2 of truth
- **spatial** — reported centre within 5% of the frame diagonal *and* on the
  right frame (right coordinates on the wrong frame is a coincidence)
- **per-typeface** — which of the 17 faces broke the most runs
- **behaviour** — shell commands, ffmpeg invocations, sampling fps chosen,
  whether the model reached for frame differencing or contrast stretching,
  images read, turns, tokens, cost

## Running it

```sh
python3 -m venv .venv && .venv/bin/pip install pillow numpy
sh scripts/fetch_footage.sh          # pulls 2-min sections from 10 videos
.venv/bin/python generator/main.py   # builds the 3 levels + ground truth
.venv/bin/python generator/verify.py # contact sheets proving ground truth

cd benchmark && cargo build --release && cd ..

# accuracy pass: wide and parallel
./benchmark/target/release/glyphhunt run --trials 3 --concurrency 8

# latency pass: strictly sequential, or the timings mean nothing
./benchmark/target/release/glyphhunt run --trials 1 --concurrency 1 \
    --pass latency --levels 2

./benchmark/target/release/glyphhunt report
```

## Isolation, and why it is not optional

The first grid attempt was invalid. Run directories sat at
`<project>/results/runs/<id>`, and agents walked `../../..`, found
`generator/`, read the source, and re-ran the deterministic generator with the
recorded seed to reproduce exact glyph coordinates. One scored a flawless
**blind L3** — the hardest cell in the grid — without analysing a single
frame. Three of the first seven runs did this.

Three independent defences now apply:

1. **Distance** — run directories live at `/private/tmp/vidtask/<id>`, outside
   the project tree, containing nothing but `clip.mp4`.
2. **Rule** — both prompts forbid reading anything outside the working
   directory, so leaving it is a stated violation rather than fair play.
3. **Detection** — every shell command is scanned for markers that can only
   come from the project tree. A hit sets `integrity_violation`, forces the
   `Cheated` outcome and zeroes the score, so a compromised run can never be
   silently counted.

The answer key and verification sheets are withheld until runs finish; see
`COMMITMENT.txt`.

## Verifying the ground truth

`verification/levelN_sheet.png` shows every target glyph twice: the crop as a
model sees it, and the same crop amplified against the clean reference. On
L3 the top row looks like ordinary video and the bottom row spells
`theolovesobsidian` in 17 typefaces.

## Caveats

- Both harnesses load their own global instruction files, which is a real
  confound: whatever `~/.claude/CLAUDE.md` and the Codex equivalent say is
  part of each model's context and is not held constant between them. One
  specific hazard is closed explicitly — both prompts forbid delegating to
  another model or CLI, because a global instruction like "use codex for
  subagents" would otherwise let a Claude run be partly executed by a GPT
  model and make the comparison meaningless.
- The downloaded footage is not redistributed. `scripts/fetch_footage.sh`
  rebuilds it, but YouTube re-encodes and re-uploads mean a rebuild is not
  guaranteed byte-identical, so absolute numbers are only comparable within a
  single generated set. `ground_truth.json` records the seed.
- Latency is only reported from the sequential pass. Rows from the parallel
  pass carry `concurrent_runs` and `load_avg_1m` so contended timings can be
  identified rather than silently trusted.
