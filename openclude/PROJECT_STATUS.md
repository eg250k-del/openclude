# PROJECT STATUS

**openclude** — AI animation production machine
`story → script → voice → shots → film`

Last updated: 2026-09-27
Verified: **217 tests passing, 4 skipped** (skips need ffmpeg, absent here)
Code: **3,340 source lines / 2,793 test lines**

---

## If you are opening a new session, do this first

```
cd G:\opencode\openclude
C:\Users\Pc\AppData\Local\Python\bin\python.exe -m pytest -q
C:\Users\Pc\AppData\Local\Python\bin\python.exe -m openclude.cli status
```

If both are green, read `src/openclude/schema.py` then this file's
**Decisions** and **Next** sections. Do not re-derive what is already decided.

> **Note on Python:** the interpreter on `PATH`
> (`AppData\Local\Programs\Python\Python314\python.exe`) is **broken** — its
> `Lib\encodings` directory is missing, so it cannot even start. Use
> `AppData\Local\Python\bin\python.exe` (3.14.7) instead. **git is not
> installed on this machine**, so nothing is committed yet.

---

## 1. What this is

A machine that turns a written story into a finished animated film with no
human in the loop between stages. The user's brief asked for 20-minute to
2-hour films on SaladCloud with a rented RTX 4090.

The hard requirements from the brief, and where each one is met:

| Requirement (from BRIEF.md) | Where it lives | Status |
|---|---|---|
| One failed scene is retried, not the film | `state.py` + `runner.py` | done, tested |
| No multi-hour batch generations | `schema.py` splits any shot over ~5s | done, tested |
| Check VRAM per workflow (24GB) | `retry.py` degradation ladder | done, tested |
| Separate repo / checkpoints / outputs / cache | `storage.py` `Layout` | done (R2, untested live) |
| Audio duration determines shot length | `audio.py` measures, never estimates | done, tested |
| Character consistency | `image_refs` + `image_start`/`image_end` | wired, no GPU proof |
| Scene continuity | `resolve_continuity()` chains final frames | done, tested |

---

## 2. The analysis that produced the decisions

Four source repositories were read at the code level, not from their READMEs.
That was the right call — every one of them claims more than it does.

### What was actually found

**A) `jeffmeloy/WanScript2Movie`** — 1,849 lines, one file, no network calls at
all. It is a graph emitter, not a pipeline: it reads a JSON of pre-written
prompts and writes a ComfyUI UI-format workflow file. It does not execute
anything. Two real problems: `FIRST_SAMPLER_NOISE_SEED = -1` makes every run
different, and the "final combined video" node is wired to the last turn only,
so 60 shots produce 60 separate files and no film. Its genuinely useful part:
correct Wan 2.2 A14B settings for 24GB (GGUF Q8_0, 1280×720, two-pass
high/low noise, steps 12/12, shift 8.0).

**B) `lilinsong1/comfyui-wan-video-pipeline`** — a 3–5 minute short-form
generator, not a film machine. Three things worth taking:
1. **audio-first duration** — continuous TTS, `silencedetect`, cut points.
   The one genuinely good idea in any of the four.
2. file-existence-as-checkpoint, applied consistently across 5 scripts.
3. frame-level resume in its upscaler.

The rest is a trap. `auto_pipeline.py` — its flagship "fully automated"
pipeline — **deadlocks**: it checks whether `raw/*.mp4` is full to decide it
reached the render step, but `raw/` is only written *by* that step, so the
check is permanently false and the loop spins forever. Also:
- `run_video_gen.sh:172` extracts frame **0** with `select='eq(n,0)'`, saves
  it as `shotN_last.png`, and calls it the last frame. Every I2V cut in the
  film is therefore seeded from the *opening* image of the previous shot.
- `workflow_generator.js:139` adds `Date.now()` to the seed, so a retried
  shot renders different frames and nothing is reproducible.
- The "quality degradation" retry only changes the **timeout**, not the
  workflow, and the seed drifts on every attempt. Attempt 3 is attempt 1 with
  a 15-minute deadline.
- `num_frames = Math.round(duration * 24)` is never snapped to the legal
  lattice. The repo documents the resulting drift itself: 50 shots → 3s.
  1,300 shots → 78s.
- It claims CosyVoice3 TTS. `grep` finds `cosyvoice` only in prose; the actual
  TTS is `edge-tts`, which the repo itself documents as capable of returning
  a **0-byte MP3** for certain inputs.
- It is not Wan 2.2. It is Wan 2.1 1.3B FP8 at 480×832, tuned for an 8GB 4060.
- `grep` for `openai|anthropic|llm|api_key|gpt|qwen|deepseek` across the whole
  repo: **zero matches**. The script splitting was done by the agent reading
  its own markdown, with human approval at every step.

**C) `kustomzone/ComfyUI-vidflows`** — 3 hand-drawn ComfyUI graphs plus one
useful Python file.
- `workflow-full-loop-Google.json` is **100% dead** for this purpose: Veo3,
  Gemini and Seedream, all paid APIs, zero Wan. The GPU sits idle.
- `workflow-full-loop-Storycreator.json` has real Wan 2.2 local (fp8, fits
  24GB) but its brain is Gemini + OpenAI, both paid, and it is shaped for
  poems, not screenplays — the shot boundary is a **line of verse**.
- `workflow-full-loop-Sora2-ComfyUI.json` is architecturally closest, but its
  LLM is `gpt-5-mini` and its TTS is MiniMax, both paid, and 356 nodes
  inside one ComfyUI execution is not resumable at all.
- The one treasure: `audio_duration_node.py`, ~143 lines, which does
  audio → frame_count → Wan length. **The concept is exactly right; the
  implementation has 8 bugs** — `int()` instead of rounding (drops a frame per
  shot), a hardcoded 22050 Hz fallback, exceptions swallowed into a silent
  0-frame result, a CUDA tensor bug (`hasattr(t,'numpy')` is always True, so
  a GPU tensor goes down the wrong path and errors out), and an `IS_CHANGED`
  that hashes `str(tensor)` on every execution.

**D) `pythongosssss/Wan2GP`** — **this repository no longer exists.** GitHub
returns 404 and the account has no such repo. The live one is
**`deepbeepmeep/Wan2GP`**, 9,643 stars. All analysis was redone against it.

**E) The Colab notebook** — manual, 100% UI clicking, no API of any kind. Its
`--profile 5` is a T4/15GB setting and is wrong for a 4090 (the engine's own
docker script uses `--profile 3` for 24GB). It sets `WAN_CACHE_DIR`, which no
code in the engine reads — a fictional variable. It symlinks `ckpts/`,
`loras/`, `outputs/` onto `/content`, which is ephemeral: **that is exactly
how the user's models were lost before.**

### ### The engine that was chosen

**Decision: `deepbeepmeep/Wan2GP` is the engine. ComfyUI is out of the
production path entirely.**

`deepbeepmeep/Wan2GP` is not a Gradio app with a script bolted on. It has four
headless entry points, verified by reading `docs/CLI.md`, `docs/API.md` and
`wgp.py`:

```python
from shared.api import init
session = init(root=..., output_dir=..., cli_args=["--profile", "4"])
result  = session.run_task({...})       # returns .generated_files / .errors
```

```bash
python wgp.py --process queue.zip --output-dir ./out --dry-run   # exit 0/1/130
python wgp.py --mcp --mcp-transport streamable-http --mcp-port 7866
```

It also declares, per model, exactly what it supports: `image.start`,
`image.end`, `image.reference`, `video.continue`, `video_length` accepting
`"10s"`, TTS with 3 distinct cloned voices and inline `[emotion]` tags, image
generation, RIFE interpolation, ffmpeg, and a `sliding_window` mechanism.
**Gradio has no API** — `grep api_name` over 14,000 lines returns nothing — so
driving the browser would have been the only alternative, and that is not
survivable unattended.

---

## 3. Corrections to BRIEF.md

Seven of the brief's premises are wrong. All are recorded in `HANDOFF.json`
under `brief_corrections`.

1. Source D is a dead repository. Use `deepbeepmeep/Wan2GP`.
2. `eg250k-del/openclude` was **empty** (409 Conflict). No prior code exists.
3. "Persistent Storage" on SaladCloud **does not exist**. From the official
   docs: *"SaladCloud does not currently support persistent storage… All data
   written to the container instance's filesystem is ephemeral"* and *"volume
   mounting using S3FS, FUSE, or NFS is not supported, as SaladCloud
   containers do not operate in privileged mode."* R2 replaces it.
4. WSL2 reserves part of the VRAM, so a 24GB 4090 is really ~22GB usable.
   Plan against 22.
5. The Colab notebook's `--profile 5` is wrong for this hardware.
6. The same notebook's `WAN_CACHE_DIR` is fictional.
7. "Wan 2.2" is two very different cost tiers, not one thing.

---

## 4. Architecture as built

```
        story (text)
            │
    ┌───────▼────────┐  llm.py       draft → Draft → Film
    │  script        │               (JSON in, JSON out, loud on prose)
    └───────┬────────┘
            │  estimated durations, marked unmeasured
    ┌───────▼────────┐  audio.py     render voice → PROBE IT → re-cut
    │  voice         │               never estimate, ever
    └───────┬────────┘
            │  measured durations
    ┌───────▼────────┐  schema.py    split anything over 5s,
    │  shots         │               resolve the continuity chain
    └───────┬────────┘
            │  one settings dict per shot
    ┌───────▼────────┐  runner.py    for each shot:
    │  render        │    state.py     begin → render → succeed/fail
    │                │    retry.py     on failure: shrink, don't die
    └───────┬────────┘    storage.py   upload the clip immediately
            │
    ┌───────▼────────┐  assembly.py  mux per shot, concat -c copy, verify
    │  film          │
    └────────────────┘
```

### The three ideas the audit gave us

**Audio first, never estimate.** The only source of a shot's length is a probe
of the rendered audio file. If the audio is 0 bytes or under 0.25s, the run
stops — because a silent voice is the most common way an entire film comes out
mute, and it looks like success until someone watches it.

**A shot is one generation.** Wan tops out at 121 frames ≈ 5.04s. Longer
narration is split into chained pieces, each separately retriable. This kills
the silent-truncation bug the audited repos all have, and it satisfies the
brief's hardest requirement for free.

**Degrade, don't die.** When the engine reports insufficient VRAM it says
"reduce the resolution or the number of frames" and stops. Left alone, an
unattended run burns all its retries re-sending the identical oversized job.
`retry.py` walks a 4-rung ladder: length → steps → resolution → quality. The
character sheet is the last thing sacrificed, because that is the only rung
that costs you the character's face.

---

## 5. Bugs found and fixed during the build

Ten real defects, each caught by a test written to fail.

| # | Bug | Consequence if shipped |
|---|---|---|
| 1 | 12s narration into a 5s generation window | dialogue cut off mid-sentence |
| 2 | `round()` instead of `ceil()` when splitting | pieces still over the limit |
| 3 | head piece inherited `SE` with no end image | render against a missing anchor |
| 4 | continuity paths baked in before splitting | film jumps back 3 shots, silently |
| 5 | `except BaseException` around the renderer | **Ctrl+C swallowed** — an operator's stop, or a container preemption, would be retried instead of honoured |
| 6 | a ladder rung added a spurious key | burns a retry while changing nothing |
| 7 | `succeed()`/`fail()` guarded on the wrong field | every correct render rejected |
| 8 | 0-byte or 1ms audio accepted | whole film silently mute |
| 9 | `llm.py` split a long shot but left anchors wrong | same as #3, second location |
| 10 | `_dump()` produced keys the loader rejected | the script stage would not resume |
| 11 | a blocked `ffprobe` crashed with a raw `WinError` | one unreadable file kills the audio stage instead of that one shot |

Number 5 is the one that would have cost real money on SaladCloud: the
container gets preempted, the runner catches it, tries again, and the retry
dies too, over and over, while the operator watches.

---

## 6. Verified behaviour

`demo_crash.py` — a 5-shot film against a hostile fake GPU:

```
RUN 1   process killed at shot 3
        shot      status    tries  frames
        s01_sh001 done          2      25
        s01_sh002 done          2      25
        s01_sh003 running       1       0     <- interrupted
        s01_sh004 pending       0       0
        s01_sh005 pending       0       0

RUN 2   fresh process, same state file
        resuming from: sh003, sh004, sh005
        film=f01 shots=5 done=5 progress=100.0%

RUN 3   renders attempted: 0
```

`demo_full.py` — all four stages, real numbers:

```
script  ok   audio ok   render ok   assemble ok
4 shots, 15.1s of measured narration, 4 clips, 1 film, everything banked
```

---

## 7. What a real film costs

Measured shot length on this machine: 3.78s. Prices are SaladCloud's RTX 4090
at $0.16 (lowest) to $0.27 (medium priority) per hour.

| Length | Shots | 5B FastWan (3-step) | 14B A14B (Lightning 4-step) |
|---|---|---|---|
| 1 min | 16 | 0.3–0.4 GPU-h · $0.04–0.11 | 0.7–1.3 GPU-h · $0.11–0.36 |
| 20 min | 317 | 5–8 GPU-h · $0.85–2.14 | 13–27 GPU-h · $2.12–7.14 |
| 60 min | 952 | 16–24 GPU-h · $2.54–6.43 | 40–79 GPU-h · $6.35–21.43 |
| 120 min | 1,905 | 32–48 GPU-h · **$5–13** | 79–159 GPU-h · **$13–43** |

> **Correction:** earlier in this project a figure of "$2–3 for a 2-hour film"
> was quoted. That was too optimistic — it assumed 24s per render, which is
> not achievable at 720p/121 frames on a 4090. The table above supersedes it.

---

## 8. Next, in order

1. **Dockerfile** — CUDA 12.8, `torch==2.10.0+cu128`, `ffmpeg>=5` (needed for
   `-fps_mode`), the engine at a pinned commit, `boto3`, `rclone`. Model
   weights are *not* baked in; they are pulled at container start into the
   cache that R2 backs.
   *Done when:* `docker build` succeeds and `import shared.api` works inside.

2. **Real clients for the two stubbed stages** — any OpenAI-compatible
   endpoint for the `ScriptWriter`; the engine's own Qwen3-TTS or IndexTTS2
   for the `Synthesiser`. Both are `Protocol`s, so this is implementation,
   not redesign.
   *Done when:* story → Film with measured durations, no stubs.

3. **First real 60-second film on one Salad container.**
   *Done when:* `state/ledger.json` in R2 shows `done=15` and `final.mp4`
   plays.

4. **VRAM headroom check.**
   *Done when:* one real shot renders with zero degradations in the ledger.

5. **Salad Job Queue worker.** The queue retries 3 times, caps responses at
   10MB (so a clip can never be returned in the response body — it must go to
   R2), and treats an instance interruption as a job failure. That last one is
   exactly the case `state.py` is built for.
   *Done when:* kill the container mid-run and a replacement resumes the same
   ledger.

6. **Persist the degradation state.**
   *Done when:* a shot that needed 25 frames does not fail once after a
   restart.

7. **Character sheet generation.** The engine can generate images through the
   same API. The `Character.views` field is already shaped for it.

8. **Lip sync**, if wanted. The engine supports it (MultiTalk,
   FantasyTalking). The `Shot` schema has no field for it yet.

---

## 9. Honest gaps

- **No real GPU has run this.** Every render path is exercised against a fake
  engine. The orchestration is verified; the generation is not.
- **No real LLM, no real TTS.** Both are `Protocol`s with one fake
  implementation each.
- **ffmpeg is absent here**, so 4 integration tests skip. The guard rails
  around it (refusing to assemble a film with a hole, refusing an oversize
  clip, refusing duration drift) are tested by stubbing the probe.
- **`S3Store` has never touched a live bucket.** `LocalStore` is tested, and
  the S3 path is a thin wrapper.
- **No git on this machine**, so none of this is committed. The remote repo
  `eg250k-del/openclude` is still empty.
