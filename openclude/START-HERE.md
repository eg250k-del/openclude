# START HERE — if you are a new session

Copy everything between the lines below and paste it as your first message.

---

Read `G:\opencode\openclude\START-HERE.md` and follow it, then
`G:\opencode\openclude\PLAN.md`. The project changed direction: build ComfyUI
on a rented GPU, in the user's browser, with storage that survives. Do not
extend the story-to-film pipeline. Run:

```
cd G:\opencode\openclude
C:\Users\Pc\AppData\Local\Python\bin\python.exe -m openclude.cli status
```

That prints where the project stands, what is decided, what is next, and what
is missing. Then read `HANDOFF.json` for the machine-readable version and
`PROJECT_STATUS.md` for the reasoning.

The user is a complete beginner who does not read long messages and does not
edit code. Do things for them rather than instructing them, keep replies
short, and give at most one instruction at a time.

---

## Read `PLAN.md` first

The project changed direction. Read `openclude/PLAN.md` before anything else.

**In short:** the user wants ComfyUI on a rented GPU, in their browser, with
the models and outputs stored somewhere that survives the container being
stopped. A previous session built a story-to-film pipeline instead. It is
being abandoned, not extended. The storage layer, the SaladCloud client and
the secrets handling from it are all reused.

**The one test that matters:** a video exists in the repo, and the user watched
it in their browser. Not the test count. Not the code. That.

## What the old plan was

An AI animation production machine: `story -> script -> voice -> shots -> film`,
built to run unattended on rented GPUs where the container is assumed to die
without warning. It works up to the render and has never produced a frame.

## Where it actually is

**Storage is live and proven.** The pipeline is built, tested and committed,
the image builds on GitHub, and the object store has been written to, read
back and emptied against a real private Hugging Face repo. Not deployed yet,
because the SaladCloud job queue does not exist and nothing has ever run on a
GPU.

| | |
|---|---|
| tests | 447 passing, 4 skipped (need ffmpeg, absent locally) |
| code | ~5,600 source lines / ~5,500 test lines |
| image | `ghcr.io/eg250k-del/openclude` built and pushed by CI |
| storage | `mostaa2500/openclude`, private, live |
| account | SaladCloud org `mostafa-ai`, project `aoutooanimation` |
| quota | 10 container replicas, 0 currently used |
| cost so far | $0 |
| repo | `github.com/eg250k-del/openclude`, branch `main` |

## Locked decisions, do not re-derive

1. **Engine is `deepbeepmeep/Wan2GP`, not ComfyUI.** The engine has a
   documented in-process Python API and a headless CLI queue. ComfyUI cannot
   distinguish queued from running and discards node errors.
2. **Source D in the original brief is a dead repository.** `pythongosssss/Wan2GP`
   returns 404. All analysis was redone against `deepbeepmeep/Wan2GP`.
3. **The target repo was empty.** GitHub returned 409 Conflict. Nothing to
   continue from.
4. **"Persistent Storage" does not exist on SaladCloud.** Container disks are
   ephemeral and volume mounting is unsupported because containers are
   unprivileged. Object storage replaces it.
5. **Model is 5B FastWan (`ti2v_2_2_fastwan`), profile 3.** The 14B A14B needs
   two 14B transformers and does not fit a 4090.
6. **One shot is one generation, max ~5 s.** Longer narration is split and
   chained, which also makes each piece separately retriable.
7. **Seeds are derived from content, never randomised.** Two audited repos add
   a timestamp, which destroys reproducibility.
8. **Output is a hybrid**: mostly still art with camera motion, real generated
   video for the beats that matter. This is what the reference channels do;
   their hashtag is #MotionComic.
9. **Model weights are never stored.** They are public on HuggingFace and total
   ~15 GB, more than any free tier holds. The container re-downloads on each
   cold start, about three minutes at Salad's reported node speeds. Storing them
   would cost more than the bandwidth.
10. **Pruning is what makes a free tier viable.** A 2-hour film is 5.4 GB before
    pruning. `prune()` deletes the clips, narration and continuity frames once
    `final.mp4` exists, dropping the steady state under 1 GB. It never deletes
    the film, the ledger, or anything belonging to an unfinished film.

## The one thing blocking deployment

Object storage credentials. The container **refuses to start** without them, by
design, because rendering into a disk that vanishes loses the film.

**Cloudflare R2 does not work for this user.** They have no credit card, and
R2's checkout stops at a payment form — this was reached in the browser and
confirmed, not assumed. Backblaze B2 and Google Cloud Storage behave the same.

The answer is **Hugging Face Hub** (`HFStore` in `src/openclude/storage.py`):
free, no card, private repos, and `huggingface_hub` is *already* in the image
because the engine uses it to fetch model weights. So it adds no dependency.

Three backends, selected by `OPENCLIDE_STORE`:

| value | needs | durable |
|---|---|---|
| `hf` | `HF_REPO`, `HF_TOKEN` | yes |
| `s3` | the four `S3_*` vars | yes |
| `local` | nothing | **no** |

Set `OPENCLIDE_STORE` and the matching variables. Check with:

```
python -m openclude.cli store
```

## The user's only remaining task

**Nothing.** Storage is live. A private Hugging Face dataset repo exists at
`mostaa2500/openclude` and has been written to, read from and emptied for real.

Credentials live in `C:\Users\Pc\.openclude\env.ps1`, deliberately outside the
git repository. Dot-source it before anything that touches storage:

```powershell
. "$env:USERPROFILE\.openclude\env.ps1"
```

## What testing against the real repo found

Five bugs, none of which any unit test could have caught, because the unit
tests used a fake client and never imported `huggingface_hub`. The installed
version is **1.33.0**, and 1.x renamed a lot:

| # | what | how it would have failed |
|---|---|---|
| 1 | `hf_hub_upload` → `upload_file` | writes never worked at all |
| 2 | `repo_type` was never passed | writes addressed to the model namespace; the repo is a dataset |
| 3 | `file_info` removed | `exists()` returned False for files that were there |
| 4 | tree listings include folders | usage inflated; `delete()` got folder names |
| 5 | `delete_file(filename=…)` → `path_in_repo` | `prune()` deleted nothing while reporting a saving |

Number 3 is the serious one. `restore_if_present()` calls `exists()` to decide
whether to resume, so every restarted container would have restarted the whole
film from shot one, silently, with no error anywhere.

Number 5 is the reason `delete()` no longer swallows exceptions. It caught a
class of bug rather than a bug: every `except Exception: pass` in the store is
a place where the project can report success while doing nothing.

The general lesson, now enforced by tests: adapter code for a third-party
library must be tested against the installed library, not a stand-in.

## Commands that exist

```powershell
$py = "C:\Users\Pc\AppData\Local\Python\bin\python.exe"
cd G:\opencode\openclude

& $py -m pytest -q                     # 379 tests
& $py -m openclude.cli status          # where the project stands
& $py -m openclude.cli store           # object storage health
& $py -m openclude.cli doctor          # this machine
& $py demo_full.py                     # the whole pipeline, fake GPU
& $py tools\refresh_handoff.py         # regenerate the numbers in HANDOFF.json
```

## Environment gotchas, already paid for once

* The Python on `PATH` (`AppData\Local\Programs\Python\Python314`) is broken:
  its `Lib\encodings` is missing. Use
  `AppData\Local\Python\bin\python.exe` (3.14.7).
* git is at `C:\Program Files\Git\cmd\git.exe` and is not on `PATH`.
* Docker is not installed. The image builds on GitHub runners instead.
* ffmpeg is not installed, so 4 integration tests skip. The guard rails around
  it are tested by stubbing the probe.
* `G:\opencode\setup.py` and the `الدبلجة` folder are a **different project**
  the user owns. Do not commit, move, or delete them.

## The next three steps, in order

1. `. "$env:USERPROFILE\.openclude\env.ps1"`, then create the SaladCloud job
   queue: `POST /organizations/mostafa-ai/projects/aoutooanimation/queues`
2. `python -m openclude.salad apply --apply` — dry-run by default, `--apply`
   is mandatory, and `min_replicas` is 0 so an idle deployment costs nothing.
3. Run the first real 60-second film. Success is `done=15` in the ledger and a
   `final.mp4` that plays.

## Working rules for this project

* Every store action is dry-run unless the user says otherwise. `apply` without
  `--apply` exits 2 on purpose and a test asserts it.
* Never put a secret in a file, a commit, or a chat message. The user has
  already pasted the Salad API key into this conversation once; it was rotated
  immediately after.
* Verify by running, never by claiming. `refresh_handoff.py` regenerates the
  numbers so the handoff file cannot drift, and a test fails if it does.
* Report failures with the real error text. Ten defects were found by tests
  written to fail during the build; none of them were hypothetical.
