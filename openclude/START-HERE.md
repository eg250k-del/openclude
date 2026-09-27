# START HERE — if you are a new session

Copy everything between the lines below and paste it as your first message.

---

Read `G:\opencode\openclude\START-HERE.md` and follow it. The project is
built and committed; do not start from scratch. Run:

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

## What this is

An AI animation production machine: `story -> script -> voice -> shots -> film`,
built to run unattended on rented GPUs where the container is assumed to die
without warning.

## Where it actually is

Built, tested, committed, and the container image builds on GitHub. **Not
deployed**, because the object storage credentials are the one missing piece
and only the user can create those.

| | |
|---|---|
| tests | 409 passing, 4 skipped (need ffmpeg, absent locally) |
| code | ~5,300 source lines / ~5,000 test lines |
| image | `ghcr.io/eg250k-del/openclude` built and pushed by CI |
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

1. `huggingface.co` → Join (free)
2. Datasets → New dataset → name it → **Private**
3. Settings → Access Tokens → Create → permission **Write** → copy the token
4. Set `HF_REPO=<user>/<repo>` and `HF_TOKEN=<token>`, then run
   `python -m openclude.cli store` until it reports usage

Never paste the token into chat. It goes in the shell only.

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

1. Create the HF dataset repo and token (above), then run
   `python -m openclude.cli store` until it reports usage.
2. Create the SaladCloud job queue:
   `POST /organizations/mostafa-ai/projects/aoutooanimation/queues`
3. `python -m openclude.salad apply --apply` — dry-run by default, `--apply`
   is mandatory, and `min_replicas` is 0 so an idle deployment costs nothing.

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
