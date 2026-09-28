# PLAN — ComfyUI on cloud GPUs, with storage that survives

**Read this first in any new session.** It is the whole plan. `HANDOFF.json`
has the machine-readable version and `PROJECT_STATUS.md` has the history of
how the previous attempt went.

---

## What the user actually wants

In their own words, roughly:

> Install ComfyUI, download the video and image models, rent a cloud GPU,
> open the ComfyUI interface from my browser, and make videos. The tools
> download through the cloud machine so my own internet is not used. It
> worked. The only problem was that closing the container and coming back
> lost everything, and I had to start over.

So: **the ComfyUI web interface, on a rented GPU, that does not forget.**

**Explicitly not wanted:** a story-to-film pipeline, a script writer, a job
queue, a headless render service, two-hour unattended renders, or a CLI with
no visible output. A previous attempt built all of that. It was the wrong
thing and it is being abandoned, not extended.

---

## What already works and must be reused

These are real, tested, and were verified against live services. Do not
rebuild them.

| Thing | Where | State |
|---|---|---|
| Private file storage | `src/openclude/storage.py`, `HFStore` | **verified against a live repo** |
| Storage doctor | `python -m openclude.cli store` | **verified** |
| SaladCloud client | `src/openclude/salad.py` | **verified**, every command tested live |
| Unattended watcher | `tools/watch.py` | tested, stops the GPU on a budget |
| Credentials | `C:\Users\Pc\.openclude\env.ps1` | **outside the repo, never commit** |
| Container group | `openclude-v2`, org `mostafa-ai`, project `aoutooanimation` | deployed, stopped |
| AI Gateway key | same env file | working, 4 models listed |

### Credentials already in place

```
HF_REPO        mostaa2500/openclude   (private)
HF_TOKEN       write, one token only
LLM_BASE_URL   https://ai.salad.cloud/v1
LLM_API_KEY    Salad AI Gateway
SALAD_API_KEY  SaladCloud API
SALAD_ORGANIZATION / SALAD_PROJECT / OPENCLIDE_GROUP
```

Load them with:

```powershell
. "$env:USERPROFILE\.openclude\env.ps1"
```

---

## Why the storage problem is already solved

SaladCloud deletes a container's disk the moment the container stops, and
volume mounts are unsupported because containers are unprivileged. So models
and outputs have to live somewhere else.

That somewhere else is a **private Hugging Face dataset repo**, chosen because
it is free, needs no credit card, and `huggingface_hub` was already in the
image for fetching model weights.

**This was the fix for exactly the problem the user described.** A previous
session recorded that the same setup was being used with ChatGPT and worked,
except that nothing survived a restart. The repo is the difference.

Test it any time:

```powershell
. "$env:USERPROFILE\.openclude\env.ps1"
& $py -m openclude.cli store
```

It prints `signed in as`, `can write: yes, verified`, and how much of the free
tier is used. A dead token is reported as a failure, not as an empty repo.

---

## The plan

### Step 1 — find the right ComfyUI image

SaladCloud's own catalog has ComfyUI images, so there may be nothing to
build. The original work that was done with ChatGPT used one of them.

1. List what is available, from the API, not from memory:
   `GET /organizations/<org>/container-group/images` (name to be confirmed
   from the Salad API reference)
2. Compare candidates on: does it include the manager, does it run headless,
   is the disk big enough for 50 GB of models, does it stay up.
3. Write the chosen image into the env file as `COMFY_IMAGE`.

**Success:** a container group starts from that image and stays running.

### Step 2 — persist the models

The models are 50 GB and the container disk is thrown away. So:

1. On the first run, download the models **through the container**, so the
   user's own internet is not used, which is what they asked for.
2. Upload them to the HF repo, under a `models/` prefix.
3. On every later start, download them from the repo at boot.

Uploading 50 GB to a free tier will not fit. So decide the real shape first:

- **Option A, store only what is expensive:** the LoRAs, VAEs and any custom
  checkpoints, and let the big base models re-download from HuggingFace, which
  is free and fast. This is what `HFStore` was designed for.
- **Option B, store everything:** needs a paid bucket. Not possible without a
  credit card, which the user does not have.

**Recommend A.** It is the same reasoning already used for the video model
weights: re-downloading is cheaper than storing.

**Success:** after `stop`, `start`, the models are present again without the
user doing anything.

### Step 3 — open the interface in the user's browser

SaladCloud has a container gateway for port forwarding. The ComfyUI port
(8188) has to reach the user's browser.

1. Check what the gateway offers: the Portal shows a URL per running
   container, or there is an API for it.
2. If the Portal URL works, the user needs to do nothing.
3. If it does not, the alternative is an SSH or tunnel port forward.

**Success:** the user opens a URL in their own browser and sees the ComfyUI
interface. This is the moment the whole project becomes real to them, and it
should be tested before anything else is polished.

### Step 4 — the first video

1. Load a checkpoint, a text prompt, generate one image.
2. Load a video model, generate one short clip.
3. Confirm the output file appears in the repo.

**Success:** one image and one clip exist in the HF repo, downloaded back to
the user's machine.

---

## Cost

| | |
|---|---|
| One idle hour, stopped | **$0** |
| One hour running | $0.27 to $0.33 at `high` priority |
| One 5-second video | about $0.01 |

`high` priority is needed because at `low` there was no capacity at all,
checked live. `low` costs about $0.16/h and is worth revisiting once this
works, because that is where the cheap capacity is supposed to be.

**Always stop the group when finished.** `python -m openclude.salad stop`.
Leaving it on overnight is the expensive mistake.

---

## What not to do

These are the mistakes that cost the most time, all of them made in this
project. Each has a test or a check behind it now.

- **Never deploy an image whose tag you guessed.** The image is tagged with
  the full 40-character commit SHA. A short SHA is a different tag and the
  node reports `Manifest Not Found`.
- **Do not trust `success` from CI.** It means the image built, not that it
  works. Compare the run's `headSha` with `git rev-parse HEAD`.
- **Do not print a command's output without checking it for secrets.**
  SaladCloud echoes the container's environment variables back in every
  response, so printing any response can print a live token. `safe_json` in
  `salad.py` handles this; use it.
- **Do not read container logs from a guessed URL.** It is
  `GET /organizations/<org>/log-entries?instance_id=...`. A wrong URL returns
  nothing, which looks exactly like a silent application.
- **Do not let a probe kill a cold start.** Startup needs minutes for a GPU
  node, tens of GB of weights, and a contended machine. `startup_probe` with
  twenty minutes of grace, and a very lenient `liveness_probe`, because a
  render can hold the process for many minutes.
- **Do not read the API's limits from memory.** Every one that was guessed
  wrong produced a live 400: `restart_policy` is a string, `image` is a
  string, `memory` is in MB, `storage_amount` is in bytes, `gpu_classes` are
  UUIDs, and every probe `failure_threshold` is capped at 20.
- **Do not use a feature variable that the user did not ask for.** The
  language defaults were `"ar"` in three places on a project with no Arabic
  content, and the tests asserted the wrong value, so nothing caught it.

---

## Where the previous attempt went, in one paragraph

A story-to-film pipeline was built: an LLM writes a script, a synthesiser
makes voices, a video model renders shots, and ffmpeg assembles them, with an
atomic ledger in object storage so a preempted container resumes. It has 554
passing tests, storage proven against a live repo, a script writer proven
against a live LLM, and a container group that was deployed and started
successfully. **It never produced a single frame of video**, because a probe
configuration killed the container during its weight download, and then a
long series of API shape errors. The architecture was sound and the delivery
was zero. The user is right that the simpler thing was wanted.

Nothing in that work is wasted: the storage, the SaladCloud client, the
secrets handling and the deployment discipline are all still needed here.

---

## The one test that matters

**A video exists in the repo, and the user watched it in their browser.**

Nothing else counts. Not the test count, not the code, not the deployment.
Say so plainly until that happens.
