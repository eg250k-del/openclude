# Deploying to SaladCloud

Everything here is written so that **nothing costs money unless you explicitly
say so**, and **no secret ever appears in a chat, a file, or a log**.

---

## The honest answer to "can you do it from here?"

**Yes, with one thing from you.** SaladCloud publishes an official API and a
set of agent skills, and the API is what `src/openclude/salad.py` wraps. But:

| Blocked? | Why |
|---|---|
| Yes | The image has to be **built and pushed to a registry** first. A container group points at an image; there is no image yet. |
| Yes | The API key must come from you. **As an environment variable, not pasted into chat.** |
| Yes | `SALAD_ORGANIZATION` and `SALAD_PROJECT` are names. The SaladCloud agent runbook says explicitly: *"Never guess or enumerate names."* |

So the order is: image → credentials → container. Never container first.

---

## What you need, and how to give it to me safely

### 1. The API key — do NOT paste it here

Get it from the Portal → **API Access**.

Set it in your environment, in the terminal you launch me from:

```powershell
$env:SALAD_API_KEY = "your-key-here"
```

That keeps it in your shell process. I read it from the environment; it never
goes into a file, a commit, or this conversation. If you paste it into chat it
is in the transcript, and it should be treated as compromised and rotated.

Salad's docs on this, verbatim:

> *Keep it in the environment, never in the prompt. Set `SALAD_API_KEY` as a
> secret environment variable and let the agent read it from there. Do not paste
> the key into a chat, a repository, or an `AGENTS.md`.*

Note also: **there is no read-only or single-organization key.** Your key acts
everywhere you can. If that worries you, invite a separate member account to
one organization and give the agent that account's key instead.

### 2. Two names

```powershell
$env:SALAD_ORGANIZATION = "your-org-name"
$env:SALAD_PROJECT      = "your-project-name"
```

Both are visible in the Portal URL and in the projects page. I will not guess
them, and a wrong guess creates resources in the wrong place.

### 3. Object storage — this is not optional

SaladCloud's own docs:

> *SaladCloud does not currently support persistent storage for container
> instances. All data written to the container instance's filesystem is
> ephemeral and will be lost when the container instance is terminated.*

> *Volume mounting using S3FS, FUSE, or NFS is not supported, as SaladCloud
> containers do not operate in privileged mode.*

So we need a real object store. Three backends are supported; pick one.

### Option A — Hugging Face (free, no credit card) ← the one we are using

```powershell
$env:OPENCLIDE_STORE = "hf"
$env:HF_REPO        = "your-username/openclude"
$env:HF_TOKEN       = "hf_..."
```

How to get those, click by click:

1. `huggingface.co` → **Join** (free, no card asked for)
2. Left menu → **Datasets** → **New dataset**. Name it anything, e.g. `openclude`.
   Set it to **Private**. It is a *dataset* repo, not a model repo — this is a
   file store, not a model.
3. Open that dataset → **Settings** (or the gear icon) → **Access Tokens** →
   **Create User Access Token** → permission **Write** → **Create token**.
4. Copy the token once; it is shown only once.
5. Your repo id is the two names in the dataset's URL, e.g.
   `huggingface.co/datasets/your-username/openclude` gives `your-username/openclude`.

Why this one: `huggingface_hub` is already installed in the image, because the
engine uses it to fetch model weights. So this backend adds no dependency and no
new failure surface. R2, Backblaze B2 and Google Cloud Storage all stop at a
payment form.

### Option B — any S3-compatible store

```powershell
$env:OPENCLIDE_STORE       = "s3"
$env:S3_ENDPOINT            = "https://<account>.r2.cloudflarestorage.com"
$env:S3_BUCKET              = "openclude"
$env:S3_ACCESS_KEY_ID       = "..."
$env:S3_SECRET_ACCESS_KEY   = "..."
```

Cloudflare R2, Backblaze B2, MinIO, or anything else that speaks S3. 10 GB is
enough: a 2-hour film is under 1 GB once the redundant clips are pruned, and
`prune()` is what makes that true.

### Option C — a local folder

```powershell
$env:OPENCLIDE_STORE = "local"
```

Works for development and for the offline demo. **Not durable.** A SaladCloud
container's disk is deleted the moment it stops.

### Check it before deploying

```powershell
& $py -m openclude.cli store
```

Prints a usage report, or exactly which variable is missing.

The container **refuses to start** without a durable store
(`OPENCLIDE_REQUIRE_STORE=1`), because rendering a film into a disk that
vanishes is worse than not starting. That is the check that would have saved
the models lost earlier.

---

## The commands, in order

```powershell
cd G:\opencode\openclude
$py = "C:\Users\Pc\AppData\Local\Python\bin\python.exe"
```

### Step 1 — read-only, free, safe

```powershell
& $py -m openclude.salad preflight
```

Reads your quota and node pool. Proves the key works. Costs nothing. If this
fails, the key or a name is wrong and nothing else will work.

### Step 2 — see exactly what would be created

```powershell
& $py -m openclude.salad plan
```

Prints the full container-group spec with secrets redacted, plus the cost
ceiling. Sends nothing.

### Step 3 — build and push the image

```powershell
docker build -t ghcr.io/eg250k-del/openclude:latest .
docker push ghcr.io/eg250k-del/openclude:latest
```

Set the image for the deploy step:

```powershell
$env:OPENCLIDE_IMAGE = "ghcr.io/eg250k-del/openclude:latest"
```

For a private image, add registry credentials to the container group spec —
Salad supports GHCR, Docker Hub, ECR, ACR and GAR.

### Step 4 — create it, for real

```powershell
& $py -m openclude.salad apply --apply
```

The `--apply` flag is mandatory. Without it the script prints a refusal and
exits, and there is a test that asserts it.

### Step 5 — watch

```powershell
& $py -m openclude.salad status
& $py -m openclude.salad logs <instance-id>
```

---

## What the deploy actually asks for

| Setting | Value | Why |
|---|---|---|
| GPU | RTX 4090, priority `low` | $0.16/h. Priority is the single biggest cost lever. |
| CPU / RAM | 8 / 32 GiB | profile 3 needs 32 GB RAM and 24 GB VRAM |
| disk | 120 GB | weights plus a 720p working set |
| **min replicas** | **0** | **idle costs nothing** |
| max replicas | 3 | only reached when the queue has work |
| `WAN2GP_PROFILE` | 3 | the engine's own 24 GB label. The Colab notebook uses 5, which is a T4 choice. |
| `WAN2GP_ATTENTION` | sage2 | fastest attention on a 4090 |
| startup probe | 400 s grace | the first model download takes minutes; a short probe kills the instance mid-download |
| architecture | amd64 | Salad does not support ARM |

**Cost ceiling at full load:** 3 × $0.16 = **$0.48/hour**. At idle: **$0.00**.
Salad's own project quota is the hard stop, and `OPENCLIDE_BUDGET_CENTS`
(default 500 = $5/h) is the second stop.

---

## How work gets in

The job queue is **a folder in object storage**, not an HTTP endpoint:

```
<bucket>/queue/pending/<job-id>.json     <- you put a job here
<bucket>/queue/done/<job-id>.json       <- the worker writes the result
<bucket>/<film-id>/state/ledger.json    <- per-shot progress
<bucket>/<film-id>/clips/*.mp4
<bucket>/<film-id>/audio/*.wav
<bucket>/<film-id>/film/final.mp4
```

A job file looks like this:

```json
{
  "id": "j001",
  "story": "a pilot takes a derelict plane out in a storm",
  "film_id": "f01",
  "target_minutes": 1,
  "language": "ar",
  "characters": [
    {
      "id": "nova",
      "name": "Nova",
      "description": "Female, 25, athletic, jet black undercut hair, grey tactical shirt",
      "views": [{ "angle": "front", "image_path": "refs/nova_front.png" }],
      "voice_reference": "voices/nova.wav"
    }
  ]
}
```

This is deliberate. Salad's Job Queue worker forwards HTTP, and this container
can be killed at any instant. Anything held only in RAM is a lost job, so the
queue is a folder of files written before processing. It is the smallest thing
that survives a preemption.

---

## What is still missing before this runs end to end

Honest list, because a container that starts and then fails is expensive:

1. **No real LLM client.** The `ScriptWriter` protocol is implemented and
   tested; only a stub exists. The worker raises a clear error naming the gap.
2. **No real TTS client.** Same shape — `Synthesiser` protocol, stub only.
3. **The image has never been built.** The Dockerfile is written against
   versions verified in the engine's own Colab notebook, but nothing has
   compiled it. Expect the first build to surface version friction.
4. **No character-sheet image generation.** The field exists; nothing fills it.
5. **git is not installed on this machine**, so nothing is committed and the
   GitHub repo is still empty. You will need to install git and push before
   `ghcr.io/eg250k-del/openclude` can be built by CI or by hand.

---

## Cost control, plainly

- `min_replicas: 0` means **an idle project costs nothing.** This is the single
  most important line in the spec.
- Run one film at a time. `max_replicas: 3` is headroom for a preemption, not a
  throughput plan — a single 4090 renders one shot at a time regardless.
- A 1-minute film on the 5B model is roughly **$0.04–0.11**. A 2-hour film is
  **$5–13**. Check `demo_full.py` for the table.
- If a run goes wrong, `python -m openclude.salad status` shows the instances
  before you decide anything. Do not scale down or delete without deciding that
  on purpose — those are the actions that lose work.
