# PLAN — ComfyUI on cloud GPUs, with storage that survives

**Read this first in any new session.** It is the whole plan. `HANDOFF.json`
has the machine-readable version and `PROJECT_STATUS.md` has the history.

---

## THE URL — it works

```
https://parmesan-jicama-ji54x1onjwo3px9i.salad.cloud
```

Opening it returns 200 and serves the ComfyUI interface. Verified:

| path | status | meaning |
|---|---|---|
| `/` | 200 | the interface |
| `/api/system_stats` | 200 | the GPU is visible |
| `/api/object_info` | 200, 389 KB | every node and model the build has |

Container logs, from the Portal's Container Logs tab:

```
ComfyUI 0.3.43 started.
Comfy UI started
To see the GUI go to: http://*:8188
ComfyUI API 1.9.1 started.
Server listening at http://[::]:3000
GET /health -> 200
Starting Comfy and any warmup workflow took 6.065s
```

**It starts in six seconds and has zero models loaded**, which is what the
user asked for: the container is empty and fast, the user opens it and picks
what they want.

---

## The setup, in the order it was done

### 1. Storage, because it is the whole point

SaladCloud deletes a container's disk when the container stops, and volume
mounts are unsupported because containers are unprivileged. So a private
Hugging Face dataset repo holds everything that must survive.

```
repo    mostaa2500/openclude   (private)
env     C:\Users\Pc\.openclude\env.ps1   (outside the repo, on purpose)
```

Verified against the live repo, not just in tests:

```powershell
. "$env:USERPROFILE\.openclude\env.ps1"
& $py -m openclude.cli store
```

It prints `signed in as`, `can write: yes, verified`, and how much of the free
tier is used. A dead token is reported as a failure, not as an empty repo,
which was a bug it found.

### 2. The image

`comfy/Dockerfile` is a single `FROM` on the official Salad image:

```
ghcr.io/saladtechnologies/comfyui-api:comfy0.3.43-api1.9.1-torch2.7.1-cuda12.8-runtime
```

Read from the live registry, not the documentation: the docs quote a newer
version pair that does not exist.

**No manifest, no models, no build steps.** The first version shipped a
manifest with FLUX and Wan 2.2 in it, about 30 GB. The user pointed out what
that costs: the container took minutes to become usable before anyone had chosen
anything, on a residential node, and it pre-decided the models, which is the
user's decision.

Built by `.github/workflows/build-comfy.yml`, separate from the film image
because the base is 6 to 16 GB and should not delay anything else. No build
cache: `type=gha` tries to re-upload the whole base layer, fills the 10 GB
repository cache limit, and fails the build at 99 percent.

### 3. The container group

```
name  comfyui
id     5f697aff-7aba-4e80-a778-838aaf72ef60
port   8188
image  ghcr.io/eg250k-del/openclude-comfy:latest
```

`networking.port` must be **8188**, the ComfyUI interface. It was 3000 for a
while, which serves the API wrapper instead, and the user saw
`404 Route GET:/ not found` because the API has no `/` route.

The probes point at **3000** `/health` and `/ready`, because that is the port
that has them. Both ports are real and they do different jobs.

`autostart_policy` is false, so nothing bills unless someone starts it.

### 4. The GPU

The account has **49 GPU classes**. Only three were checked at first, all three
showed zero, and the watcher waited while 32 others had capacity.

`tools/comfy_watch.py` now reads the whole class list from the API, keeps the
NVIDIA ones with 24 GB or more, sorts by memory then by free nodes, and takes
the best. It picked **RTX 5090, 32 GB**.

**NVIDIA only.** The first version picked an **AMD RX 7900 XTX** with 24 GB
free, which cannot run a CUDA PyTorch image at all. A memory number says
nothing about whether the image was built for the card.

---

## Every mistake, so none of them is repeated

### Deployment shapes, each a live 400

| what | was | is |
|---|---|---|
| `image` | object | plain string |
| `restart_policy` | object | `"always"` |
| `memory` | `32` = 32 MB | `32768` = MB |
| `storage_amount` | `120000` = 117 KB | bytes |
| `gpu_classes` | `"RTX 4090"` | a UUID from the API |
| probe `headers` | `{}` | `[]` |
| `failure_threshold` | `90`, then `40` | max **20** everywhere |

**The unit mistakes are the dangerous ones.** Nothing rejects a valid number in
the wrong unit, so a dry run cannot catch them and only a container that dies
on boot will.

### Other mistakes, each found the expensive way

- **Deployed a 7-character SHA as the image tag.** The image is tagged with the
  full 40-character SHA. A short one is a different tag and the node reports
  `Manifest Not Found`, which reads like a registry fault.
- **A manifest pointing at a model repository that does not exist**, at paths
  taken from documentation rather than a file listing. Every URL is now
  HEAD-checked in CI, so a bad path fails the build in seconds instead of
  costing an hour on a rented GPU.
- **Read container logs from a URL that does not exist**, so the watcher reported
  "running, no output" for seventeen minutes while the container was
  crash-looping. A wrong URL and a silent application look identical. The
  Portal's Container Logs tab is the reliable source.
- **Pinned three GPU classes and gave up when they were busy.** "No capacity" was
  never true.
- **Chose an AMD card.** See above.
- **`language` defaulted to `"ar"` in three places** on a project with no Arabic
  content, and the tests asserted the wrong value, so nothing caught it.
- **`HF_TOKEN` printed in full by two commands**, because SaladCloud echoes the
  container's environment back in every response and one redaction only matched
  `SECRET` and `KEY`. Every print path now goes through `safe_json`.
- **A liveness probe killed the container during its weight download**, because
  the entrypoint downloaded 15 GB before the process serving `/health` existed.
  A single shot holds the process for minutes, so the probe is now a
  forty-five-minute backstop and a `startup_probe` covers the cold start.
- **Believed `success` from CI meant the image worked.** It means the image
  built. Compare the run's `headSha` with `git rev-parse HEAD`.

### Process mistakes

- **Seven commits sat unpushed for a whole session.** The checkout was on a
  feature branch, so `git push origin main` pushed the unmoved local `main` and
  reported "Everything up-to-date", and `git push -q` hid it. See
  `docs/how-to-push.md`.
- **Wrote tests while the user waited for a video.** They said several times
  they wanted to see a result. They were right each time.

---

## The order the user described, which is the plan

> You start the container and open ComfyUI, you pick a template, ComfyUI tells
> you which tools it needs, and I put them somewhere they survive.

The user's hands are on the interface. The agent does the plumbing: the
container is up, the address works, and whatever the user installs is saved
before the container is stopped.

### Next: step three, the storage

`tools/model_store.py` mirrors models between the container and the repo:

```
sync      copy anything new out of the container into the repo
restore   copy anything in the repo that the container is missing
```

Neither is wired to the running container yet, because SaladCloud has no exec
endpoint. The route is the job-queue worker, or driving it from the ComfyUI
API's `/download` endpoint, which the wrapper already exposes.

**A decision for the user, not for them:** the base models are tens of
gigabytes and a free tier holds ten, so they cannot all be kept. Re-downloading
from Hugging Face is free and takes minutes on a cloud node; keeping everything
needs a paid bucket, which needs a credit card this account does not have.
Whatever is decided, **the user's own generated images and videos always go in
the repo.**

---

## The one test that matters

**The user opens the URL, picks a template, and something appears in the repo
afterwards.** Not the test count. Not the code. That.

It works up to the first half now: the URL answers and the interface is
served. The second half is unproven.
