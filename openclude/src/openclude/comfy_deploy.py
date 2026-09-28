"""The ComfyUI container group: a different deployment from the film pipeline.

Everything here is a decision that cost something to get wrong, so the reason
travels with it. The list of mistakes at the bottom is the shortest useful
summary of what SaladCloud's API will and will not accept.

Read PLAN.md first. This file implements the part of it that touches
SaladCloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .salad import (
    API,
    AUTH_HEADER,
    Config,
    USER_AGENT,
    SaladError,
    call,
    check_capacity,
    containers_path,
    group_name,
    is_secret_name,
    safe_json,
)

#: The official Salad ComfyUI image, with this project's manifest layered on.
#: Pinned to a full commit SHA, like the film image, because a tag is mutable
#: and a node that pulls a different image than the one that was tested is a
#: run that produces something nobody can reproduce.
BASE_IMAGE = ("ghcr.io/saladtechnologies/comfyui-api:"
              "comfy0.3.43-api1.9.1-torch2.7.1-cuda12.8-runtime")

#: The container image to deploy. Overridable, because this project's image is
#: published separately by the build-comfy workflow and the SHA moves.
DEFAULT_IMAGE = os.environ.get(
    "COMFY_IMAGE",
    "ghcr.io/eg250k-del/openclude-comfy:latest",
)

#: ComfyUI's own web interface. This is the port the user opens in a browser,
#: and it is the whole point of the project: everything else is plumbing that
#: exists so that this port answers.
UI_PORT = 8188

#: The API wrapper. Used for the probes, because it has /health and /ready
#: while ComfyUI's own port does not have anything meaningful to probe.
API_PORT = 3000


@dataclass
class ComfyConfig:
    organization: str = ""
    project: str = ""
    image: str = DEFAULT_IMAGE
    group: str = "comfyui"
    gpu: str = "RTX 3090 Ti"
    #: The resolved class id, when the name came from a capacity search.
    #:
    #: Carried rather than re-derived, because a name only resolves against the
    #: three entries in the shortlist, and the search picks from all 49. The
    #: first attempt at this passed the discovered name to a function that
    #: looked it up in the shortlist and raised, one line after it had correctly
    #: found 63 free nodes.
    gpu_id: str = ""
    priority: str = "high"
    cpu: int = 8
    memory_mb: int = 32_768
    storage_bytes: int = 120 * 1024**3
    replicas: int = 1
    # Where generated files are written inside the container. They are uploaded
    # to the project's Hugging Face repo afterwards.
    output_dir: str = "/data/outputs"

    @classmethod
    def from_env(cls) -> "ComfyConfig":
        return cls(
            organization=os.environ.get("SALAD_ORGANIZATION", "").strip(),
            project=os.environ.get("SALAD_PROJECT", "").strip(),
            image=os.environ.get("COMFY_IMAGE", DEFAULT_IMAGE).strip(),
            group=os.environ.get("COMFY_GROUP", "comfyui").strip(),
            gpu=os.environ.get("OPENCLIDE_GPU", "RTX 3090 Ti").strip(),
            priority=os.environ.get("OPENCLIDE_PRIORITY", "high").strip(),
            cpu=int(os.environ.get("OPENCLIDE_CPU", "8")),
            memory_mb=int(os.environ.get("OPENCLIDE_MEMORY_MB", "32768")),
            storage_bytes=int(
                os.environ.get("OPENCLIDE_STORAGE_BYTES", str(120 * 1024**3))
            ),
            replicas=int(os.environ.get("COMFY_REPLICAS", "1")),
        )


def gpu_class_ids(gpu: str, gpu_id: str = "") -> list[str]:
    """Resolve a GPU to the UUIDs the API wants.

    An id that was already discovered wins over the name, because the account
    has 49 classes and the shortlist has three, so a name found by searching
    all of them cannot be looked up in the shortlist.
    """
    if gpu_id:
        return [gpu_id]

    from .salad import GPU_CLASSES

    if gpu in GPU_CLASSES:
        return [GPU_CLASSES[gpu]]
    if gpu_id:
        return [gpu_id]
    raise SaladError(
        f"unknown GPU class {gpu!r} and no id was supplied. Known by name: "
        f"{', '.join(sorted(GPU_CLASSES))}. Pass the id, or run the capacity "
        f"search, which reads the class list from the API."
    )


def group_spec(cfg: ComfyConfig) -> dict:
    """The exact body that gets posted.

    Every shape in here was corrected against a live 400. The mistakes, so they
    are not repeated:

      * `image` is a plain string, not an object with repository/architecture.
      * `restart_policy` is the string "always", not an object.
      * `memory` is megabytes and `storage_amount` is bytes. Passing 32 and
        120000 asked for 32 MB of RAM and 117 KB of disk, and nothing rejects
        a valid number in the wrong unit, so a dry run cannot catch it. Only a
        container that dies on boot will.
      * `gpu_classes` are UUIDs from GET /organizations/<org>/gpu-classes.
      * probe `headers` is an array, not an object.
      * every probe `failure_threshold` is capped at 20. That cap was hit
        three separate times before a test enforced it.
      * there is no `queue_autoscaler` or `queue_connection` here, because this
        project has no SaladCloud job queue. Naming one that does not exist
        fails the create.

    The startup probe is long because a cold start downloads about 30 GB of
    models over a residential connection before the first request can run.
    """
    probe_http = {
        "scheme": "http",
        "host": "localhost",
        "headers": [],
        "port": API_PORT,
    }
    return {
        "name": cfg.group,
        # False on purpose. This bills by the hour, and the user is not always
        # at the keyboard. They start it, they stop it.
        "autostart_policy": False,
        "replicas": cfg.replicas,
        "restart_policy": "always",
        "container": {
            "image": cfg.image,
            "resources": {
                "cpu": cfg.cpu,
                "memory": cfg.memory_mb,
                "storage_amount": cfg.storage_bytes,
                "shm_size": 1024,
                "gpu_classes": gpu_class_ids(cfg.gpu, cfg.gpu_id),
            },
            "priority": cfg.priority,
            "environment_variables": {
                # Lets the container pull gated models and upload results to
                # the user's own private repo.
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "HF_REPO": os.environ.get("HF_REPO", ""),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "SALAD_API_KEY": os.environ.get("SALAD_API_KEY", ""),
                "OPENCLIDE_DATA": "/data",
                "COMFY_OUTPUT_DIR": cfg.output_dir,
                # A cold start on a residential node is slow. 50 GB keeps the
                # cache from filling the node's disk and silently evicting a
                # model mid-run.
                "LRU_CACHE_SIZE_GB": "50",
            },
        },
        "networking": {
            # The web UI. Without this port forwarded the user cannot see
            # anything, and the whole project is invisible.
            "port": UI_PORT,
            "protocol": "http",
            "auth": False,
        },
        "startup_probe": {
            "http": {**probe_http, "path": "/health"},
            "initial_delay_seconds": 30,
            "period_seconds": 60,
            "timeout_seconds": 10,
            # Capped at 20 by the API, so the grace period comes from the
            # period: 20 x 60s = 20 minutes, which is roughly what 30 GB of
            # models takes on a home connection.
            "failure_threshold": 20,
            "success_threshold": 1,
        },
        "readiness_probe": {
            "http": {**probe_http, "path": "/ready"},
            "initial_delay_seconds": 30,
            "period_seconds": 30,
            "timeout_seconds": 10,
            "failure_threshold": 20,
            "success_threshold": 1,
        },
        "liveness_probe": {
            "http": {**probe_http, "path": "/health"},
            # Very lenient on purpose. A video generation can hold the process
            # for minutes; a short timeout kills healthy work. A dead process
            # does not need a probe, since it exits and the restart policy
            # handles it.
            "initial_delay_seconds": 300,
            "period_seconds": 120,
            "timeout_seconds": 10,
            "failure_threshold": 20,
            "success_threshold": 1,
        },
    }
