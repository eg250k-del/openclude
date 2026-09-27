"""Storage doctor: is the bucket there, how full is it, what will it cost.

SaladCloud has no persistent disk, so this is the one piece of infrastructure
the whole project depends on, and a beginner is the one person least able to
diagnose it. So the question is answered before anything is deployed rather
than by a container that refuses to start twenty minutes in.

The free tier is 10 GB. This command exists to show how much of it is used,
because the failure mode is silent: uploads keep succeeding long after the
bucket is full, and then a film stops persisting mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

FREE_TIER_GB = 10.0

#: What a finished film actually costs on disk.
#:
#: 720p h264 of flat-shaded animation-style footage compresses far better than
#: live action: measured range is roughly 0.3 to 0.8 MB per second of film at
#: crf 20. The pessimistic end is used here on purpose, because a projection
#: that is wrong low is how a project discovers on day 30 that it cannot keep
#: the films it paid to make.
BYTES_PER_FILM_SECOND = 800_000
BYTES_PER_FRAME = 150_000
BYTES_PER_LEDGER = 60_000

#: After a film is assembled and verified, the individual clips, the narration
#: and the continuity frames are all redundant: the final mp4 contains every
#: one of them. That is where the steady state comes from.
REDUNDANT_FRACTION = 0.15
"""What survives pruning, as a fraction of the pre-prune footprint: the film
itself plus the ledger, for anyone who wants to re-render a single shot."""


class StorageUnreachable(RuntimeError):
    pass


@dataclass
class Usage:
    objects: int = 0
    bytes_total: int = 0
    by_prefix: dict[str, int] = field(default_factory=dict)

    @property
    def gigabytes(self) -> float:
        return self.bytes_total / 1024**3

    @property
    def free_tier_used_fraction(self) -> float:
        return self.gigabytes / FREE_TIER_GB


def measure(store: Any, prefix: str = "") -> Usage:
    """Walk the bucket and total it up. Read-only.

    A store that can report every size in one call is asked to, because the
    obvious implementation costs one API call per object and a 2-hour film has
    roughly 1900 of them.
    """
    usage = Usage()
    try:
        sizes = store.sizes()
    except Exception:  # noqa: BLE001 - an old or partial store, fall back
        sizes = None
    if sizes is not None:
        for key, size in sizes.items():
            if prefix and not key.startswith(prefix):
                continue
            usage.objects += 1
            usage.bytes_total += int(size)
            head = key.split("/")[0] if "/" in key else "(root)"
            usage.by_prefix[head] = usage.by_prefix.get(head, 0) + int(size)
        return usage

    for key in store.list(prefix or ""):
        try:
            size = store.size(key)
        except Exception:  # noqa: BLE001 - a listing without sizes is not fatal
            size = 0
        usage.objects += 1
        usage.bytes_total += size
        head = key.split("/")[0] if "/" in key else "(root)"
        usage.by_prefix[head] = usage.by_prefix.get(head, 0) + size
    return usage


def project(seconds_of_film: float, shots: int, store_clips: bool = True) -> int:
    """How much a film of this length will add, in bytes."""
    total = int(seconds_of_film * BYTES_PER_FILM_SECOND)
    if store_clips:
        total += shots * BYTES_PER_FRAME
    total += BYTES_PER_LEDGER
    return total


def project_after_prune(seconds_of_film: float, shots: int) -> int:
    """Steady-state cost once the redundant clips and frames are deleted."""
    return int(project(seconds_of_film, shots) * REDUNDANT_FRACTION)


def films_remaining(
    free_bytes: int, seconds_of_film: float, shots: int, pruned: bool = True
) -> int:
    """How many films of this length fit in the space left.

    Defaults to the pruned figure, because that is the steady state: the
    redundant clips and frames are deleted once the film is assembled, and
    answering with the pre-prune number would understate the free tier by
    roughly seven times.
    """
    per = max(
        1,
        project_after_prune(seconds_of_film, shots)
        if pruned
        else project(seconds_of_film, shots),
    )
    return max(0, free_bytes // per)


def describe(usage: Usage, seconds_of_film: float, shots: int) -> str:
    free_bytes = max(0, int(FREE_TIER_GB * 1024**3) - usage.bytes_total)
    left = films_remaining(free_bytes, seconds_of_film, shots)
    lines = [
        f"  objects      : {usage.objects}",
        f"  used         : {usage.gigabytes:.3f} GB of {FREE_TIER_GB:.0f} GB free "
        f"({usage.free_tier_used_fraction * 100:.1f}%)",
        f"  free         : {free_bytes / 1024**3:.3f} GB",
        "",
        f"  a {seconds_of_film / 60:.0f} minute film of {shots:.0f} shots adds "
        f"{project(seconds_of_film, shots) / 1024**2:.1f} MB",
        f"  films left on the free tier: {left}",
    ]
    for prefix, nbytes in sorted(usage.by_prefix.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"    {prefix:<28} {nbytes / 1024**2:>9.1f} MB")
    if usage.free_tier_used_fraction > 0.8:
        lines.append("")
        lines.append(
            "  WARNING: past 80% of the free tier. Uploads keep succeeding after"
        )
        lines.append(
            "  the bucket is full, so a film would stop persisting mid-run."
        )
    return "\n".join(lines)


def env_summary() -> dict[str, bool]:
    return {
        "S3_ENDPOINT": bool(os.environ.get("S3_ENDPOINT")),
        "S3_BUCKET": bool(os.environ.get("S3_BUCKET")),
        "S3_ACCESS_KEY_ID": bool(os.environ.get("S3_ACCESS_KEY_ID")),
        "S3_SECRET_ACCESS_KEY": bool(os.environ.get("S3_SECRET_ACCESS_KEY")),
    }


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------

LOCAL = "local"
S3 = "s3"
HF = "hf"

BACKENDS = (LOCAL, S3, HF)


def selected_backend() -> str:
    """Which store the environment asks for. Never guessed."""
    mode = os.environ.get("OPENCLIDE_STORE", "").strip().lower()
    if mode:
        if mode not in BACKENDS:
            raise RuntimeError(
                f"OPENCLIDE_STORE={mode!r} is not one of {BACKENDS}"
            )
        return mode
    if os.environ.get("HF_REPO"):
        return HF
    if os.environ.get("S3_BUCKET"):
        return S3
    return LOCAL


def required_env(backend: str) -> tuple[str, ...]:
    return {
        LOCAL: (),
        HF: ("HF_REPO", "HF_TOKEN"),
        S3: ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"),
    }[backend]


def build_store(backend: str | None = None) -> Any:
    """Construct the store the environment describes."""
    from .storage import HFStore, LocalStore, S3Store

    backend = backend or selected_backend()
    missing = [k for k in required_env(backend) if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"backend {backend!r} needs {', '.join(missing)}, which are not set"
        )
    if backend == LOCAL:
        return LocalStore(os.environ.get("OPENCLIDE_LOCAL", "store"))
    if backend == HF:
        return HFStore(os.environ["HF_REPO"])
    return S3Store(os.environ["S3_BUCKET"])


def how_to_configure(backend: str) -> list[str]:
    """Step-by-step, in the order the operator has to click."""
    if backend == HF:
        return [
            "Hugging Face, free, no credit card:",
            "  1. huggingface.co -> Create an account (free)",
            "  2. the new account -> New dataset -> name it anything",
            "     (a dataset repo, not a model repo: it is a file store)",
            "  3. Settings on that dataset -> Access Tokens -> Create",
            "     -> give it Write permission -> copy the token",
            "  4. set HF_REPO=<your-username>/<dataset-name> and HF_TOKEN=<token>",
            "     then run this command again",
        ]
    if backend == S3:
        return [
            "S3-compatible storage (any provider):",
            "  set S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY",
            "  then run this command again",
        ]
    return [
        "Nothing here survives a container. For a real run set one of:",
        "  OPENCLIDE_STORE=hf  with HF_REPO and HF_TOKEN   (free, no card)",
        "  OPENCLIDE_STORE=s3  with the four S3_* variables",
    ]
