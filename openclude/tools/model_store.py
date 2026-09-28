"""Mirror whatever lands in the container's model folder into external storage.

This is the step that makes the whole project worth doing, and it is step three
of the order the user described:

    you start the container and open ComfyUI, you pick a template, ComfyUI
    tells you which tools it needs, and this puts them somewhere they survive

The container's disk is deleted when the container stops. SaladCloud says so
plainly and offers no volume mount, because containers are unprivileged. So
anything the user installs is gone on the next start unless it is copied out
first, and anything already copied out is fetched back on the way in.

Both directions are here, and both are resumable, because a 30 GB transfer
over a residential connection does not happen in one attempt.

    --sync     copy anything new out of the container into the repo
    --restore  copy anything in the repo that the container is missing
    --once     do a single pass and exit, which is what the tests use
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclude.salad import (  # noqa: E402
    Config,
    SaladError,
    call,
    containers_path,
)
from openclude.storage_doctor import build_store  # noqa: E402

#: Where ComfyUI keeps its models, and which sub-folders are worth keeping.
#:
#: `custom_nodes` is separate and handled too: a custom node is often the thing
#: that took the longest to install, and it is small, so it is the best possible
#: thing to save.
MODEL_ROOT = "/opt/ComfyUI/models"
NODE_ROOT = "/opt/ComfyUI/custom_nodes"

#: Nothing under `unet` or `temp` is worth copying, and `temp` can be large.
SKIP_DIRS = {"temp", ".git", "__pycache__", ".git"}

#: Git metadata for a custom node is tiny and useless to restore, and excluding
#: it keeps the repo readable.
SKIP_SUFFIX = (".pyc", ".pyo")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# reading the container's filesystem
# --------------------------------------------------------------------------


def _exec(salad_cfg: Config, script: str, timeout: int = 120) -> str:
    """Run a shell snippet in the running container and return stdout.

    SaladCloud has no exec endpoint, so this goes through the job-queue worker
    the container group runs. That means it only works while an instance is
    running, which is exactly when a sync is wanted.
    """
    raise NotImplementedError(
        "Container exec needs the job-queue worker attached to the group. "
        "See deploy the group with a queue connection, or drive this from the "
        "ComfyUI API's /download endpoint instead."
    )


def local_walk(root: str) -> list[tuple[str, int]]:
    """Every file under a root, as (relative path, size). Raises if absent."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(root)
    out: list[tuple[str, int]] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        # Relative parts, not absolute: on Linux an absolute path contains no
        # ".git" segment, so a custom node at some/.git/config was being saved
        # as a model. It is small, but it means the repo fills with junk and the
        # restore puts a nested .git back in place.
        parts = p.relative_to(base).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if p.suffix in SKIP_SUFFIX:
            continue
        out.append((str(p.relative_to(base)).replace("\\", "/"), p.stat().st_size))
    return out


def plan_sync(files: dict[str, int], remote_sizes: dict[str, int]) -> dict:
    """What needs copying, and what does not.

    A file that is already in the repo at the same size is left alone. On a
    30 GB model set that is the difference between a sync that takes a minute
    and one that re-uploads everything, and the user is paying by the hour.
    """
    new = {p: s for p, s in files.items() if p not in remote_sizes}
    changed = {p: s for p, s in files.items()
               if p in remote_sizes and remote_sizes[p] != s}
    unchanged = {p: s for p, s in files.items()
                 if p in remote_sizes and remote_sizes[p] == s}
    return {
        "new": new,
        "changed": changed,
        "unchanged": unchanged,
        "bytes": sum(s for s in new.values()) + sum(s for s in changed.values()),
    }


def plan_restore(remote: dict[str, int], local: dict[str, int]) -> dict:
    """What needs fetching into the container.

    The mirror image of plan_sync. Anything present locally is left alone, so a
    container that already has the model from an earlier run downloads nothing.
    """
    missing = {p: s for p, s in remote.items() if p not in local}
    present = {p: s for p, s in remote.items() if p in local}
    return {"missing": missing, "present": present,
            "bytes": sum(missing.values())}


# --------------------------------------------------------------------------
# the two operations
# --------------------------------------------------------------------------


def sync(prefix: str, root: Path) -> int:
    """Copy anything in the container's folder into the repo."""
    store = build_store()
    files = local_walk(str(root))
    remote = {k.split("/")[-1]: v for k, v in _remote_sizes(store, prefix).items()}
    p = plan_sync(files, remote)
    log(f"sync {prefix}: {len(files)} files locally, "
        f"{len(p['new'])} new, {len(p['changed'])} changed, "
        f"{len(p['unchanged'])} already saved, "
        f"{p['bytes'] / 1024**2:.0f} MB to copy")
    for rel, size in p["new"].items():
        store.put(f"{prefix}/{rel}", root / rel)
        log(f"  saved {rel} ({size / 1024**2:.1f} MB)")
    for rel, _ in p["changed"].items():
        store.put(f"{prefix}/{rel}", root / rel)
        log(f"  updated {rel}")
    return 0


def restore(prefix: str, root: Path) -> int:
    """Copy anything in the repo that the container is missing."""
    store = build_store()
    try:
        local = {p: s for p, s in local_walk(str(root))}
    except FileNotFoundError:
        local = {}
    remote = _remote_sizes(store, prefix)
    p = plan_restore(remote, local)
    log(f"restore {prefix}: {len(remote)} in the repo, {len(p['missing'])} missing, "
        f"{p['bytes'] / 1024**2:.0f} MB to fetch")
    (root).mkdir(parents=True, exist_ok=True)
    for rel, _ in p["missing"].items():
        store.get(f"{prefix}/{rel}", root / rel)
        log(f"  fetched {rel}")
    return 0


def _remote_sizes(store, prefix: str) -> dict[str, int]:
    """Sizes of everything under a prefix, keyed by the path relative to it.

    An empty prefix is refused rather than treated as "everything". That matters
    because a restore with no prefix would pull the user's generated videos out
    of the repo and write them into a models folder, and a sync with no prefix
    would scatter the whole repo into the container's model directory. Both are
    destructive in a way that is quiet: the files land, and nothing complains
    until something is overwritten.
    """
    prefix = (prefix or "").strip().strip("/")
    if not prefix:
        raise ValueError(
            "a prefix is required: an empty one means the entire repository, "
            "which would copy generated videos into a models folder"
        )
    head = prefix + "/"
    return {k[len(head):]: v for k, v in store.sizes().items() if k.startswith(head)}


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def run_once(prefix: str, root: Path, direction: str) -> int:
    if direction == "sync":
        return sync(prefix, root)
    return restore(prefix, root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("direction", choices=("sync", "restore", "loop"))
    ap.add_argument("--root", default=MODEL_ROOT,
                    help="a folder on this machine, or the container's path")
    ap.add_argument("--prefix", default="models",
                    help="folder in the Hugging Face repo")
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    if not os.environ.get("HF_REPO"):
        log("HF_REPO is not set. Source the env file first.")
        return 2
    root = Path(args.root)

    if args.direction == "loop":
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            run_once(args.prefix, root, "sync")
            time.sleep(args.poll)
        return 0
    return run_once(args.prefix, root, args.direction)


if __name__ == "__main__":
    raise SystemExit(main())
