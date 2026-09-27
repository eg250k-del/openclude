"""Command line entrypoint.

`openclude status` is the command a new session runs first: it prints where
the project stands, what is verified, and what is next. Everything else here
exists so the machine can be driven without writing code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

HANDOFF = "HANDOFF.json"


def _handoff() -> dict:
    p = Path(__file__).resolve().parents[2] / HANDOFF
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def cmd_status(args: argparse.Namespace) -> int:
    h = _handoff()
    if not h:
        print("No HANDOFF.json found. Run the test suite to confirm the build.")
        return 1

    print()
    print("  A NEW SESSION: read openclude/START-HERE.md first. It has a")
    print("  copy-pasteable resume prompt and the locked decisions.")

    print("=" * 72)
    print(f"  openclude {h.get('version', __version__)}")
    print(f"  {h.get('headline', '')}")
    print("=" * 72)

    v = h.get("verified", {})
    print()
    print(f"  tests    : {v.get('tests', '?')} passing, {v.get('skipped', 0)} skipped")
    print(f"  code     : {v.get('source_lines', 0)} source / {v.get('test_lines', 0)} test lines")
    print(f"  last run : {v.get('timestamp', 'unknown')}")

    print()
    print("  STAGES")
    for s in h.get("stages", []):
        mark = {"done": "done", "partial": "part", "todo": "TODO"}[s["state"]]
        print(f"    [{mark:>4}] {s['name']:<12} {s.get('note', '')}")

    print()
    print("  DECISIONS LOCKED")
    for d in h.get("decisions", []):
        print(f"    - {d['decision']}")
        print(f"      why: {d['why']}")

    print()
    print("  NEXT")
    for i, n in enumerate(h.get("next", []), start=1):
        print(f"    {i}. {n['task']}")
        print(f"       {n['done_when']}")

    print()
    print("  READ FIRST, IN THIS ORDER")
    for f in h.get("read_first", []):
        print(f"    {f}")

    print()
    print("  CORRECTIONS TO THE ORIGINAL BRIEF")
    for c in h.get("brief_corrections", []):
        print(f"    {c}")

    print()
    print("  KNOWN GAPS")
    for g in h.get("known_gaps", []):
        print(f"    - {g}")

    print()
    print(f"  updated: {h.get('updated', 'unknown')}")
    print("=" * 72)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the machine this is running on, so failures are explained."""
    import shutil

    print("ENVIRONMENT")
    print(f"  python        {sys.version.split()[0]}")
    print(f"  ffmpeg        {shutil.which('ffmpeg') or 'MISSING'}")
    print(f"  ffprobe       {shutil.which('ffprobe') or 'MISSING'}")
    try:
        import torch

        print(f"  torch         {torch.__version__} cuda={torch.cuda.is_available()}")
    except ImportError:
        print("  torch         not installed (fine outside the container)")
    for var in ("S3_ENDPOINT", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        import os

        print(f"  {var:<13} {'set' if os.environ.get(var) else 'unset'}")
    return 0


def cmd_store(args: argparse.Namespace) -> int:
    """Report the object store, and how much room is left."""
    import os

    from .storage_doctor import (
        FREE_TIER_GB,
        build_store,
        describe,
        how_to_configure,
        measure,
        required_env,
        selected_backend,
    )

    print("STORAGE DOCTOR\n")
    try:
        backend = selected_backend()
    except RuntimeError as exc:
        print(f"  {exc}")
        return 1
    print(f"  {'backend':<20} {backend}")

    for key in ("OPENCLIDE_STORE", "HF_REPO", "HF_TOKEN", "S3_ENDPOINT",
                "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        print(f"  {key:<20} {'set' if os.environ.get(key) else '-'}")

    missing = [k for k in required_env(backend) if not os.environ.get(k)]
    if missing:
        print()
        print("  Not configured. Nothing else can run: SaladCloud deletes a")
        print("  container's disk when it stops, so the ledger and the clips")
        print("  would vanish with it.")
        print()
        for line in how_to_configure(backend):
            print(f"  {line}")
        return 1

    try:
        store = build_store(backend)
        usage = measure(store)
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"  UNREACHABLE: {type(exc).__name__}: {exc}")
        return 1

    print()
    print(f"  {'target':<20} {getattr(store, 'repo_id', None) or getattr(store, 'bucket', None) or store.root}")
    print(f"  {'free tier':<20} {FREE_TIER_GB:.0f} GB")
    print()
    print(describe(usage, seconds_of_film=120.0, shots=400))

    if backend == "local":
        # Not an error: the local backend is right for development and for the
        # offline demo. But it has to say plainly that it is not durable,
        # because the only way to find out otherwise is to lose a film.
        print()
        print("  NOT DURABLE. This is a folder on one machine.")
        print("  SaladCloud deletes a container's disk the moment it stops, so")
        print("  on Salad this backend loses the ledger and every clip.")
        for line in how_to_configure("local"):
            print(f"  {line}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as demo_main

    demo_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="openclude", description="AI animation production machine")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="where the project stands").set_defaults(fn=cmd_status)
    sub.add_parser("doctor", help="check this machine").set_defaults(fn=cmd_doctor)
    sub.add_parser("store", help="check the object store").set_defaults(fn=cmd_store)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        args = p.parse_args(["status"])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
