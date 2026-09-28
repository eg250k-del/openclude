"""Watch a SaladCloud group until it runs, fails, or the budget expires.

Runs unattended on purpose. The user is away and the answer should arrive on
its own rather than as a question.

Two things this does that a human would not do well:

  * it pulls the container logs the moment an instance exists, which is the
    only place the real reason for a failure appears. `allocating` for five
    minutes and `creating` for five minutes look identical from the outside.
  * it stops the group when the budget runs out. Leaving a GPU running is
    money, and a person who is not at the keyboard is exactly when that
    matters.

Usage:  python watch.py <seconds> [poll-seconds]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclude.salad import (  # noqa: E402
    Config,
    SaladError,
    call,
    containers_path_for,
    group_name,
    check_capacity,
    safe_json,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def instance_ids(cfg: Config, path: str) -> list[str]:
    try:
        r = call(cfg, "GET", f"{path}/instances")
    except SaladError as e:
        log(f"instance list failed: {str(e)[:120]}")
        return []
    items = r if isinstance(r, list) else r.get("items", [])
    return [i.get("id") for i in items if i.get("id")]


def last_logs(cfg: Config, iid: str, limit: int = 60) -> str:
    """Container logs for one instance.

    The endpoint is the organisation's `log-entries`, filtered by instance.
    The first version of this used `/instances/<id>/logs`, which does not
    exist, so it returned nothing and the watcher reported "RUNNING" with no
    output for seventeen minutes while the container was crash-looping. An empty
    answer from a wrong URL looks exactly like a silent application, which is
    the one thing a watcher must never pretend to know.
    """
    try:
        r = call(cfg, "GET",
                 f"/organizations/{cfg.organization}/log-entries?instance_id={iid}")
    except SaladError as e:
        return f"(logs unavailable: {str(e)[:160]})"
    if isinstance(r, str):
        return r[-limit * 200:] or "(empty response)"
    if isinstance(r, list):
        entries = r
    elif isinstance(r, dict):
        entries = r.get("logs") or r.get("items") or r.get("entries") or []
    else:
        entries = []
    lines = []
    for entry in entries:
        if isinstance(entry, str):
            lines.append(entry)
        else:
            stamp = entry.get("created_at", "")
            stream = entry.get("stream", "")
            msg = entry.get("message", "")
            lines.append(f"{stamp} {stream:<6} {msg}")
    if not lines:
        return "(no log entries for this instance)"
    return "\n".join(lines[-limit:])


def store_progress(cfg: Config) -> str:
    """What the worker has written so far. This is the actual progress."""
    try:
        from openclude.storage_doctor import build_store, measure
        s = build_store()
        keys = sorted(s.list("f01/"))
        usage = measure(s)
        return (f"{len(keys)} objects, {usage.bytes_total / 1024**2:.1f} MB"
                f" | {keys[-3:] if keys else 'nothing yet'}")
    except Exception as e:  # noqa: BLE001
        return f"store unreadable: {str(e)[:100]}"


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cfg = Config.from_env()
    path = containers_path_for(cfg)
    log(f"watching {group_name(cfg)} for up to {budget}s, polling {poll}s")

    cap = check_capacity(cfg)
    log(f"capacity: {cap}")

    deadline = time.time() + budget
    last_state = None
    logged_for: set[str] = set()

    # Bounded by poll count, not only by wall time. budget // poll is the same
    # number the wall clock would produce, but it is a number a test can make
    # small, so the loop is never an untestable infinite spin.
    max_polls = max(1, int(budget / max(1, poll)))

    for _ in range(max_polls):
        if time.time() >= deadline:
            log("wall-clock budget reached")
            break
        try:
            g = call(cfg, "GET", path)
        except SaladError as e:
            log(f"read failed: {str(e)[:140]}")
            time.sleep(poll)
            continue

        cs = g.get("current_state") or {}
        status = cs.get("status")
        counts = {k.replace("_count", ""): v
                  for k, v in (cs.get("instance_status_counts") or {}).items() if v}
        desc = (cs.get("description") or "")[:70]

        line = f"{status} {counts} {desc}".strip()
        if line != last_state:
            log(f"state: {line}")
            last_state = line

        if counts.get("running") or status == "running":
            log("RUNNING. container logs:")
            for iid in instance_ids(cfg, path):
                if iid not in logged_for:
                    logged_for.add(iid)
                    log(f"--- instance {iid} ---\n{last_logs(cfg, iid)}")
            log(f"store: {store_progress(cfg)}")
            # leave it running; the caller decides when to stop
            time.sleep(poll)
            continue

        if status == "failed":
            log("FAILED. logs:")
            for iid in instance_ids(cfg, path):
                if iid not in logged_for:
                    logged_for.add(iid)
                    log(f"--- instance {iid} ---\n{last_logs(cfg, iid)}")
            log("stopping the group so it does not keep billing")
            try:
                call(cfg, "POST", f"{path}/stop", {})
            except SaladError as e:
                log(f"stop refused: {str(e)[:120]}")
            return 1

        time.sleep(poll)

    log("budget exhausted, stopping the group so it does not keep billing")
    try:
        call(cfg, "POST", f"{path}/stop", {})
    except SaladError as e:
        log(f"stop refused: {str(e)[:120]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
