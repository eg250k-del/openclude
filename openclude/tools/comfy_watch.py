"""Deploy the ComfyUI group, start it, and watch until the interface answers.

Run unattended. The person paying is not at the keyboard, and the one thing
they want is a URL in their browser.

The ordering is the whole design. ComfyUI is useless to the user until the web
UI answers, so the URL is found and printed before anything else is reported,
and the watcher keeps going through failures rather than stopping at the first
one, because "the container died" and "the container is up but the manifest
failed" are very different problems and only the logs tell them apart.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import openclude.comfy_deploy as cd  # noqa: E402
from openclude.salad import (  # noqa: E402
    Config,
    SaladError,
    call,
    check_capacity,
    containers_path,
    group_name,
    safe_json,
)

UI_PORT = cd.UI_PORT


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_path(cfg: cd.ComfyConfig) -> str:
    return f"{containers_path(Config.from_env())}/{cfg.group}"


def instances(salad_cfg: Config, path: str) -> list[dict]:
    try:
        r = call(salad_cfg, "GET", f"{path}/instances")
    except SaladError as e:
        log(f"instances unreadable: {str(e)[:120]}")
        return []
    return r if isinstance(r, list) else r.get("items", [])


def logs_for(salad_cfg: Config, iid: str, limit: int = 80) -> str:
    """Container logs, from the endpoint that exists.

    The organisation's log-entries, filtered by instance. The first version of
    the watcher used /instances/<id>/logs, which does not exist, so it printed
    nothing and reported "running" for seventeen minutes while the container
    was crash-looping. A wrong URL and a silent application look identical.
    """
    try:
        r = call(salad_cfg, "GET",
                 f"/organizations/{salad_cfg.organization}/log-entries?instance_id={iid}")
    except SaladError as e:
        return f"(logs unavailable: {str(e)[:140]})"
    if isinstance(r, str):
        return r[-4000:]
    entries = r.get("logs", []) if isinstance(r, dict) else r
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append(e)
        else:
            out.append(f"{e.get('created_at','')} {e.get('stream',''):<6} "
                       f"{e.get('message','')}")
    return "\n".join(out[-limit:]) or "(no log entries)"


def find_url(salad_cfg: Config, cfg: cd.ComfyConfig) -> str:
    """The address the user opens. This is the deliverable.

    SaladCloud gives every running container group an access domain. It is not
    in the group representation, so it has to be read from the instance or from
    the gateway API, and every endpoint tried is recorded when none works,
    because "I could not find the URL" is useless to somebody waiting.
    """
    path = group_path(cfg)
    for inst in instances(salad_cfg, path):
        for field in ("access_domain_name", "access_url", "url", "gateway_url"):
            value = inst.get(field)
            if value:
                return value if str(value).startswith("http") else f"https://{value}"
    for endpoint in (f"{path}/access-domain", f"/organizations/{salad_cfg.organization}/gateway"):
        try:
            r = call(salad_cfg, "GET", endpoint)
        except SaladError as e:
            log(f"  {endpoint} -> {str(e)[:90]}")
            continue
        blob = safe_json(r)
        m = re_url(blob)
        if m:
            return m
        log(f"  {endpoint} -> 200 but no URL in it")
    return ""


def re_url(blob: str) -> str:
    import re
    m = re.search(r"https://[a-z0-9.-]+\.salad\.cloud", blob)
    return m.group(0) if m else ""


def wait_for_capacity(salad_cfg: Config, cfg: cd.ComfyConfig, budget: int,
                      poll: int) -> str:
    """Wait for any usable GPU, and name it.

    Two mistakes are avoided here. The first version pinned one GPU class and
    gave up when it was busy, which is not the same as there being no capacity:
    a 3090 Ti, an A5000 and a 4090 are three different pools, and at any moment
    at least one of them usually has something free. The second mistake was
    exiting when nothing was free. The person paying is not at the keyboard, and
    "no capacity right now" is a normal condition on a marketplace of
    residential PCs, not a failure to report and walk away from.

    So: check every class, take whatever is free, and if nothing is, keep
    checking until the budget runs out.
    """
    from openclude.salad import GPU_CLASSES

    order = [cfg.gpu] + [g for g in GPU_CLASSES if g != cfg.gpu]
    deadline = time.time() + budget
    announced = None
    while time.time() < deadline:
        free: list[tuple[str, int]] = []
        for name in order:
            probe = Config(
                api_key=salad_cfg.api_key,
                organization=salad_cfg.organization,
                project=salad_cfg.project,
                gpu=name, priority=cfg.priority, cpu=cfg.cpu,
                memory_mb=cfg.memory_mb, storage_bytes=cfg.storage_bytes,
            )
            try:
                r = check_capacity(probe)
            except Exception:  # noqa: BLE001
                continue
            if r.get("ok"):
                free.append((name, int(r.get("free") or 0)))
        if free:
            name, count = free[0]
            log(f"capacity found: {count} x {name} free at {cfg.priority}")
            return name
        if announced != "none":
            log(f"no capacity at {cfg.priority} for any of "
                f"{', '.join(order)}. Waiting, not billing.")
            announced = "none"
        time.sleep(poll)
    return ""


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2700
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cfg = cd.ComfyConfig.from_env()
    salad_cfg = Config.from_env()

    log(f"image : {cfg.image}")
    log(f"group : {cfg.group}")
    log(f"ui    : port {UI_PORT}")

    # Half the budget looking for a GPU, half running. The search can legitimately
    # take a long time, and the run has to leave room to actually happen.
    gpu = wait_for_capacity(salad_cfg, cfg, budget // 2, poll)
    if not gpu:
        log("no GPU became available. Nothing was started and nothing was billed.")
        return 1
    cfg.gpu = gpu
    log(f"gpu   : {cfg.gpu} / {cfg.priority}")

    spec = cd.group_spec(cfg)
    log("spec: " + " ".join(
        f"{k}={v}" for k, v in (
            ("image", spec["container"]["image"]),
            ("memory_mb", spec["container"]["resources"]["memory"]),
            ("storage_gb", spec["container"]["resources"]["storage_amount"] // 1024**3),
            ("gpu", spec["container"]["resources"]["gpu_classes"]),
            ("port", spec["networking"]["port"]),
        )))

    base = containers_path(salad_cfg)
    path = f"{base}/{cfg.group}"
    try:
        existing = call(salad_cfg, "GET", path)
        log("group exists; patching")
        call(salad_cfg, "PATCH", path, spec, )
    except SaladError:
        log("creating group")
        try:
            call(salad_cfg, "POST", base, spec)
        except SaladError as e:
            log(f"create FAILED: {e}")
            return 1

    log("waiting for the image to be pulled and the group to settle")
    settled = False
    for _ in range(20):
        try:
            g = call(salad_cfg, "GET", path)
        except SaladError as e:
            log(f"  read: {str(e)[:110]}")
            time.sleep(15)
            continue
        cs = g.get("current_state") or {}
        log(f"  {cs.get('status')}  {cs.get('description') or ''}"[:150])
        if cs.get("status") in ("stopped", "running", "failed"):
            settled = True
            break
        time.sleep(15)
    if not settled:
        log("never settled")
        return 1

    try:
        call(salad_cfg, "POST", f"{path}/start", {})
        log("start accepted. the model download is about 30 GB on a home connection")
    except SaladError as e:
        log(f"start refused: {str(e)[:200]}")
        return 1

    # ------------------------------------------------------------------ watch
    deadline = time.time() + budget
    last = None
    logged: set[str] = set()
    max_polls = max(1, int(budget / max(1, poll)))
    reported_url = False

    for _ in range(max_polls):
        try:
            g = call(salad_cfg, "GET", path)
        except SaladError as e:
            log(f"read failed: {str(e)[:120]}")
            time.sleep(poll)
            continue
        cs = g.get("current_state") or {}
        counts = {k.replace("_count", ""): v
                  for k, v in (cs.get("instance_status_counts") or {}).items() if v}
        line = f"{cs.get('status')} {counts} {cs.get('description') or ''}"[:160]
        if line != last:
            log(f"state: {line}")
            last = line

        if counts.get("running") or cs.get("status") == "running":
            url = find_url(salad_cfg, cfg)
            if url and not reported_url:
                log("")
                log(f"  >>> OPEN THIS IN YOUR BROWSER:  {url}")
                log("")
                reported_url = True
            for inst in instances(salad_cfg, path):
                iid = inst.get("id")
                if iid and iid not in logged:
                    logged.add(iid)
                    log(f"--- logs, instance {iid} ---\n{logs_for(salad_cfg, iid)}")
            time.sleep(poll)
            continue

        if cs.get("status") == "failed":
            log("FAILED. logs:")
            for inst in instances(salad_cfg, path):
                iid = inst.get("id")
                if iid and iid not in logged:
                    logged.add(iid)
                    log(f"--- logs, instance {iid} ---\n{logs_for(salad_cfg, iid)}")
            log("stopping so it does not keep billing")
            try:
                call(salad_cfg, "POST", f"{path}/stop", {})
            except SaladError as e:
                log(f"stop refused: {str(e)[:100]}")
            # Not the end. A group can fail because the node it landed on went
            # away mid-pull, which is ordinary on residential hardware, and the
            # next attempt on a different node is often fine. The logs above say
            # which it was.
            left = deadline - time.time()
            if left > 120:
                log(f"retrying in 60s, {int(left / 60)} min of budget left")
                time.sleep(60)
                try:
                    call(salad_cfg, "POST", f"{path}/start", {})
                except SaladError as e:
                    log(f"  restart refused: {str(e)[:120]}")
                logged.clear()
                last = None
                continue
            return 1

        time.sleep(poll)

    log("budget reached, stopping so it does not keep billing")
    try:
        call(salad_cfg, "POST", f"{path}/stop", {})
    except SaladError as e:
        log(f"stop refused: {str(e)[:100]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
