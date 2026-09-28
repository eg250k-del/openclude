"""Fetch models into the running ComfyUI container, through its own API.

The ComfyUI interface tells the user which models a template needs and offers a
Download button next to each. That button downloads straight to the container's
disk, which SaladCloud deletes when the container stops. So the download works
and then evaporates, which is the whole problem this project exists to solve.

This script does the same download from outside, and immediately mirrors the
file into the project's Hugging Face repo, so the next container can restore it
instead of pulling 18 GB again.

The one wrinkle: the `/download` endpoint lives on the API wrapper's port,
3000, while the interface the user opens is on 8188. The container gateway
forwards exactly one port. So the port is pointed at 3000, the models are
fetched, and it is pointed back at 8188. That is one restart, once, rather than
18 GB re-downloaded on every container.

Two things are never done here: writing a secret anywhere, and re-downloading a
file that is already present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import openclude.comfy_deploy as cd  # noqa: E402
from openclude.salad import (  # noqa: E402
    Config,
    SaladError,
    call,
    containers_path_for,
)
from openclude.storage_doctor import build_store  # noqa: E402

#: The two files the ltxv_text_to_video template asks for, verified with a
#: HEAD request on 2026-09-28: both 200, both from public repos with no gating.
#:
#: The checkpoint is from Lightricks/LTX-Video (gated=False). The text encoder
#: is from comfyanonymous/flux_text_encoders, which is where ComfyUI's own
#: templates point. Both are the exact filenames the interface asked for, so
#: the graph will load without editing the template.
MODELS = [
    {
        "filename": "ltx-video-2b-v0.9.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
               "ltx-video-2b-v0.9.safetensors",
        "model_type": "checkpoints",
    },
    {
        "filename": "t5xxl_fp16.safetensors",
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/"
               "resolve/main/t5xxl_fp16.safetensors",
        "model_type": "text_encoders",
    },
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api_base(port: int) -> str:
    return f"https://parmesan-jicama-ji54x1onjwo3px9i.salad.cloud:{port}"


def _post(url: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "openclude-fetch/0.1")
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {url} -> HTTP {e.code}: {e.read()[:300]}") from None
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"POST {url} -> {type(e).__name__}: {e}") from None


def set_gateway_port(port: int) -> None:
    """Point the container gateway at one port, then wait for the UI to answer.

    SaladCloud applies a networking change as a new version, so this restarts the
    container. It is done twice, not per model, because the download is the
    expensive part and it survives neither.
    """
    cfg = cd.ComfyConfig.from_env()
    salad_cfg = Config.from_env()
    path = containers_path_for(salad_cfg, cfg.group)

    live = call(salad_cfg, "GET", path)
    if int((live.get("networking") or {}).get("port", 0)) == port:
        log(f"  gateway already on {port}")
        return

    spec = cd.group_spec(cfg)
    spec["networking"]["port"] = port
    call(salad_cfg, "PATCH", path, spec)
    log(f"  gateway -> {port}, waiting for the container to come back")

    for _ in range(40):
        time.sleep(20)
        g = call(salad_cfg, "GET", path)
        cs = g.get("current_state") or {}
        counts = {k.replace("_count", ""): v
                  for k, v in (cs.get("instance_status_counts") or {}).items() if v}
        if counts.get("running"):
            time.sleep(30)
            log(f"  running, gateway on {port}")
            return
    raise RuntimeError(f"the container did not come back on port {port}")


def already_present(model: dict) -> bool:
    """Is this model saved in the repo already?

    18 GB is not something to re-fetch because someone restarted a container, and
    this is the one question the whole storage layer exists to answer.
    """
    key = f"models/{model['model_type']}/{model['filename']}"
    try:
        store = build_store()
    except Exception as exc:  # noqa: BLE001
        log(f"  storage unreadable, will download anyway: {str(exc)[:80]}")
        return False
    if key in store.sizes():
        log(f"  already saved: {key} "
            f"({store.sizes()[key] / 1024**3:.2f} GB)")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true",
                    help="block until each download finishes")
    ap.add_argument("--no-port-switch", action="store_true",
                    help="assume the gateway is already on the API port")
    args = ap.parse_args()

    if not os.environ.get("HF_REPO"):
        log("HF_REPO is not set. Source the env file first.")
        return 2
    # OPENCLIDE_IMAGE is only used by the film pipeline's Config validation and
    # is not needed to change a networking port. Its absence is not a reason to
    # refuse, and demanding it made the script fail before doing anything.
    os.environ.setdefault("OPENCLIDE_IMAGE", "unused-for-this-command")

    pending = [m for m in MODELS if not already_present(m)]
    if not pending:
        log("every model is already in the repo, nothing to download")
        return 0

    for m in pending:
        log(f"need {m['filename']}  ({m['model_type']})")

    if not args.no_port_switch:
        set_gateway_port(3000)
    base = api_base(3000)

    for m in pending:
        log(f"downloading {m['filename']} ...")
        try:
            r = _post(f"{base}/download",
                      {"url": m["url"], "model_type": m["model_type"],
                       "filename": m["filename"], "wait": args.wait},
                      timeout=7200 if args.wait else 120)
        except RuntimeError as e:
            log(f"  FAILED: {e}")
            return 1
        log(f"  accepted: {json.dumps(r)[:160]}")

    if not args.no_port_switch:
        set_gateway_port(cd.UI_PORT)
    log("gateway back on the interface port")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
