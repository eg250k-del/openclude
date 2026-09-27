#!/usr/bin/env bash
# Container entrypoint.
#
# Responsibilities, in order:
#   1. make sure no crash lock is lying around, or the engine boots into Safe
#      Mode and silently disables its plugins
#   2. verify the durable directories are actually backed by object storage,
#      because a container whose disk is not synced will silently lose a film
#   3. pull the model weights once, into the store-backed ckpts directory
#   4. hand over to the openclude worker, which owns the ledger
#
# Model weights are fetched through the engine's own downloader so that the
# URL list stays in one place: the engine's defaults/<model>.json files. Adding
# a model here means editing the engine, not this script.

set -euo pipefail

DATA="${OPENCLIDE_DATA:-/data}"
ENGINE="${WAN2GP_ROOT:-/opt/WanGP}"
MODEL="${OPENCLIDE_MODEL:-ti2v_2_2_fastwan}"

log() { printf '%s [entrypoint] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# ---------------------------------------------------------------- 1. crash lock
# The engine writes startup.lock and, if it finds one, waits then enters Safe
# Mode with plugins disabled. There is no --no-safe-mode flag. A container that
# was preempted mid-render therefore comes back crippled unless this is removed.
rm -f "${ENGINE}/startup.lock" "${DATA}/config/startup.lock"
log "cleared any crash lock"

# ------------------------------------------------------------- 2. durable disk
# This is the check that would have saved the models lost earlier. If the store
# is not mounted, refuse to start rather than render into a disk that vanishes.
if [ "${OPENCLIDE_REQUIRE_STORE:-1}" = "1" ]; then
  if [ -z "${S3_ENDPOINT:-}" ] && [ -z "${S3_ACCESS_KEY_ID:-}" ]; then
    log "FATAL: no object storage configured."
    log "  SaladCloud filesystems are ephemeral. Set S3_ENDPOINT,"
    log "  S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY, or set"
    log "  OPENCLIDE_REQUIRE_STORE=0 for a throwaway local run."
    exit 78   # EX_CONFIG
  fi
  log "object storage configured"
fi

mkdir -p "${DATA}"/{ckpts,loras,outputs,frames,audio,cache,config,work}

# ---------------------------------------------------------------- 3. GPU facts
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
  log "WARNING: nvidia-smi is absent. The image is wrong or the node has no GPU."
fi

python3.11 - <<'PY' || log "WARNING: torch reports no CUDA device"
import torch, sys
print(f"[entrypoint] torch {torch.__version__} cuda={torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY

# ---------------------------------------------------------------- 4. weights
# Fetched through the engine so the URL list lives in defaults/*.json. Skipped
# when the files are already present, which is the normal case after the first
# run, because ckpts/ is store-backed.
if [ "${OPENCLIDE_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  log "ensuring model weights for ${MODEL}"
  python3.11 - "$ENGINE" "$MODEL" <<'PY' || log "WARNING: weight prefetch failed; the render stage will retry"
import importlib, json, sys
from pathlib import Path

engine, model = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(engine))
importlib.import_module("wgp")

definition = json.loads((engine / "defaults" / f"{model}.json").read_text("utf-8"))
need = [u for u in definition.get("URLs", []) if "quanto" in u or "int8" in u] or definition.get("URLs", [])
need = need[:1] + definition.get("VAE_URLs", [])[:1]
need += definition.get("text_encoder_URLs", [])[:1]

for key in ("URLs2", "loras", "preload_URLs"):
    extra = definition.get(key, [])
    if isinstance(extra, list) and extra and isinstance(extra[0], str):
        need += extra[:1]

from shared.utils.download import download_file
from shared.utils.files_locator import get_smart_download_location

for url in need:
    name = url.rsplit("/", 1)[-1]
    target = get_smart_download_location(name)
    if Path(target).exists() and Path(target).stat().st_size > 0:
        print(f"[entrypoint] have {name}")
        continue
    print(f"[entrypoint] fetching {name}")
    download_file(url, target)
print("[entrypoint] weights ready")
PY
fi

# ------------------------------------------------------------------- 5. work
log "starting the openclude worker"
cd /app
exec python3.11 -m openclude.worker "$@"
