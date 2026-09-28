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
# is not configured, refuse to start rather than render into a disk that vanishes.
#
# Any backend counts. Hugging Face is the default because the user has no credit
# card and R2's checkout requires one.
if [ "${OPENCLIDE_REQUIRE_STORE:-1}" = "1" ]; then
  BACKEND="${OPENCLIDE_STORE:-}"
  if [ -z "${BACKEND}" ]; then
    if [ -n "${HF_REPO:-}" ]; then BACKEND=hf
    elif [ -n "${S3_BUCKET:-}" ]; then BACKEND=s3
    else BACKEND=local
    fi
  fi

  if [ "${BACKEND}" = "hf" ]; then
    if [ -z "${HF_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
      log "FATAL: OPENCLIDE_STORE=hf needs HF_REPO and HF_TOKEN."
      exit 78
    fi
  elif [ "${BACKEND}" = "s3" ]; then
    if [ -z "${S3_ENDPOINT:-}" ] || [ -z "${S3_BUCKET:-}" ] \
       || [ -z "${S3_ACCESS_KEY_ID:-}" ] || [ -z "${S3_SECRET_ACCESS_KEY:-}" ]; then
      log "FATAL: OPENCLIDE_STORE=s3 needs S3_ENDPOINT, S3_BUCKET,"
      log "  S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY."
      exit 78
    fi
  elif [ "${BACKEND}" != "local" ]; then
    log "FATAL: OPENCLIDE_STORE=${BACKEND} is not one of local, s3, hf."
    exit 78
  fi

  if [ "${BACKEND}" = "local" ]; then
    log "FATAL: no durable object storage configured."
    log "  SaladCloud filesystems are ephemeral: a container's disk is deleted"
    log "  the moment it stops, so the ledger and every clip would go with it."
    log "  Free and no credit card: create a private HF dataset repo and set"
    log "  HF_REPO=<user>/<repo> and HF_TOKEN, or set OPENCLIDE_REQUIRE_STORE=0"
    log "  for a throwaway run you are willing to lose."
    exit 78   # EX_CONFIG
  fi

  log "object storage backend: ${BACKEND}"
  export OPENCLIDE_STORE="${BACKEND}"
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
# Deliberately nothing here.
#
# The weight download used to run in this script, before the worker started.
# That meant nothing served port 8000 for the three minutes the download took,
# the liveness probe counted that as failure, the container was killed, and
# `restart_policy: always` turned it into a loop. The group reported `running`
# and then `creating`, over and over, writing nothing at all.
#
# The download now lives in the worker, after the health server is listening,
# so /health answers within a second and reports "downloading model weights"
# while it happens. See WanGPAdapter.prefetch.
#
# The other reason to move it: this heredoc imported `wgp`, which is not the
# module anything calls. The renderer uses `shared.api`. The step could not
# have worked even if it had been reached.

# ------------------------------------------------------------------- 5. work
log "starting the openclude worker"
cd /app
exec python3.11 -m openclude.worker "$@"
