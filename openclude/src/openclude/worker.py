"""The long-running process inside the container.

Two jobs, in this order:

  1. serve `/health` and `/ready` so SaladCloud's probes work and a stuck
     instance gets recycled instead of sitting there billing
  2. pull jobs off the queue, one shot at a time, writing every decision to the
     ledger before and after each shot

Deliberately NOT here: the LLM client and the TTS client. Those are separate
Protocols so the worker can be exercised with fakes, and so swapping a
provider does not touch the loop.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .engine import EnginePaths, WanGPAdapter
from .pipeline import Config, Pipeline
from .state import Ledger
from .storage import Layout, LocalStore, S3Store
from .storage_doctor import build_store

log = logging.getLogger("openclude.worker")

PORT = int(os.environ.get("OPENCLIDE_PORT", "8000"))
DATA = Path(os.environ.get("OPENCLUDE_DATA", os.environ.get("OPENCLIDE_DATA", "/data")))
MODEL = os.environ.get("OPENCLIDE_MODEL", "ti2v_2_2_fastwan")
ENGINE_PROFILE = os.environ.get("WAN2GP_PROFILE", "3")
ENGINE_ATTENTION = os.environ.get("WAN2GP_ATTENTION", "sage2")


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


class _State:
    ready = False
    detail = "starting"
    shots_done = 0
    shots_total = 0


STATE = _State()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict) -> None:
        blob = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            # liveness only: is this process still thinking
            self._send(200, {"status": "ok", "detail": STATE.detail})
        elif self.path.startswith("/ready"):
            if STATE.ready:
                self._send(200, {"status": "ready", **self._progress()})
            else:
                self._send(503, {"status": "not-ready", "detail": STATE.detail})
        else:
            self._send(404, {"error": "not found"})

    def _progress(self) -> dict[str, Any]:
        return {
            "shots_done": STATE.shots_done,
            "shots_total": STATE.shots_total,
        }

    def log_message(self, *args: Any) -> None:  # silence per-request noise
        return


def serve_health() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="health").start()
    log.info("health server on :%d", PORT)
    return httpd


# --------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    story: str
    film_id: str
    target_minutes: int
    language: str = "ar"
    characters: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.characters is None:
            self.characters = []


def poll_queue(store: Any, prefix: str) -> list[Job]:
    """Read pending jobs from object storage.

    The job queue is a folder. That is deliberate: the Salad job-queue worker
    forwards an HTTP request, and this process can be killed at any instant, so
    anything held only in memory is a lost job. A folder of JSON files written
    before processing and deleted after is the smallest thing that survives.
    """
    jobs: list[Job] = []
    for key in store.list(prefix):
        if not key.endswith(".json"):
            continue
        try:
            raw = store.get(key, Path(DATA) / "work" / Path(key).name)
            data = json.loads(Path(raw).read_text("utf-8"))
            jobs.append(Job(**data))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable job %s: %s", key, exc)
    return jobs


def run_job(job: Job, store: Any, deps: dict) -> int:
    """Run one job to completion, or as far as the container's life allows."""
    from .llm import ScriptError
    from .schema import Character, CharacterView

    STATE.shots_total = 0
    cast = [
        Character(
            id=c["id"],
            name=c.get("name", c["id"]),
            description=c.get("description", ""),
            views=tuple(
                CharacterView(angle=v["angle"], image_path=v["image_path"])
                for v in c.get("views", [])
            ),
            voice_reference=c.get("voice_reference", ""),
        )
        for c in job.characters
    ]

    layout = Layout(job.film_id, base=DATA / "work")
    cfg = Config(
        film_id=job.film_id,
        language=job.language,
        target_minutes=job.target_minutes,
        model_type=MODEL,
        work_dir=str(DATA / "work"),
    )

    adapter = deps["engine"]
    pipeline = Pipeline(
        cfg, store, deps["writer"], deps["synth"], adapter.render, cast
    )

    STATE.detail = f"film={job.film_id}"
    try:
        result = pipeline.run(job.story)
    except ScriptError as exc:
        log.error("job %s: script stage failed: %s", job.id, exc)
        _finish(store, job, ok=False, note=str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        _finish(store, job, ok=False, note=f"{type(exc).__name__}: {exc}")
        return 1

    log.info("job %s: %s", job.id, result.line())
    _finish(store, job, ok=result.complete, note=result.output)
    return 0 if result.complete else 1


def _finish(store: Any, job: Job, ok: bool, note: str) -> None:
    store.put(
        f"queue/done/{job.id}.json",
        _write_json({"id": job.id, "ok": ok, "note": note, "at": time.time()}),
    )


def _write_json(data: dict) -> str:
    p = Path(DATA) / "work" / f"out-{os.getpid()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    httpd = serve_health()

    stopping = threading.Event()

    def shutdown(signum: int, _frame: Any) -> None:
        # A container being preempted arrives here. Stop claiming new work, let
        # the ledger finish its write, and exit. The next instance resumes.
        log.warning("signal %s received; finishing the current shot and stopping", signum)
        stopping.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    store = build_store()
    STATE.detail = "downloading model weights"

    adapter = WanGPAdapter(
        paths=EnginePaths(
            root=os.environ.get("WAN2GP_ROOT", "/opt/WanGP"),
            config=str(DATA / "config"),
            output=str(DATA / "outputs"),
            frames=str(DATA / "frames"),
            audio=str(DATA / "audio"),
        ),
        attention=ENGINE_ATTENTION,
        profile=int(ENGINE_PROFILE),
    )

    STATE.detail = "loading the engine"
    session = None
    try:
        adapter.start()
        session = adapter.session
    except Exception as exc:  # noqa: BLE001
        # Do not exit. A cold start can fail while weights are still landing;
        # staying alive lets the liveness probe pass and the retry happen.
        log.error("engine did not start: %s", exc)
        STATE.detail = f"engine unavailable: {exc}"

    # Weights, if they are not there yet, are fetched after the health server
    # is already listening. Not before: see WanGPAdapter.prefetch. The engine's
    # own init will not notice a partial download, so a missing VAE surfaces as
    # an unrelated error much later, and a preflight that says which files it
    # just fetched turns that into a log line.
    try:
        fetched = adapter.prefetch(MODEL)
        log.info("weights present: %s", ", ".join(fetched) or "(engine reported none)")
        STATE.detail = "loading the engine"
    except Exception as exc:  # noqa: BLE001
        # Not fatal. The engine's own init may still find what it needs, and
        # refusing to start over a prefetch would lose a working render.
        log.warning("weight prefetch failed: %s", exc)

    # Built after start(), not before: the speech backend picks engine mode
    # based on having a live session, and adapter.render needs the session too.
    deps = {
        "engine": adapter,
        "writer": deps_writer(),
        "synth": deps_synth(session),
    }
    log.info("speech backend: %s", type(deps["synth"]).__name__)

    STATE.ready = True
    STATE.detail = "idle"
    log.info("worker ready")

    while not stopping.is_set():
        try:
            jobs = poll_queue(store, "queue/pending/")
        except Exception as exc:  # noqa: BLE001
            log.warning("queue poll failed: %s", exc)
            jobs = []
        for job in jobs:
            if stopping.is_set():
                break
            try:
                store.delete(f"queue/pending/{job.id}.json")
            except Exception:  # noqa: BLE001
                pass
            run_job(job, store, deps)
            STATE.shots_done = 0
        time.sleep(5)

    log.info("stopping cleanly")
    try:
        adapter.stop()
    finally:
        httpd.shutdown()
    return 0


def deps_writer() -> Any:
    """The ScriptWriter.

    Returns the real client when one is configured, and a writer that refuses
    loudly when one is not. A silent default here would let a job reach the
    render stage with a script that was never written.
    """
    from .llm import ScriptError

    if os.environ.get("LLM_API_KEY", "").strip():
        from .llm_client import writer_from_env

        log.info("script writer: OpenAI-compatible endpoint, model %s",
                 os.environ.get("LLM_MODEL", "default"))
        return writer_from_env()

    def missing(*_a: Any, **_k: Any) -> str:
        raise ScriptError(
            "no LLM configured. Set LLM_API_KEY (and LLM_BASE_URL, LLM_MODEL). "
            "The SaladCloud AI Gateway works: LLM_BASE_URL="
            "https://api.salad.com/api/public with a gateway key."
        )

    return type("UnconfiguredWriter", (), {"write": staticmethod(missing)})()


def deps_synth(session: Any = None) -> Any:
    """The Synthesiser.

    Engine mode is preferred when the session is up, because it clones a
    distinct voice per character and reads `[emotion]` tags, which the HTTP
    backends do not.

    A misconfiguration is never swallowed. Only "engine mode with no session
    yet" falls back to a synth that refuses, because that is a real ordering
    state during a cold start; a bad TTS_MODE is the operator's typo and
    replacing its message would hide the actual problem.
    """
    from .audio import AudioError
    from .tts_client import EngineSynth, OpenAICompatSynth, synth_from_env

    try:
        return synth_from_env(session)
    except AudioError as exc:
        if session is None and "needs the engine session" in str(exc):
            log.warning("engine speech unavailable during startup: %s", exc)
            return _RefusingSynth(
                "no speech backend is available. The engine session has not "
                "finished loading and no HTTP backend is configured. Set "
                "TTS_API_KEY / TTS_BASE_URL / TTS_MODEL to speak without the engine."
            )
        raise


def _RefusingSynth(message: str) -> Any:
    """A synth that exists only to fail with a useful message."""

    def speak(*_a: Any, **_k: Any) -> str:
        from .audio import AudioError

        raise AudioError(message)

    return type("RefusingSynth", (), {"speak": staticmethod(speak)})()


if __name__ == "__main__":
    raise SystemExit(main())
