"""Tests for the in-container worker and its health probes.

A probe that lies is worse than no probe: a broken instance that keeps
reporting healthy bills until you notice.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from openclude import worker
from openclude.state import Ledger
from openclude.storage import LocalStore


# --------------------------------------------------------------------------
# the probes
# --------------------------------------------------------------------------


@pytest.fixture
def server():
    worker.STATE.ready = False
    worker.STATE.detail = "starting"
    httpd = worker.serve_health.__wrapped__() if hasattr(worker.serve_health, "__wrapped__") else None
    # serve_health starts its own thread on the configured port; use a local
    # server on an ephemeral port instead so tests never collide
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def get(base: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def get_status(base: str, path: str) -> tuple[int, dict]:
    """Like get(), but returns non-2xx instead of raising."""
    import urllib.error

    try:
        return get(base, path)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_is_ok_even_before_ready(server) -> None:
    """Liveness must pass while the engine loads, or the node gets recycled
    during a multi-minute model download."""
    worker.STATE.ready = False
    status, body = get(server, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["detail"] == "starting"


def test_ready_returns_503_until_the_worker_is_up(server) -> None:
    worker.STATE.ready = False
    status, body = get_status(server, "/ready")
    assert status == 503
    assert body["status"] == "not-ready"


def test_ready_flips_once_the_worker_starts(server) -> None:
    worker.STATE.ready = True
    worker.STATE.shots_done = 3
    worker.STATE.shots_total = 10
    status, body = get(server, "/ready")
    assert status == 200
    assert body["status"] == "ready"
    assert body["shots_done"] == 3
    assert body["shots_total"] == 10


def test_health_reports_the_detail_string(server) -> None:
    worker.STATE.detail = "engine unavailable: no driver"
    _, body = get(server, "/health")
    assert "no driver" in body["detail"]


def test_an_unknown_path_is_404(server) -> None:
    try:
        get(server, "/nope")
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


# --------------------------------------------------------------------------
# the queue: a folder, so a killed container cannot lose a job
# --------------------------------------------------------------------------


def test_a_pending_job_is_read_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "DATA", tmp_path / "data")
    store = LocalStore(tmp_path / "bucket")
    job = {"id": "j1", "story": "a pilot", "film_id": "f01", "target_minutes": 1}
    store.put("queue/pending/j1.json", worker._write_json(job))
    found = worker.poll_queue(store, "queue/pending/")
    assert [j.id for j in found] == ["j1"]
    assert found[0].story == "a pilot"


def test_a_corrupt_job_is_skipped_not_fatal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "DATA", tmp_path / "data")
    store = LocalStore(tmp_path / "bucket")
    good = worker._write_json({"id": "j2", "story": "s", "film_id": "f02", "target_minutes": 1})
    store.put("queue/pending/j2.json", good)
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    store.put("queue/pending/j1.json", broken)

    found = worker.poll_queue(store, "queue/pending/")
    assert [j.id for j in found] == ["j2"]


def test_non_json_files_are_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "DATA", tmp_path / "data")
    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "x.txt"
    src.write_bytes(b"hello")
    store.put("queue/pending/readme.txt", src)
    assert worker.poll_queue(store, "queue/pending/") == []


def test_an_empty_queue_is_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "DATA", tmp_path / "data")
    store = LocalStore(tmp_path / "bucket")
    assert worker.poll_queue(store, "queue/pending/") == []


def test_a_job_defaults_its_character_list() -> None:
    job = worker.Job(id="j", story="s", film_id="f", target_minutes=1)
    assert job.characters == []
    assert job.language == "ar"


# --------------------------------------------------------------------------
# the unconfigured clients fail loudly instead of silently
# --------------------------------------------------------------------------


def test_the_missing_writer_says_what_to_do(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(worker, "DATA", tmp_path / "data")
    from openclude.llm import ScriptError

    with pytest.raises(ScriptError, match="ScriptWriter protocol"):
        worker.deps_writer().write("p", [])


def test_the_missing_synth_says_what_to_do() -> None:
    from openclude.audio import AudioError

    with pytest.raises(AudioError, match="Synthesiser protocol"):
        worker.deps_synth().speak("t", "o.wav")


# --------------------------------------------------------------------------
# configuration is read from the environment, not hardcoded
# --------------------------------------------------------------------------


def test_the_profile_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("WAN2GP_PROFILE", "3")
    assert worker.ENGINE_PROFILE == "3"


def test_the_default_profile_is_three_not_five(monkeypatch) -> None:
    """Profile 5 is a 10 GB fail-safe; a 4090 is profile 3."""
    monkeypatch.delenv("WAN2GP_PROFILE", raising=False)
    import importlib

    reloaded = importlib.reload(worker)
    assert reloaded.ENGINE_PROFILE == "3"
    importlib.reload(worker)
