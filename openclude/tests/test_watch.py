"""Tests for the unattended watcher.

The watcher exists because the person paying is not at the keyboard. That
raises the stakes on two behaviours, and both are tested here:

  * it must stop the group when its budget runs out. A GPU left running bills
    by the hour, and "I will check later" is how that becomes a morning of
    surprise.
  * it must report a failure instead of waiting. `allocating` and `creating`
    look identical from outside, and the reason is only in the container logs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import openclude.salad as salad


WATCH = Path(__file__).resolve().parents[1] / "tools" / "watch.py"


@pytest.fixture
def watcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("watch", WATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def config(monkeypatch):
    for name in ("OPENCLIDE_IMAGE", "SALAD_ORGANIZATION", "SALAD_PROJECT",
                 "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")
    return salad.Config.from_env()


def _group(status: str, counts: dict, desc: str = "") -> dict:
    return {"current_state": {"status": status,
                              "instance_status_counts": counts,
                              "description": desc}}


def test_the_watcher_stops_the_group_when_the_budget_runs_out(watcher, config,
                                                              monkeypatch) -> None:
    """The whole point of a budget: it stops the billing."""
    calls: list[tuple] = []

    monkeypatch.setattr(watcher, "call",
                        lambda cfg, m, p, b=None: calls.append((m, p)) or {})
    monkeypatch.setattr(watcher.sys, "argv", ["watch.py", "1", "1"])
    monkeypatch.setattr(watcher.time, "sleep", lambda _s: None)

    assert watcher.main() == 2
    assert ("POST", f"{watcher.containers_path_for(config)}/stop") in calls


def test_the_watcher_stops_on_failure_and_does_not_keep_waiting(watcher, config,
                                                                monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_call(cfg, m, p, b=None):
        calls.append((m, p))
        if m == "GET" and p.endswith("/instances"):
            return {"items": [{"id": "i1"}]}
        if m == "GET" and "/logs" in p:
            return {"items": [{"message": "entrypoint: FATAL: no storage"}]}
        if m == "GET":
            return _group("failed", {})
        return {}

    monkeypatch.setattr(watcher, "call", fake_call)
    monkeypatch.setattr(watcher.sys, "argv", ["watch.py", "1", "1"])
    monkeypatch.setattr(watcher.time, "sleep", lambda _s: None)
    captured: list[str] = []
    monkeypatch.setattr(watcher, "log", lambda m: captured.append(m))

    assert watcher.main() == 1
    assert any("FATAL: no storage" in m for m in captured), "the reason was lost"
    assert ("POST", f"{watcher.containers_path_for(config)}/stop") in calls


def test_a_running_instance_is_reported_with_its_logs(watcher, config,
                                                      monkeypatch) -> None:
    def fake_call(cfg, m, p, b=None):
        if m == "GET" and p.endswith("/instances"):
            return {"items": [{"id": "i1", "state": "running"}]}
        if m == "GET" and "/logs" in p:
            return {"items": [{"message": "wrote 15 shots"}]}
        if m == "GET":
            return _group("running", {"running_count": 1})
        return {}

    monkeypatch.setattr(watcher, "call", fake_call)
    monkeypatch.setattr(watcher.sys, "argv", ["watch.py", "1", "1"])
    monkeypatch.setattr(watcher.time, "sleep", lambda _s: None)
    captured: list[str] = []
    monkeypatch.setattr(watcher, "log", lambda m: captured.append(m))

    watcher.main()
    assert any("wrote 15 shots" in m for m in captured)


def test_a_running_group_is_only_stopped_when_the_budget_runs_out(watcher, config,
                                                                   monkeypatch) -> None:
    """Not "never stopped". Stopped at the end, and not one poll before.

    A running instance is a film being made, and stopping it is the caller's
    decision. But leaving it running past the budget is money, and the person
    who would have stopped it is not at the keyboard. So the rule is precise:
    keep watching while there is budget, stop when there is not.
    """
    polls = {"n": 0}
    stops: list[int] = []

    def fake_call(cfg, m, p, b=None):
        if m == "POST" and p.endswith("/stop"):
            stops.append(polls["n"])
            return {}
        if m == "GET" and p.endswith("/instances"):
            return {"items": [{"id": "i1", "state": "running"}]}
        if m == "GET" and "/logs" in p:
            return []
        if m == "GET":
            polls["n"] += 1
            return _group("running", {"running_count": 1})
        return {}

    monkeypatch.setattr(watcher, "call", fake_call)
    monkeypatch.setattr(watcher.sys, "argv", ["watch.py", "5", "1"])
    monkeypatch.setattr(watcher.time, "sleep", lambda _s: None)
    monkeypatch.setattr(watcher, "log", lambda _m: None)

    watcher.main()
    assert stops, "it must stop when the budget ends"
    assert stops[0] >= 5, f"it stopped on poll {stops[0]} of 5"


def test_a_transient_read_error_does_not_end_the_watch(watcher, config,
                                                        monkeypatch) -> None:
    """A 500 or a blip must not look like a failure."""
    state = {"n": 0}

    def fake_call(cfg, m, p, b=None):
        if m == "GET":
            state["n"] += 1
            if state["n"] < 3:
                raise salad.SaladError("503 Service Unavailable")
            return _group("stopped", {})
        return {}

    monkeypatch.setattr(watcher, "call", fake_call)
    monkeypatch.setattr(watcher.sys, "argv", ["watch.py", "1", "1"])
    monkeypatch.setattr(watcher.time, "sleep", lambda _s: None)
    assert watcher.main() == 2          # reached the budget, not a false failure


def test_the_store_report_survives_a_broken_store(watcher, config) -> None:
    """Progress reporting must not be the thing that crashes the watcher."""
    out = watcher.store_progress(config)
    assert isinstance(out, str)
    assert out
