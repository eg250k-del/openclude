"""Tests for the ComfyUI watcher.

Two behaviours matter more than the rest, and both were wrong in the first
version:

  * it must not give up when no GPU is free. SaladCloud is a marketplace of
    residential PCs; "nothing free right now" is a normal condition, and the
    person paying is not at the keyboard to press retry.
  * it must not give up when a group fails. A node that vanishes mid-pull is
    ordinary on home hardware, and the next attempt is often fine.

And one that matters most:

  * it must print the URL. The entire purpose of this project is a link the
    user can open, and a watcher that runs for forty minutes and reports
    nothing useful has failed at its job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import openclude.salad as salad

WATCH = Path(__file__).resolve().parents[1] / "tools" / "comfy_watch.py"


@pytest.fixture
def watcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("comfy_watch", WATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def _salad_env(monkeypatch):
    """Everything Config.from_env insists on, so these tests never depend on
    which credentials happen to be in the developer's shell."""
    for name in ("OPENCLIDE_IMAGE", "SALAD_ORGANIZATION", "SALAD_PROJECT",
                 "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")


@pytest.fixture
def salad_cfg(_salad_env):
    return salad.Config.from_env()


@pytest.fixture
def cfg(_salad_env):
    import openclude.comfy_deploy as cd
    return cd.ComfyConfig.from_env()


def _group(status: str, counts: dict, desc: str = "") -> dict:
    return {"current_state": {"status": status,
                              "instance_status_counts": counts,
                              "description": desc}}


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------


def test_it_picks_any_gpu_that_is_free_not_just_the_configured_one(watcher,
                                                                   salad_cfg,
                                                                   cfg,
                                                                   monkeypatch):
    """Three card classes are three different pools. Pinning one and refusing
    when it is busy throws away free capacity sitting right there."""
    calls: list[str] = []

    def fake_check(c):
        calls.append(c.gpu)
        busy = c.gpu == cfg.gpu
        return {"ok": not busy, "free": 0 if busy else 3}

    monkeypatch.setattr(watcher, "check_capacity", fake_check)
    got = watcher.wait_for_capacity(salad_cfg, cfg, budget=5, poll=0.01)
    assert got and got != cfg.gpu, "it stuck to the busy card"
    assert len(calls) >= 2, "it only checked the configured GPU"


def test_it_keeps_waiting_instead_of_giving_up(watcher, salad_cfg, cfg,
                                               monkeypatch):
    """`no capacity` is a condition, not a failure. The first version returned
    immediately and the entire run was wasted."""
    attempts = {"n": 0}

    def fake_check(c):
        attempts["n"] += 1
        return {"ok": attempts["n"] >= 4, "free": 2}

    monkeypatch.setattr(watcher, "check_capacity", fake_check)
    got = watcher.wait_for_capacity(salad_cfg, cfg, budget=30, poll=0.01)
    assert got != "", "it gave up instead of waiting"
    assert attempts["n"] >= 4


def test_it_gives_up_only_when_the_budget_is_spent(watcher, salad_cfg, cfg,
                                                   monkeypatch):
    monkeypatch.setattr(watcher, "check_capacity",
                        lambda c: {"ok": False, "free": 0})
    assert watcher.wait_for_capacity(salad_cfg, cfg, budget=1, poll=0.05) == ""


# --------------------------------------------------------------------------
# the URL, which is the deliverable
# --------------------------------------------------------------------------


def test_the_url_is_taken_from_the_instance(watcher, salad_cfg, cfg, monkeypatch):
    monkeypatch.setattr(watcher, "instances", lambda s, p: [
        {"id": "i1", "access_domain_name": "abc-def.salad.cloud"}])
    monkeypatch.setattr(watcher, "call", lambda *a, **k: {})
    assert watcher.find_url(salad_cfg, cfg) == "https://abc-def.salad.cloud"


def test_a_bare_host_gets_a_scheme(watcher, salad_cfg, cfg, monkeypatch):
    monkeypatch.setattr(watcher, "instances", lambda s, p: [
        {"id": "i1", "access_domain_name": "abc.salad.cloud"}])
    monkeypatch.setattr(watcher, "call", lambda *a, **k: {})
    assert watcher.find_url(salad_cfg, cfg).startswith("https://")


def test_no_url_returns_empty_rather_than_a_guess(watcher, salad_cfg, cfg,
                                                  monkeypatch):
    monkeypatch.setattr(watcher, "instances", lambda s, p: [{"id": "i1"}])

    def boom(*a, **k):
        raise salad.SaladError("404")
    monkeypatch.setattr(watcher, "call", boom)
    assert watcher.find_url(salad_cfg, cfg) == ""


# --------------------------------------------------------------------------
# failure is not the end
# --------------------------------------------------------------------------


def test_logs_come_from_the_endpoint_that_exists(watcher):
    """`/instances/<id>/logs` does not exist. It returns nothing, which looks
    exactly like a silent application, and that is how seventeen minutes were
    spent watching a crash loop through a URL that 404s."""
    import inspect
    src = inspect.getsource(watcher.logs_for)
    assert "log-entries" in src
    assert "/instances/" not in src.split('"""')[2] if src.count('"""') > 2 else True


def test_empty_logs_say_so(watcher, salad_cfg, monkeypatch):
    monkeypatch.setattr(watcher, "call", lambda *a, **k: {"logs": []})
    assert "no log entries" in watcher.logs_for(salad_cfg, "i1").lower()


def test_a_broken_log_read_says_so(watcher, salad_cfg, monkeypatch):
    def boom(*a, **k):
        raise salad.SaladError("500")
    monkeypatch.setattr(watcher, "call", boom)
    assert "unavailable" in watcher.logs_for(salad_cfg, "i1")
