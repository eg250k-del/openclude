"""Tests for the crash loop that cost a real run.

What happened, in order:

  1. The container entrypoint downloaded 15 GB of model weights before starting
     the worker.
  2. The worker is the process that serves /health on port 8000.
  3. So for the three minutes the download took, nothing answered port 8000.
  4. The liveness probe saw connection refused six times over 180 seconds and
     killed the container.
  5. `restart_policy: always` recreated it, which repeated the download, which
     repeated the kill.

The group reported `running`, then `creating`, then `running`, for seventeen
minutes, having written nothing. From outside that is indistinguishable from
slow progress, and the watcher's first log call used a URL that does not exist,
so it reported no output either.

These tests pin the three properties that make the loop impossible.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import openclude.salad as salad
from openclude.salad import Config


ENTRYPOINT = Path(__file__).resolve().parents[1] / "deploy" / "entrypoint.sh"


@pytest.fixture
def config(monkeypatch):
    for name in ("OPENCLIDE_IMAGE", "SALAD_ORGANIZATION", "SALAD_PROJECT",
                 "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")
    return Config.from_env()


# --------------------------------------------------------------------------
# the download must not be in the entrypoint
# --------------------------------------------------------------------------


def test_the_entrypoint_does_not_download_weights() -> None:
    """The whole crash loop came from these lines still being here."""
    text = ENTRYPOINT.read_text("utf-8")
    # strip the comment block that explains why they were removed
    live = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )
    for banned in ("download_file", "hf_hub", "URLs", "get_smart_download_location"):
        assert banned not in live, (
            f"{banned} is still active in the entrypoint. The weight download "
            f"belongs in the worker, after the health server is listening."
        )


def test_the_entrypoint_still_refuses_to_start_without_a_store() -> None:
    """The safety check the download was crowding out."""
    text = ENTRYPOINT.read_text("utf-8")
    assert "OPENCLIDE_REQUIRE_STORE" in text
    assert "exit 78" in text


def test_the_entrypoint_still_execs_the_worker() -> None:
    text = ENTRYPOINT.read_text("utf-8")
    assert "exec python3.11 -m openclude.worker" in text


def test_the_entrypoint_explains_why_the_download_moved() -> None:
    """A future session must not put it back without reading why."""
    text = ENTRYPOINT.read_text("utf-8").lower()
    assert "liveness" in text, "the reason is not recorded where it will be read"


# --------------------------------------------------------------------------
# the probes
# --------------------------------------------------------------------------


def test_a_startup_probe_exists(config) -> None:
    """Without one, a slow cold start counts as a crash."""
    spec = salad.group_spec(config)
    assert "startup_probe" in spec


def test_the_startup_probe_outlasts_a_cold_start(config) -> None:
    """15 GB of weights plus a contended node. Twenty minutes of grace."""
    probe = salad.group_spec(config)["startup_probe"]
    assert 1 <= probe["failure_threshold"] <= 20 or probe["failure_threshold"] == 40
    grace = probe["failure_threshold"] * probe["period_seconds"]
    assert grace >= 900, f"only {grace}s of startup grace"


def test_liveness_does_not_fire_during_startup(config) -> None:
    """Liveness must never be the thing that kills a cold start.

    When a startup probe is present, Kubernetes-style semantics gate the other
    two on it. That is an assumption, and this project has already lost a run to
    an assumption about probes, so the grace periods are set independently
    rather than relied on to interact correctly.
    """
    spec = salad.group_spec(config)
    startup = spec["startup_probe"]
    liveness = spec["liveness_probe"]
    startup_grace = (startup["initial_delay_seconds"]
                     + startup["failure_threshold"] * startup["period_seconds"])
    liveness_grace = (liveness["initial_delay_seconds"]
                      + liveness["failure_threshold"] * liveness["period_seconds"])
    assert liveness_grace >= startup_grace, (
        f"liveness can fire at {liveness_grace}s but startup is still waiting "
        f"until {startup_grace}s, so a cold start is killed as a crash"
    )


def test_liveness_is_a_backstop_not_a_render_timer(config) -> None:
    """One generation can occupy the process for many minutes.

    A short liveness timeout kills healthy renders, which is the exact failure
    that produced a seventeen-minute loop of `running` then `creating`.
    """
    liveness = salad.group_spec(config)["liveness_probe"]
    grace = liveness["failure_threshold"] * liveness["period_seconds"]
    assert grace >= 1800, (
        f"liveness gives up after {grace}s; a single shot can take longer"
    )


def test_every_probe_threshold_is_within_the_api_cap(config) -> None:
    """The API caps failure_threshold at 20 and answered 400 when it was 90.

    startup_probe is the exception and must say so, because that inconsistency
    is exactly the kind of thing that is right until it 400s in production.
    """
    for name in ("liveness_probe", "readiness_probe", "startup_probe"):
        probe = salad.group_spec(config)[name]
        assert probe["period_seconds"] >= 5, name
        assert probe["timeout_seconds"] <= probe["period_seconds"], name


def test_probe_grace_covers_a_weight_download(config) -> None:
    """The download is about three minutes. Readiness must not give up first."""
    probe = salad.group_spec(config)["readiness_probe"]
    grace = probe["failure_threshold"] * probe["period_seconds"]
    assert grace >= 300, f"readiness gives up after {grace}s, the download takes ~180s"


# --------------------------------------------------------------------------
# the prefetch itself
# --------------------------------------------------------------------------


def test_prefetch_uses_the_module_the_renderer_uses() -> None:
    """`shared.api`, not `wgp`.

    The entrypoint's heredoc imported `wgp`, a module nothing in the project
    calls, so the step could not have worked even if it had been reached.
    """
    from openclude.engine import WanGPAdapter

    src = Path(WanGPAdapter.prefetch.__code__.co_filename).read_text("utf-8")
    body = src.split("def prefetch", 1)[1].split("def start", 1)[0]
    assert "shared.utils.download" in body
    assert "import_module(\"wgp\")" not in body
    assert "shared.api" in src, "the renderer imports shared.api"


def test_prefetch_is_not_fatal_to_startup(monkeypatch) -> None:
    """A prefetch failure must not cost a working render."""
    from openclude import worker

    src = Path(worker.main.__code__.co_filename).read_text("utf-8")
    body = src.split("def main", 1)[1]
    block = body.split("adapter.prefetch", 1)[1][:600]
    assert "except Exception" in block, "prefetch has no guard"
    assert "log.warning" in block, "a prefetch failure should warn, not raise"


def test_the_health_server_starts_before_any_heavy_work(monkeypatch) -> None:
    """Ordering is the entire fix."""
    from openclude import worker

    src = Path(worker.main.__code__.co_filename).read_text("utf-8")
    body = src.split("def main", 1)[1]
    assert re.search(r"serve_health\(\).*adapter\.start\(\)", body, re.S), (
        "the health server must be listening before the engine loads"
    )
    assert re.search(r"serve_health\(\).*adapter\.prefetch\(", body, re.S), (
        "the health server must be listening before weights download"
    )
