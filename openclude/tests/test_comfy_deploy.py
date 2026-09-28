"""Tests for the ComfyUI deployment.

The deployment half of this project has been wrong nine times against the live
API, and every time the cause was a shape or a limit that a dry run cannot see.
So this module pins the shapes, and it pins the limits, and the reasons are in
the code where the next person will read them.
"""

from __future__ import annotations

import re

import pytest

import openclude.comfy_deploy as cd
from openclude.salad import GPU_CLASSES, SaladError


@pytest.fixture
def cfg(monkeypatch):
    for name in ("SALAD_ORGANIZATION", "SALAD_PROJECT", "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")
    return cd.ComfyConfig.from_env()


@pytest.fixture
def spec(cfg) -> dict:
    return cd.group_spec(cfg)


# --------------------------------------------------------------------------
# the shapes the API rejects
# --------------------------------------------------------------------------


def test_image_is_a_plain_string(spec) -> None:
    """An object with repository/architecture is a 400 that says nothing useful."""
    assert isinstance(spec["container"]["image"], str)


def test_restart_policy_is_a_string(spec) -> None:
    assert spec["restart_policy"] == "always"


def test_memory_is_in_megabytes(spec) -> None:
    """It was 32 once, which reads as 32 MB of RAM.

    Nothing rejects a plausible number in the wrong unit, so a dry run cannot
    catch this and only a container that dies on boot will.
    """
    assert spec["container"]["resources"]["memory"] >= 8192


def test_storage_is_in_bytes(spec) -> None:
    """It was 120000 once, which reads as 117 KB of disk."""
    assert spec["container"]["resources"]["storage_amount"] >= 10 * 1024**3


def test_gpu_classes_are_uuids(spec) -> None:
    values = spec["container"]["resources"]["gpu_classes"]
    assert values
    for v in values:
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", v), v


def test_an_unknown_gpu_is_refused_before_the_api_is_called() -> None:
    """A name typo must not look like an API outage."""
    with pytest.raises(SaladError, match="unknown GPU class"):
        cd.gpu_class_ids("RTX 9999")


def test_probe_headers_are_an_array(spec) -> None:
    for name in ("startup_probe", "readiness_probe", "liveness_probe"):
        assert isinstance(spec[name]["http"]["headers"], list), name


def test_every_probe_threshold_is_within_the_cap(spec) -> None:
    """Capped at 20. Hit three times, and the grace period comes from the
    period_seconds for exactly this reason."""
    for name, probe in spec.items():
        if name.endswith("_probe"):
            assert 1 <= probe["failure_threshold"] <= 20, name
            assert 1 <= probe["success_threshold"] <= 20, name


def test_no_salad_job_queue_is_named(spec) -> None:
    """There is no job queue in this project, and naming one that does not
    exist fails the create."""
    assert "queue_autoscaler" not in spec
    assert "queue_connection" not in spec
    assert "queue_name" not in str(spec)


def test_liveness_outlasts_startup(spec) -> None:
    """A cold start downloads 30 GB. Liveness must not be what kills it."""
    s = spec["startup_probe"]
    l = spec["liveness_probe"]
    s_grace = s["initial_delay_seconds"] + s["failure_threshold"] * s["period_seconds"]
    l_grace = l["initial_delay_seconds"] + l["failure_threshold"] * l["period_seconds"]
    assert l_grace >= s_grace, f"liveness fires at {l_grace}s, startup waits to {s_grace}s"


def test_startup_grace_covers_a_model_download(spec) -> None:
    s = spec["startup_probe"]
    grace = s["failure_threshold"] * s["period_seconds"]
    assert grace >= 900, f"only {grace}s for about 30 GB on a home connection"


# --------------------------------------------------------------------------
# the two things that make this project work for the user
# --------------------------------------------------------------------------


def test_the_web_ui_port_is_exposed(spec) -> None:
    """Without this the user sees nothing, and the project is invisible."""
    assert spec["networking"]["port"] == cd.UI_PORT == 8188


def test_the_gateway_is_not_authenticated(spec) -> None:
    """The user opens a URL in their own browser. A password prompt is not a
    URL they can paste, and this is a private, paid resource rather than a
    public service."""
    assert spec["networking"]["auth"] is False


def test_the_hf_token_reaches_the_container(spec) -> None:
    """Gated models and the results upload both need it."""
    env = spec["container"]["environment_variables"]
    assert "HF_TOKEN" in env
    assert "HF_REPO" in env


def test_the_container_can_write_results(spec) -> None:
    env = spec["container"]["environment_variables"]
    assert env["COMFY_OUTPUT_DIR"].startswith("/data")


def test_the_cache_is_bounded(spec) -> None:
    """An unbounded model cache fills the node's disk and evicts a model in the
    middle of a render, which fails as an out-of-space error with no clue."""
    env = spec["container"]["environment_variables"]
    assert int(env["LRU_CACHE_SIZE_GB"]) > 0


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_the_group_does_not_start_itself(spec) -> None:
    """This bills by the hour and the user is not always at the keyboard."""
    assert spec["autostart_policy"] is False


def test_one_replica_to_start_with(spec) -> None:
    """ComfyUI runs one workflow at a time. More replicas means more money for
    the same single job."""
    assert spec["replicas"] == 1


# --------------------------------------------------------------------------
# no secrets in the spec that gets printed
# --------------------------------------------------------------------------


def test_the_spec_is_printable_without_leaking(spec) -> None:
    """SaladCloud echoes the container's environment back in every response, so
    any command that prints a response can print the live token."""
    from openclude.salad import safe_json

    blob = safe_json(spec)
    assert "***set***" in blob
    assert "hf_" not in blob or "hf_TOKEN" in blob
