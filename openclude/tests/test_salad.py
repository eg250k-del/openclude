"""Tests for the SaladCloud control script.

The dangerous property here is not "does it build a correct request" but
"can it ever spend money without being told to". These tests are mostly about
that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from openclude import salad
from openclude.salad import Config, SaladError, group_spec, main

ROOT = Path(__file__).resolve().parents[1]


def env(**over) -> dict[str, str]:
    base = {
        "SALAD_API_KEY": "test-key-not-real",
        "SALAD_ORGANIZATION": "acme",
        "SALAD_PROJECT": "studio",
        "OPENCLIDE_IMAGE": "ghcr.io/eg250k-del/openclude:latest",
    }
    base.update(over)
    return base


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("SALAD_", "OPENCLIDE_", "S3_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "environ", env())
    yield


# --------------------------------------------------------------------------
# safety: nothing billable without an explicit flag
# --------------------------------------------------------------------------


def test_apply_without_the_flag_refuses(clean_env) -> None:
    assert main(["apply"]) == 2


def test_the_group_name_alone_never_applies(clean_env) -> None:
    """Even a well-formed invocation must not create anything by accident."""
    for argv in (["apply"], ["apply", "--yes"], ["APPLY"], ["apply", "-f"]):
        assert main(argv) != 0, argv


def test_preflight_and_plan_need_no_apply_flag(clean_env) -> None:
    assert main(["plan"]) == 0


# --------------------------------------------------------------------------
# credentials: never printed, never guessed
# --------------------------------------------------------------------------


def test_a_missing_key_is_a_clear_error_not_a_guess(clean_env) -> None:
    os.environ.pop("SALAD_API_KEY", None)
    with pytest.raises(SaladError, match="SALAD_API_KEY"):
        Config.from_env()


def test_a_missing_organization_is_never_inferred(clean_env) -> None:
    os.environ.pop("SALAD_ORGANIZATION", None)
    with pytest.raises(SaladError, match="SALAD_ORGANIZATION"):
        Config.from_env()


def test_every_missing_name_is_reported_at_once(clean_env) -> None:
    for k in ("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "OPENCLIDE_IMAGE"):
        os.environ.pop(k, None)
    with pytest.raises(SaladError) as exc:
        Config.from_env()
    for name in ("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "OPENCLIDE_IMAGE"):
        assert name in str(exc.value)


def test_the_error_tells_the_user_where_the_key_comes_from(clean_env) -> None:
    os.environ.pop("SALAD_API_KEY", None)
    with pytest.raises(SaladError, match="environment variable"):
        Config.from_env()


def test_the_key_is_never_in_the_repr(clean_env) -> None:
    """`api_key` is declared repr=False precisely so a traceback cannot leak it."""
    cfg = Config.from_env()
    assert "test-key-not-real" not in repr(cfg)
    assert "api_key" not in repr(cfg)


def test_the_key_is_never_in_the_group_spec(clean_env) -> None:
    spec = group_spec(Config.from_env())
    blob = json.dumps(spec)
    assert "test-key-not-real" not in blob


def test_no_secret_is_printed_by_plan(clean_env, capsys) -> None:
    os.environ["S3_SECRET_ACCESS_KEY"] = "super-secret-value"
    assert main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "***set***" in out


# --------------------------------------------------------------------------
# the spec matches the official OpenAPI shape
#
# These assertions exist because the first draft of group_spec() guessed the
# shape and every part of it was wrong: image at the top level instead of
# inside container, resources.storage instead of storage_amount, a `probes`
# dict instead of liveness_probe/readiness_probe, and no autostart_policy at
# all. SaladCloud rejects a body that does not match ContainerGroupPrototype.
# --------------------------------------------------------------------------

SPEC = "https://raw.githubusercontent.com/SaladTechnologies/salad-cloud-docs/main/api-specs/salad-cloud.yaml"


def container_of(cfg=None) -> dict:
    return group_spec(cfg or Config.from_env())["container"]


def test_the_spec_targets_amd64(clean_env) -> None:
    assert container_of()["image"]["architecture"] == "amd64"


def test_the_spec_uses_the_image_repository(clean_env) -> None:
    assert container_of()["image"]["repository"] == os.environ["OPENCLIDE_IMAGE"]


def test_the_spec_asks_for_profile_3_for_24gb(clean_env) -> None:
    """Profile 3 is the engine's own label for 24 GB VRAM; 5 is a 10 GB choice."""
    env = container_of()["environment_variables"]
    assert env["WAN2GP_PROFILE"] == "3"


def test_the_spec_uses_sage_attention(clean_env) -> None:
    assert container_of()["environment_variables"]["WAN2GP_ATTENTION"] == "sage2"


def test_the_spec_asks_for_a_large_disk(clean_env) -> None:
    """The node must have room for the weights plus a 720p working set."""
    assert container_of()["resources"]["storage_amount"] >= 100_000


def test_the_spec_does_not_autostart(clean_env) -> None:
    """A group that autostarts bills all night. The autoscaler decides."""
    assert group_spec(Config.from_env())["autostart_policy"] is False


def test_the_spec_declares_a_queue_autoscaler(clean_env) -> None:
    autoscaler = group_spec(Config.from_env())["queue_autoscaler"]
    assert autoscaler["min_replicas"] == 0
    assert autoscaler["max_replicas"] >= 1
    assert autoscaler["desired_queue_length"] >= 1


def test_the_spec_satisfies_every_required_field(clean_env) -> None:
    """The five fields ContainerGroupPrototype marks required."""
    spec = group_spec(Config.from_env())
    for field in ("autostart_policy", "container", "name", "replicas",
                  "restart_policy"):
        assert field in spec, field


def test_the_nested_containers_satisfy_their_required_fields(clean_env) -> None:
    spec = group_spec(Config.from_env())
    for field in ("image", "resources"):
        assert field in spec["container"], field
    for field in ("cpu", "memory"):
        assert field in spec["container"]["resources"], field
    for field in ("auth", "port", "protocol"):
        assert field in spec["networking"], field


def test_every_probe_carries_all_five_required_fields(clean_env) -> None:
    spec = group_spec(Config.from_env())
    for name in ("liveness_probe", "readiness_probe"):
        probe = spec[name]
        for field in ("failure_threshold", "initial_delay_seconds",
                      "period_seconds", "success_threshold", "timeout_seconds"):
            assert field in probe, f"{name}.{field}"
        for field in ("headers", "path", "port", "scheme"):
            assert field in probe["http"], f"{name}.http.{field}"


def test_the_startup_probe_allows_a_long_model_download(clean_env) -> None:
    """A short probe window kills the instance mid-download and wastes the run."""
    probe = group_spec(Config.from_env())["readiness_probe"]
    grace = probe["failure_threshold"] * probe["period_seconds"]
    assert grace >= 300


def test_the_spec_requires_object_storage(clean_env) -> None:
    """The container must refuse to start without a durable store."""
    env = container_of()["environment_variables"]
    assert env["OPENCLIDE_REQUIRE_STORE"] == "1"


def test_the_spec_restarts_on_failure(clean_env) -> None:
    assert group_spec(Config.from_env())["restart_policy"]["condition"] == "always"


def test_the_spec_exposes_one_http_port_without_auth(clean_env) -> None:
    net = group_spec(Config.from_env())["networking"]
    assert net["port"] == 8000
    assert net["protocol"] == "http"
    assert net["auth"] is False


def test_the_base_url_is_the_one_in_the_spec(clean_env) -> None:
    """`/api/v1` 404s for every documented path. The spec says /api/public."""
    assert salad.API.endswith("/api/public")


def test_the_auth_header_is_the_one_in_the_spec(clean_env) -> None:
    assert salad.AUTH_HEADER == "Salad-Api-Key"


def test_the_user_agent_is_set(clean_env) -> None:
    """Cloudflare rejects the default urllib agent with error 1010."""
    assert "Python-urllib" not in salad.USER_AGENT
    assert salad.USER_AGENT


def test_the_quota_path_comes_from_the_spec(clean_env) -> None:
    import inspect

    src = inspect.getsource(salad.cmd_preflight)
    assert "/organizations/{cfg.organization}/quotas" in src
    assert "/node-pools" not in src
