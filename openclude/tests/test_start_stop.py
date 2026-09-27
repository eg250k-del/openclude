"""Tests for start and stop.

These are the two commands that move money. `apply` creates a group but leaves
it stopped, because `autostart_policy` is false, so nothing bills until someone
runs `start`. And `stop` is the answer to "it is still running and I am being
charged".

They are also the two commands most likely to be pressed twice, and a second
`stop` on an already-stopped group must be a success, not an error.
"""

from __future__ import annotations

import json

import pytest

from openclude.salad import GROUP_NAME, Config, cmd_start, cmd_stop
import openclude.salad as salad


@pytest.fixture
def config(monkeypatch):
    for name in ("OPENCLIDE_IMAGE", "SALAD_ORGANIZATION", "SALAD_PROJECT",
                 "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")
    return Config.from_env()


@pytest.fixture
def clean_env(config):
    return config


def _capture(monkeypatch, result=None, error: str | None = None) -> list:
    """Record every API call instead of making one."""
    calls: list[tuple] = []

    def fake_call(cfg, method, path, body=None):
        calls.append((method, path, body))
        if error:
            raise salad.SaladError(error)
        return result if result is not None else {"id": "abc123"}

    monkeypatch.setattr(salad, "call", fake_call)
    return calls


# --------------------------------------------------------------------------
# start
# --------------------------------------------------------------------------


def test_start_posts_to_the_group_start_endpoint(config, monkeypatch, capsys) -> None:
    calls = _capture(monkeypatch)
    assert cmd_start(config, []) == 0
    method, path, _ = calls[0]
    assert method == "POST"
    assert path.endswith(f"/containers/{GROUP_NAME}/start")
    assert "accepted" in capsys.readouterr().out


def test_start_uses_the_configured_org_and_project(config, monkeypatch) -> None:
    calls = _capture(monkeypatch)
    cmd_start(config, [])
    path = calls[0][1]
    assert config.organization in path
    assert config.project in path


def test_start_failure_is_reported_not_swallowed(config, monkeypatch, capsys) -> None:
    _capture(monkeypatch, error="401 Unauthorized")
    assert cmd_start(config, []) == 1
    assert "refused" in capsys.readouterr().out


def test_start_tells_the_operator_it_is_not_instant(config, monkeypatch, capsys) -> None:
    """Instances appear asynchronously and the first one downloads weights."""
    _capture(monkeypatch)
    cmd_start(config, [])
    out = capsys.readouterr().out
    assert "asynchronously" in out
    assert "weights" in out


# --------------------------------------------------------------------------
# stop
# --------------------------------------------------------------------------


def test_stop_posts_to_the_group_stop_endpoint(config, monkeypatch, capsys) -> None:
    calls = _capture(monkeypatch)
    assert cmd_stop(config, []) == 0
    assert calls[0][1].endswith(f"/containers/{GROUP_NAME}/stop")
    assert "accepted" in capsys.readouterr().out


def test_stopping_something_already_stopped_is_a_success(config, monkeypatch, capsys) -> None:
    """Pressing stop twice is the natural reaction, and must not be an error."""
    _capture(monkeypatch, error="404 Not Found")
    assert cmd_stop(config, []) == 0
    assert "nothing to stop" in capsys.readouterr().out


def test_a_real_stop_failure_is_still_an_error(config, monkeypatch, capsys) -> None:
    """A 404 is benign; a 403 is not, and must not be hidden by the same branch."""
    _capture(monkeypatch, error="403 Forbidden")
    assert cmd_stop(config, []) == 1
    assert "refused" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the cost-relevant property
# --------------------------------------------------------------------------


def test_apply_leaves_the_group_stopped_so_nothing_bills_by_itself(config) -> None:
    """The whole point of autostart_policy false.

    If this ever flips, a group left alone starts rendering, and for someone
    paying by the hour that is an unbounded bill discovered the next morning.
    """
    spec = salad.group_spec(config)
    assert spec["autostart_policy"] is False


def test_a_patch_declares_itself_a_merge_patch(config, monkeypatch) -> None:
    """application/json to PATCH is answered with 415.

    The merge-patch content type is the only way to update an existing group,
    and the failure is a 415 that says nothing about the cause.
    """
    import urllib.request

    seen: dict = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["content_type"] = req.get_header("Content-type")
        return Resp()

    monkeypatch.setattr(salad.urllib.request, "urlopen", fake_urlopen)
    salad.call(config, "PATCH", "/x", {"replicas": 1})
    assert seen["content_type"] == "application/merge-patch+json"

    salad.call(config, "POST", "/x", {"replicas": 1})
    assert seen["content_type"] == "application/json", "a create is plain JSON"


def test_every_path_goes_through_one_helper(config) -> None:
    """The path was spelled out in four places and one of them 404'd.

    `/projects/<project>/container-groups` is missing the organization segment
    and names the collection wrongly. It is exactly the kind of divergence a
    single function prevents.
    """
    path = salad.containers_path(config)
    assert path == (f"/organizations/{config.organization}"
                    f"/projects/{config.project}/containers")
    assert salad.containers_path_for(config, "g") == path + "/g"
    for command in ("cmd_status", "cmd_start", "cmd_stop", "cmd_apply"):
        source = getattr(salad, command).__code__
        literals = [c for c in source.co_consts if isinstance(c, str)]
        assert not any("container-groups" in c for c in literals), command
        assert not any(c.startswith("/projects/") for c in literals), command


def test_no_salad_job_queue_is_referenced(clean_env) -> None:
    """The queue is a folder in the object store, not a SaladCloud feature.

    Leaving queue_autoscaler and queue_connection in the spec meant every
    create attempt failed, because the API validates that the named queue
    exists and none does. A reference to something that does not exist is not
    a harmless extra field.
    """
    spec = salad.group_spec(Config.from_env())
    assert "queue_autoscaler" not in spec
    assert "queue_connection" not in spec
    # Not QUEUE_NAME: that happens to equal GROUP_NAME, so the name legitimately
    # appears. What must not appear is anything that binds the group to a queue.
    assert "queue_name" not in json.dumps(spec)


def test_probe_grace_survives_the_threshold_cap(clean_env) -> None:
    """failure_threshold is capped at 20, so the window comes from the period.

    The first version asked for 90 and the API answered "must be between 1
    and 20". Reading the cap as a style preference rather than a limit is how
    a deploy fails three times in a row.
    """
    probe = salad.group_spec(Config.from_env())["readiness_probe"]
    assert 1 <= probe["failure_threshold"] <= 20
    grace = probe["failure_threshold"] * probe["period_seconds"]
    assert grace >= 300, f"only {grace}s of grace for a model download"
