"""No command may print a credential, whatever the API sends back.

The leak was not one bug but a shape. SaladCloud echoes the container's
environment variables in every response it returns, so any command that prints
a response prints the live token. That happened three times in one session:

  1. `plan` printed the spec we built, because the redaction only matched
     SECRET and KEY and the variable was called HF_TOKEN.
  2. `status` printed the group we read back, which the API had populated with
     every environment variable including the token.
  3. A 400 from `apply` printed an error body that quoted the request.

Fixing them one at a time is how the fourth happens. So this module runs every
command against a fake API whose responses are seeded with a recognisable
secret, and asserts the secret never reaches stdout.
"""

from __future__ import annotations

import json

import pytest

import openclude.salad as salad
from openclude.salad import Config, is_secret_name, redact_any, safe_json


SECRET = "LEAKCANARY_9f3a2b7c"
LEAKS = ("HF_TOKEN", "SALAD_API_KEY", "S3_SECRET_ACCESS_KEY",
         "S3_ACCESS_KEY_ID", "HF_REPO", "OPENCLIDE_STORE")


def poisoned_env() -> dict:
    """What the API gives back: every variable we sent, token included."""
    return {name: (SECRET if is_secret_name(name) else f"value-{name}")
            for name in LEAKS}


def poisoned_group() -> dict:
    return {
        "id": "abc",
        "name": "openclude",
        "autostart_policy": False,
        "replicas": 1,
        "restart_policy": "always",
        "container": {
            "image": "ghcr.io/x/y:z",
            "environment_variables": poisoned_env(),
            "resources": {"cpu": 8, "memory": 32768, "storage_amount": 1},
        },
        "current_state": {"instance_status_counts": {"running": 1}},
    }


@pytest.fixture
def config(monkeypatch):
    for name in ("OPENCLIDE_IMAGE", "SALAD_ORGANIZATION", "SALAD_PROJECT",
                 "SALAD_API_KEY"):
        monkeypatch.setenv(name, f"t-{name.lower()}")
    monkeypatch.setenv("HF_TOKEN", SECRET)
    return Config.from_env()


# --------------------------------------------------------------------------
# the redactor itself
# --------------------------------------------------------------------------


def test_a_token_key_is_classified_as_secret() -> None:
    assert is_secret_name("HF_TOKEN")


def test_the_redactor_catches_a_nested_environment() -> None:
    out = redact_any(poisoned_group())
    assert SECRET not in json.dumps(out)


def test_the_redactor_walks_lists() -> None:
    out = redact_any([{"HF_TOKEN": SECRET}, {"S3_SECRET_ACCESS_KEY": SECRET}])
    assert SECRET not in json.dumps(out)


def test_the_redactor_keeps_the_names() -> None:
    """The names are the useful part of a plan. Only the values go."""
    out = redact_any(poisoned_group())
    envs = out["container"]["environment_variables"]
    assert envs["HF_TOKEN"] == "***set***"
    assert envs["HF_REPO"] == "value-HF_REPO", "a non-secret must survive"
    assert "HF_REPO" in envs


def test_safe_json_never_emits_the_value() -> None:
    assert SECRET not in safe_json(poisoned_group())


# --------------------------------------------------------------------------
# every command, end to end
# --------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch):
    """A fake API that returns credential-bearing payloads for every verb."""
    def fake_call(cfg, method, path, body=None):
        if method == "GET" and path.endswith("/instances"):
            return {"items": [poisoned_group()]}
        return poisoned_group()
    monkeypatch.setattr(salad, "call", fake_call)
    monkeypatch.setattr(salad.time, "sleep", lambda _s: None)
    return fake_call


@pytest.mark.parametrize("command", ["plan", "preflight", "status", "apply"])
def test_no_command_prints_a_credential(config, api, capsys, command) -> None:
    args = ["--apply"] if command == "apply" else []
    try:
        getattr(salad, f"cmd_{command}")(config, args)
    except Exception:  # noqa: BLE001 - a crash is not a leak
        pass
    out = capsys.readouterr().out
    assert SECRET not in out, f"`{command}` printed the live token"
    assert "***set***" in out or command == "preflight", (
        f"`{command}` printed nothing recognisable, so the assertion above may "
        f"be passing for the wrong reason"
    )


def test_the_error_path_is_redacted_too(config, monkeypatch, capsys) -> None:
    """A 400 on create quotes the request, which carries the credentials.

    This exercises `call()` rather than `cmd_apply`, because that is where the
    error body is read. The third leak came from exactly this place: a 400 whose
    message quoted back the environment we had sent.
    """
    import io
    import urllib.error

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(json.dumps(poisoned_group()).encode()),
        )

    monkeypatch.setattr(salad.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(salad.SaladError) as exc:
        salad.call(config, "POST", "/x", {"HF_TOKEN": SECRET})
    assert SECRET not in str(exc.value)
    assert "***set***" in str(exc.value)


def test_a_non_json_error_body_is_still_printed(config, monkeypatch) -> None:
    """Redaction must not swallow a plain-text error the operator needs."""
    import io
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {},
            io.BytesIO(b"Invalid Salad-Api-Key"),
        )

    monkeypatch.setattr(salad.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(salad.SaladError) as exc:
        salad.call(config, "GET", "/x")
    assert "Invalid Salad-Api-Key" in str(exc.value)


def test_start_and_stop_are_also_safe(config, api, capsys) -> None:
    for command in ("cmd_start", "cmd_stop"):
        getattr(salad, command)(config, [])
        assert SECRET not in capsys.readouterr().out, command
