"""Tests that a secret can never be printed.

One of these exists because it already happened. `HF_TOKEN` matched neither
"SECRET" nor "KEY", so `python -m openclude.salad plan` printed the live
Hugging Face token in full. It was read off the screen, and the token had to be
rotated.

The rule these tests enforce is not "remember to redact the keys we know about".
It is: any variable whose name looks like a credential is redacted, whatever it
is called, and the check is applied to the real manifest so a new variable
cannot slip past it.
"""

from __future__ import annotations

import json

import pytest

from openclude.salad import (
    SECRET_MARKERS,
    is_secret_name,
    redact_env,
    group_spec,
    Config,
)


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "HF_TOKEN", "SALAD_API_KEY", "S3_SECRET_ACCESS_KEY", "S3_ACCESS_KEY_ID",
    "DB_PASSWORD", "AWS_SESSION_TOKEN", "AUTH_HEADER", "GITHUB_COOKIE",
    "WEBHOOK_SIGNATURE",
])
def test_credential_names_are_recognised(name: str) -> None:
    assert is_secret_name(name)


@pytest.mark.parametrize("name", [
    "OPENCLIDE_DATA", "WAN2GP_ROOT", "WAN2GP_PROFILE", "WAN2GP_ATTENTION",
    "OPENCLIDE_REQUIRE_STORE", "OPENCLIDE_STORE", "HF_REPO", "HF_REPO_ID",
    "S3_ENDPOINT", "S3_BUCKET", "HF_HUB_DISABLE_TELEMETRY",
])
def test_ordinary_names_are_not_treated_as_secrets(name: str) -> None:
    assert not is_secret_name(name)


def test_the_classifier_is_case_insensitive() -> None:
    assert is_secret_name("hf_token") and is_secret_name("Hf_Token")


def test_a_false_positive_is_harmless_and_a_false_negative_is_not() -> None:
    """That asymmetry is why the marker list is deliberately wide."""
    assert is_secret_name("SOMETHING_AUTHENTICATED")   # false positive, fine
    assert not is_secret_name("HF_REPO")                # correctly not a secret


def test_the_marker_list_covers_the_obvious_words() -> None:
    for word in ("SECRET", "TOKEN", "KEY", "PASSWORD"):
        assert word in SECRET_MARKERS


# --------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------


def test_a_token_value_is_replaced() -> None:
    out = redact_env({"HF_TOKEN": "hf_realsecretvalue"})
    assert out["HF_TOKEN"] == "***set***"
    assert "hf_realsecretvalue" not in json.dumps(out)


def test_the_name_survives_so_the_plan_is_still_useful() -> None:
    out = redact_env({"HF_TOKEN": "hf_x"})
    assert "HF_TOKEN" in out


def test_an_unset_secret_shows_as_empty_not_as_a_marker() -> None:
    out = redact_env({"HF_TOKEN": ""})
    assert out["HF_TOKEN"] == ""


def test_ordinary_values_pass_through_untouched() -> None:
    env = {"HF_REPO": "someone/openclude", "OPENCLIDE_STORE": "hf"}
    assert redact_env(env) == env


def test_no_secret_survives_a_mixed_environment() -> None:
    env = {
        "HF_TOKEN": "hf_aaa", "SALAD_API_KEY": "salad_bbb",
        "S3_SECRET_ACCESS_KEY": "s3_ccc", "HF_REPO": "someone/openclude",
    }
    blob = json.dumps(redact_env(env))
    for leaked in ("hf_aaa", "salad_bbb", "s3_ccc"):
        assert leaked not in blob
    assert "someone/openclude" in blob


# --------------------------------------------------------------------------
# the real manifest
# --------------------------------------------------------------------------


@pytest.fixture
def deployable(monkeypatch):
    """The four variables Config.from_env insists on, with harmless values."""
    for name, value in (
        ("OPENCLIDE_IMAGE", "ghcr.io/eg250k-del/openclude:0" * 0 + "test"),
        ("SALAD_ORGANIZATION", "mostafa-ai"),
        ("SALAD_PROJECT", "aoutooanimation"),
    ):
        monkeypatch.setenv(name, value)
    return Config.from_env()


def test_the_actual_group_spec_leaks_nothing(monkeypatch, deployable) -> None:
    """The end-to-end version: build the real spec with fake secrets in it."""
    for name, value in (
        ("HF_TOKEN", "hf_THISWOULDLEAK"),
        ("SALAD_API_KEY", "salad_THISWOULDLEAK"),
        ("S3_SECRET_ACCESS_KEY", "s3_THISWOULDLEAK"),
        ("S3_ACCESS_KEY_ID", "s3id_THISWOULDLEAK"),
    ):
        monkeypatch.setenv(name, value)

    envs = group_spec(deployable).get("container", {}).get(
        "environment_variables", {}
    )
    redacted = redact_env(envs)
    blob = json.dumps(redacted)
    for leaked in ("hf_THISWOULDLEAK", "salad_THISWOULDLEAK",
                   "s3_THISWOULDLEAK", "s3id_THISWOULDLEAK"):
        assert leaked not in blob, f"{leaked} survived redaction"
    # and the useful information is still there
    assert "HF_REPO" in redacted


def test_plan_output_contains_no_secret(monkeypatch, capsys, deployable) -> None:
    """Run the command and read what it actually printed."""
    from openclude.salad import cmd_plan

    monkeypatch.setenv("HF_TOKEN", "hf_PLANLEAK")
    monkeypatch.setenv("SALAD_API_KEY", "salad_PLANLEAK")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "s3_PLANLEAK")

    cmd_plan(deployable, [])
    out = capsys.readouterr().out
    for leaked in ("hf_PLANLEAK", "salad_PLANLEAK", "s3_PLANLEAK"):
        assert leaked not in out, f"plan printed {leaked}"
    assert "***set***" in out


def test_a_future_variable_cannot_slip_past() -> None:
    """If someone adds a credential, the classifier must catch it by default."""
    for invented in ("HF_XET_TOKEN", "WAN2GP_API_KEY", "OPENCLIDE_AUTH",
                     "WEBUI_SESSION", "R2_ACCESS_KEY_ID"):
        assert is_secret_name(invented), invented
