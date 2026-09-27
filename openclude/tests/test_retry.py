"""Tests for the degradation ladder.

The property under test is not "it retries" but "it retries something
DIFFERENT, in a sensible order, and eventually stops".
"""

from __future__ import annotations

import pytest

from openclude.retry import (
    CANCELLED,
    DOWNLOAD,
    LadderExhausted,
    MISSING,
    OOM_RAM,
    TERMINAL,
    VRAM,
    classify,
    next_attempt,
    settings_produced_by,
    should_retry,
)
from openclude.schema import WAN22_5B


def settings(**over):
    base = {
        "model_type": "ti2v_2_2_fastwan",
        "prompt": "a pilot walks",
        "resolution": "1280x720",
        "video_length": 121,
        "num_inference_steps": 6,
        "image_refs": ["refs/nova_front.png"],
        "video_prompt_type": "I",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA out of memory", OOM_RAM),
        ("it is likely that you have unsufficient VRAM and you should therefore reduce", VRAM),
        ("CUDA error: out of memory", OOM_RAM),
        ("CUDA error: too many resources requested", OOM_RAM),
        ("This model doesn't accept an End Image", "validation"),
        ("This model does not accept an End Image", "validation"),
        ("You must provide an End Image", "validation"),
        ("No such file or directory: model.safetensors", MISSING),
        ("Read timed out", DOWNLOAD),
        ("generation aborted", CANCELLED),
        ("something nobody has seen before", "unknown"),
    ],
)
def test_failures_are_classified_from_the_wording(message, expected) -> None:
    assert classify(message) == expected


def test_terminal_failures_are_never_retried() -> None:
    for kind in TERMINAL:
        assert should_retry(kind) is False
    for kind in (VRAM, OOM_RAM, DOWNLOAD, "engine", "unknown"):
        assert should_retry(kind) is True


def test_a_terminal_failure_stops_immediately() -> None:
    with pytest.raises(LadderExhausted, match="terminal"):
        next_attempt(settings(), MISSING)


# --------------------------------------------------------------------------
# the ladder itself
# --------------------------------------------------------------------------


def test_vram_failure_shrinks_length_first() -> None:
    step = next_attempt(settings(), VRAM)
    assert step.rung == "length"
    assert step.settings["video_length"] == 61
    assert step.settings["video_length"] < 121


def test_shrunk_length_stays_on_the_legal_lattice() -> None:
    for attempts in range(1, 6):
        out = settings_produced_by(settings(), VRAM, attempts)
        frames = out["video_length"]
        assert (frames - WAN22_5B.minimum) % WAN22_5B.step == 0
        assert frames >= WAN22_5B.minimum


def test_the_ladder_terminates_instead_of_looping() -> None:
    used: set[str] = set()
    out = settings()
    for _ in range(20):
        try:
            step = next_attempt(out, VRAM, frozenset(used))
        except LadderExhausted:
            break
        out = step.settings
        used.add(step.rung)
    else:
        pytest.fail("ladder never terminated")
    assert used == {"length", "steps", "resolution", "quality"}


def test_a_rung_at_its_floor_is_skipped_not_burned() -> None:
    """A job already at the minimum must not spend a retry doing nothing."""
    out = settings(video_length=21, num_inference_steps=2)
    step = next_attempt(out, VRAM)
    assert step.rung == "resolution"
    assert step.settings["video_length"] == 21


def test_resolution_shrinks_and_stays_aligned() -> None:
    out = settings_produced_by(settings(video_length=21, num_inference_steps=2), VRAM, 3)
    w, h = (int(x) for x in out["resolution"].split("x"))
    assert w % 16 == 0 and h % 16 == 0
    assert (w, h) < (1280, 720)


def test_character_sheet_is_the_last_thing_to_go() -> None:
    out = settings()
    used: list[str] = []
    # walk until the refs are dropped, and confirm they survived everything before
    while "image_refs" in out:
        step = next_attempt(out, VRAM, frozenset(used))
        used.append(step.rung)
        out = step.settings
    # length, steps and resolution were all spent before quality
    assert used == ["length", "steps", "resolution", "quality"]


def test_no_spurious_keys_appear_during_degradation() -> None:
    """A rung that invents a key would look like progress while changing nothing.

    The quality rung is the one legitimate exception: it removes keys on
    purpose, so it is excluded from this check.
    """
    out = settings_produced_by(settings(), VRAM, 3)  # length, steps, resolution
    assert set(out) == set(settings())


def test_transient_failure_resends_the_same_job_once() -> None:
    step = next_attempt(settings(), DOWNLOAD)
    assert step.rung == "resend"
    assert step.settings == settings()
    with pytest.raises(LadderExhausted, match="already retried"):
        next_attempt(settings(), DOWNLOAD, frozenset({"resend"}))


def test_every_step_actually_changes_something() -> None:
    """A rung that returns identical settings would burn a retry for nothing."""
    used: set[str] = set()
    out = settings(video_length=21, num_inference_steps=2)
    while True:
        try:
            step = next_attempt(out, VRAM, frozenset(used))
        except LadderExhausted:
            return
        assert step.settings != out, f"rung {step.rung} changed nothing"
        used.add(step.rung)
        out = step.settings


def test_a_whole_ladder_run_ends_in_a_still_valid_job() -> None:
    out = settings_produced_by(settings(), VRAM, 4)
    assert out["prompt"] == "a pilot walks"
    assert out["model_type"] == "ti2v_2_2_fastwan"
    assert int(out["video_length"]) >= WAN22_5B.minimum


def test_unknown_failure_is_treated_as_transient() -> None:
    assert next_attempt(settings(), "unknown").rung == "resend"


def test_unparseable_resolution_is_a_clear_failure() -> None:
    with pytest.raises(LadderExhausted, match="cannot parse resolution"):
        next_attempt(
            settings(video_length=21, num_inference_steps=2, resolution="huge"),
            VRAM,
            frozenset({"length", "steps"}),
        )


def test_the_ladder_never_raises_before_being_asked_to() -> None:
    """Sanity: a fresh job must always have at least one rung available."""
    for kind in (VRAM, OOM_RAM, DOWNLOAD, "engine", "unknown"):
        assert next_attempt(settings(), kind).settings is not None
