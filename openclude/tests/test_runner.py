"""End-to-end tests for the render loop, with a fake GPU.

The fake engine is deliberately hostile: it runs out of memory on big jobs,
loses the network, and dies mid-shot.  The film has to finish anyway.
"""

from __future__ import annotations

import json

import pytest

from openclude.retry import VRAM
from openclude.runner import run_film
from openclude.schema import Character, Film, Scene, Shot
from openclude.state import Ledger, ShotStatus


def build_film(n: int = 4) -> Film:
    nova = Character(id="nova", name="Nova", description="a pilot")
    shots = tuple(
        Shot(
            id=f"s01_sh{i:03d}", scene_id="s01", index=i,
            narration=f"Line {i}", visual=f"Shot {i}", character_ids=("nova",),
        ).measured(2.0)
        for i in range(1, n + 1)
    )
    return Film(
        id="f01", title="t", language="ar", target_minutes=1,
        characters=(nova,),
        scenes=(
            Scene(
                id="s01", index=1, summary="x", location="y",
                time_of_day="dawn", shots=shots,
            ),
        ),
    )


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_a_clean_film_completes() -> None:
    film = build_film()
    led = Ledger.in_memory(film)
    seen: list[dict] = []

    def fake(settings: dict) -> str:
        seen.append(settings)
        return f"out/{settings['seed']}.mp4"

    report = run_film(film, led, fake)
    assert report.succeeded == 4
    assert report.failed == 0
    assert report.degraded == 0
    assert led.done_count() == 4
    assert len(seen) == 4


def test_one_render_per_shot_when_nothing_goes_wrong() -> None:
    film = build_film(3)
    led = Ledger.in_memory(film)
    calls: list[int] = []

    def fake(settings: dict) -> str:
        calls.append(settings["seed"])
        return "out/a.mp4"

    run_film(film, led, fake)
    assert len(calls) == len(set(calls)) == 3


# --------------------------------------------------------------------------
# failure isolation — the brief's hardest requirement
# --------------------------------------------------------------------------


def test_one_bad_shot_does_not_kill_the_film() -> None:
    film = build_film(5)
    led = Ledger.in_memory(film, max_attempts=1)

    def fake(settings: dict) -> str:
        if settings["prompt"].find("Shot 3") >= 0:
            raise RuntimeError("This model doesn't accept an End Image")
        return "out/a.mp4"

    report = run_film(film, led, fake)
    assert report.succeeded == 4
    assert report.failed == 1
    assert [sid for sid, _ in report.failures] == ["s01_sh003"]
    assert led.done_count() == 4
    assert led.status("s01_sh003") is ShotStatus.FAILED


def test_a_shot_that_fails_validation_is_not_retried_forever() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film, max_attempts=5)
    calls = {"n": 0}

    def fake(settings: dict) -> str:
        calls["n"] += 1
        raise RuntimeError("This model doesn't accept an End Image")

    run_film(film, led, fake)
    assert calls["n"] == 1  # terminal failure, one attempt, done


def test_vram_failure_degrades_and_then_succeeds() -> None:
    film = build_film(1)          # 2.0 s of audio -> 49 frames
    led = Ledger.in_memory(film, max_attempts=4)
    attempts: list[int] = []

    def fake(settings: dict) -> str:
        attempts.append(settings["video_length"])
        if settings["video_length"] > 25:
            raise RuntimeError("you have unsufficient VRAM, reduce frames")
        return "out/a.mp4"

    report = run_film(film, led, fake)
    assert report.succeeded == 1
    assert report.degraded == 1
    assert attempts == [49, 25]   # halved, still on the 21+4n lattice
    assert led.status("s01_sh001") is ShotStatus.DONE


def test_degradation_is_recorded_in_the_ledger() -> None:
    """When a shot finally succeeds, the log must show it was shrunk to get there."""
    film = build_film(1)
    led = Ledger.in_memory(film, max_attempts=4)

    def fake(settings: dict) -> str:
        if settings["video_length"] > 25:
            raise RuntimeError("you have unsufficient VRAM")
        return "out/a.mp4"

    run_film(film, led, fake)
    attempts = led.record("s01_sh001").attempts
    assert [a.ok for a in attempts] == [False, True]
    assert attempts[0].kind == VRAM
    assert attempts[1].ok


def test_a_shot_that_never_fits_is_given_up_on_not_looped() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film, max_attempts=3)
    calls = {"n": 0}

    def fake(settings: dict) -> str:
        calls["n"] += 1
        raise RuntimeError("CUDA error: out of memory")

    run_film(film, led, fake)
    rec = led.record("s01_sh001")
    assert rec.exhausted(3)
    assert rec.status is ShotStatus.FAILED
    assert calls["n"] <= 3  # bounded, not infinite


# --------------------------------------------------------------------------
# crash and resume
# --------------------------------------------------------------------------


def test_the_loop_resumes_after_a_process_death(tmp_path) -> None:
    """Two 'processes', the first dies at shot 3.  The second finishes the film."""
    film = build_film(6)
    path = tmp_path / "state.json"

    class Died(BaseException):
        """BaseException so nothing swallows it, like a real SIGKILL."""

    def dying(settings: dict) -> str:
        if settings["prompt"].find("Shot 3") >= 0:
            raise Died("container preempted")
        return f"out/{settings['seed']}.mp4"

    first = Ledger.open(path, film, max_attempts=4)
    with pytest.raises(Died):
        run_film(film, first, dying)

    # shots 1 and 2 are banked, shot 3 is back to pending
    reopened = Ledger.open(path, film, max_attempts=4)
    assert reopened.done_count() == 2
    assert reopened.status("s01_sh003") is ShotStatus.PENDING
    assert [s.id for s in reopened.workable(film)] == [
        "s01_sh003", "s01_sh004", "s01_sh005", "s01_sh006"
    ]

    def healthy(settings: dict) -> str:
        return f"out/{settings['seed']}.mp4"

    second = Ledger.open(path, film, max_attempts=4)
    report = run_film(film, second, healthy)
    assert report.succeeded == 4
    assert second.done_count() == 6
    assert second.progress(film) == 1.0


def test_rerunning_a_finished_film_renders_nothing() -> None:
    film = build_film(3)
    led = Ledger.in_memory(film)
    run_film(film, led, lambda s: "out/a.mp4")

    calls = {"n": 0}

    def counting(settings: dict) -> str:
        calls["n"] += 1
        return "out/a.mp4"

    report = run_film(film, led, counting)
    assert calls["n"] == 0
    assert report.succeeded == 0
    assert report.skipped_done == 3


def test_resumed_shots_render_identical_frames(tmp_path) -> None:
    """A shot resumed tomorrow must produce the same pixels as today."""
    film = build_film(3)
    path = tmp_path / "state.json"
    first_batch: list[dict] = []

    def dying(settings: dict) -> str:
        first_batch.append(dict(settings))
        if settings["prompt"].find("Shot 2") >= 0:
            raise KeyboardInterrupt("stop")
        return "out/a.mp4"

    with pytest.raises(KeyboardInterrupt):
        run_film(film, Ledger.open(path, film), dying)

    second_batch: list[dict] = []
    run_film(
        film,
        Ledger.open(path, film),
        lambda s: (second_batch.append(dict(s)), "out/a.mp4")[1],
    )
    # the shot that was interrupted appears in both, with identical settings
    interrupted = [s for s in first_batch if s["prompt"].find("Shot 2") >= 0]
    resumed = [s for s in second_batch if s["prompt"].find("Shot 2") >= 0]
    assert interrupted and resumed
    assert interrupted[0]["seed"] == resumed[0]["seed"]
    assert interrupted[0]["video_length"] == resumed[0]["video_length"]


# --------------------------------------------------------------------------
# observability
# --------------------------------------------------------------------------


def test_the_run_ends_with_a_line_a_human_can_act_on(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="openclude.runner")
    film = build_film(3)
    led = Ledger.in_memory(film)
    run_film(film, led, lambda s: "out/a.mp4")
    text = caplog.text
    assert "shot=s01_sh001 ok" in text
    assert "film=f01 shots=3 done=3" in led.summary(film)


def test_the_report_counts_successes_and_failures() -> None:
    film = build_film(3)
    led = Ledger.in_memory(film, max_attempts=1)

    def fake(settings: dict) -> str:
        if settings["prompt"].find("Shot 2") >= 0:
            raise RuntimeError("unsupported setting")
        return "out/a.mp4"

    report = run_film(film, led, fake)
    assert "attempted=3 ok=2 failed=1" in report.line()


def test_state_survives_as_plain_json(tmp_path) -> None:
    """A human must be able to read the file when something goes wrong."""
    film = build_film(2)
    path = tmp_path / "state.json"
    led = Ledger.open(path, film)
    run_film(film, led, lambda s: "out/a.mp4")
    data = json.loads(path.read_text("utf-8"))
    assert data["version"] == 1
    assert set(data["records"]) == {"s01_sh001", "s01_sh002"}
    assert all(r["status"] == "done" for r in data["records"].values())
    assert all(r["seed"] > 0 for r in data["records"].values())
