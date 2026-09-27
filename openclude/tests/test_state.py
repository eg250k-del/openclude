"""Crash and resume tests for the ledger.

The point of this module is that a container can die at any moment.  These
tests kill it on purpose.
"""

from __future__ import annotations

import json

import pytest

from openclude.schema import Character, Film, Scene, Shot
from openclude.state import (
    DEFAULT_MAX_ATTEMPTS,
    Ledger,
    LedgerError,
    ShotStatus,
    fingerprint,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def build_film(n: int = 5) -> Film:
    nova = Character(id="nova", name="Nova", description="a pilot")
    shots = tuple(
        Shot(
            id=f"s01_sh{i:03d}",
            scene_id="s01",
            index=i,
            narration=f"Line {i}",
            visual=f"Shot {i}",
        ).measured(2.0)
        for i in range(1, n + 1)
    )
    return Film(
        id="f01",
        title="t",
        language="ar",
        target_minutes=1,
        characters=(nova,),
        scenes=(
            Scene(
                id="s01", index=1, summary="x", location="y",
                time_of_day="dawn", shots=shots,
            ),
        ),
    )


def settings_for(shot: Shot) -> dict:
    return shot.to_engine_settings([Character(id="nova", name="Nova", description="a pilot")])


# --------------------------------------------------------------------------
# crash recovery — the whole point
# --------------------------------------------------------------------------


def test_a_shot_left_running_comes_back_as_pending(tmp_path) -> None:
    """Container died mid-render.  The shot must be retried, not lost."""
    film = build_film()
    path = tmp_path / "state.json"

    led = Ledger.open(path, film)
    led.begin(film.shots[0], settings_for(film.shots[0]))
    assert led.status("s01_sh001") is ShotStatus.RUNNING
    # no succeed() — simulate the process being killed here

    reopened = Ledger.open(path, film)
    assert reopened.status("s01_sh001") is ShotStatus.PENDING
    assert reopened.workable(film)[0].id == "s01_sh001"


def test_completed_work_survives_a_restart(tmp_path) -> None:
    film = build_film()
    path = tmp_path / "state.json"

    led = Ledger.open(path, film)
    for shot in film.shots[:3]:
        led.begin(shot, settings_for(shot))
        led.succeed(shot.id, f"out/{shot.id}.mp4", seconds=90.0, frames=49)

    reopened = Ledger.open(path, film)
    assert reopened.done_count() == 3
    assert [s.id for s in reopened.workable(film)] == ["s01_sh004", "s01_sh005"]
    assert reopened.progress(film) == pytest.approx(0.6)


def test_reopening_twice_changes_nothing(tmp_path) -> None:
    film = build_film()
    path = tmp_path / "state.json"
    a = Ledger.open(path, film)
    a.begin(film.shots[0], settings_for(film.shots[0]))
    a.succeed("s01_sh001", "out/a.mp4", seconds=10)
    b = Ledger.open(path, film)
    c = Ledger.open(path, film)
    assert b.records == c.records


# --------------------------------------------------------------------------
# seed pinning — reproducibility across a restart
# --------------------------------------------------------------------------


def test_seed_is_pinned_and_survives_restart(tmp_path) -> None:
    film = build_film()
    path = tmp_path / "state.json"

    led = Ledger.open(path, film)
    claimed = led.begin(film.shots[0], settings_for(film.shots[0]))
    pinned = claimed.seed
    assert pinned == film.shots[0].resolved_seed()

    reopened = Ledger.open(path, film)
    again = reopened.begin(film.shots[0], settings_for(film.shots[0]))
    assert again.seed == pinned


def test_pinned_seed_differs_per_shot(tmp_path) -> None:
    film = build_film()
    led = Ledger.in_memory(film)
    seeds = {
        led.begin(shot, settings_for(shot)).seed for shot in film.shots
    }
    assert len(seeds) == len(film.shots)


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_a_done_shot_is_never_rendered_again(tmp_path) -> None:
    film = build_film()
    led = Ledger.in_memory(film)
    shot = film.shots[0]
    led.begin(shot, settings_for(shot))
    led.succeed(shot.id, "out/a.mp4", seconds=5)

    with pytest.raises(LedgerError, match="already done"):
        led.begin(shot, settings_for(shot))


def test_rerunning_a_finished_film_does_nothing(tmp_path) -> None:
    film = build_film()
    path = tmp_path / "state.json"
    led = Ledger.open(path, film)
    for shot in film.shots:
        led.begin(shot, settings_for(shot))
        led.succeed(shot.id, f"out/{shot.id}.mp4", seconds=1)

    again = Ledger.open(path, film)
    assert again.workable(film) == ()


# --------------------------------------------------------------------------
# bounded retries
# --------------------------------------------------------------------------


def test_retries_are_bounded_then_the_shot_is_given_up_on() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film, max_attempts=3)
    shot = film.shots[0]

    for _ in range(3):
        if not led.workable(film):
            break
        led.begin(shot, settings_for(shot))
        led.fail(shot.id, "CUDA out of memory", kind="vram")

    rec = led.record(shot.id)
    assert rec.attempt_count == 3
    assert rec.exhausted(3)
    assert led.workable(film) == ()
    assert [s.id for s in led.blocked(film)] == [shot.id]


def test_attempt_history_is_append_only() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film)
    shot = film.shots[0]
    led.begin(shot, settings_for(shot))
    led.fail(shot.id, "boom", kind="engine")
    led.begin(shot, settings_for(shot))
    led.succeed(shot.id, "out/a.mp4", seconds=42)

    rec = led.record(shot.id)
    assert [a.ok for a in rec.attempts] == [False, True]
    assert rec.last_kind() == ""
    assert rec.seconds == pytest.approx(42.0)


def test_failure_kind_is_kept_for_the_retry_ladder() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film)
    shot = film.shots[0]
    led.begin(shot, settings_for(shot))
    led.fail(shot.id, "insufficient VRAM, reduce frames", kind="vram")
    assert led.record(shot.id).last_kind() == "vram"


def test_reset_clears_failures_but_keeps_the_seed() -> None:
    film = build_film(1)
    led = Ledger.in_memory(film, max_attempts=1)
    shot = film.shots[0]
    led.begin(shot, settings_for(shot))
    led.fail(shot.id, "boom")
    pinned = led.record(shot.id).seed

    led.reset(shot.id)
    rec = led.record(shot.id)
    assert rec.status is ShotStatus.PENDING
    assert rec.attempt_count == 0
    assert rec.seed == pinned
    assert led.workable(film)  # workable again


# --------------------------------------------------------------------------
# fail loud, never guess
# --------------------------------------------------------------------------


def test_a_corrupt_ledger_is_refused_not_guessed(tmp_path) -> None:
    film = build_film()
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(LedgerError, match="corrupt"):
        Ledger.open(path, film)


def test_a_ledger_from_another_film_is_refused(tmp_path) -> None:
    path = tmp_path / "state.json"
    Ledger.open(path, build_film(3)).save()

    other = Film(
        id="f99", title="t", language="ar", target_minutes=1, scenes=build_film(2).scenes
    )
    with pytest.raises(LedgerError, match="belongs to film"):
        Ledger.open(path, other)


def test_succeed_without_begin_is_refused() -> None:
    led = Ledger.in_memory(build_film(1))
    with pytest.raises(LedgerError, match="without begin"):
        led.succeed("s01_sh001", "out/a.mp4")


def test_fail_without_begin_is_refused() -> None:
    led = Ledger.in_memory(build_film(1))
    with pytest.raises(LedgerError, match="without begin"):
        led.fail("s01_sh001", "boom")


def test_unknown_shot_id_is_refused() -> None:
    led = Ledger.in_memory(build_film(1))
    with pytest.raises(LedgerError, match="not in this ledger"):
        led.status("s01_sh999")


# --------------------------------------------------------------------------
# atomicity
# --------------------------------------------------------------------------


def test_save_never_leaves_a_half_written_file(tmp_path) -> None:
    """A crash mid-write must leave the previous good state, not garbage."""
    film = build_film(3)
    path = tmp_path / "state.json"
    led = Ledger.open(path, film)
    led.begin(film.shots[0], settings_for(film.shots[0]))
    led.succeed("s01_sh001", "out/a.mp4", seconds=1)
    good = path.read_text("utf-8")

    class Boom(RuntimeError):
        pass

    real_replace = __import__("os").replace

    def explode(*a, **kw):
        raise Boom("disk died")

    import openclude.state as st

    st.os.replace = explode
    try:
        with pytest.raises(Boom):
            led.begin(film.shots[1], settings_for(film.shots[1]))
    finally:
        st.os.replace = real_replace

    # the old state is intact and still parseable
    assert path.read_text("utf-8") == good
    assert json.loads(path.read_text("utf-8"))["records"]["s01_sh001"]["status"] == "done"
    # and no temp files were left behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_ledger_keeps_unknown_shots_out(tmp_path) -> None:
    """If the film shrinks, stale records must not linger."""
    big = build_film(5)
    path = tmp_path / "state.json"
    led = Ledger.open(path, big)
    led.begin(big.shots[4], settings_for(big.shots[4]))

    smaller = build_film(2)
    again = Ledger.open(path, smaller)
    assert "s01_sh005" not in again.records
    assert set(again.records) == {"s01_sh001", "s01_sh002"}


# --------------------------------------------------------------------------
# fingerprinting
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_and_order_independent() -> None:
    a = {"model_type": "x", "seed": 1, "prompt": "hello"}
    b = {"prompt": "hello", "seed": 1, "model_type": "x"}
    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_changes_when_settings_change() -> None:
    a = {"model_type": "x", "video_length": 49}
    b = {"model_type": "x", "video_length": 81}
    assert fingerprint(a) != fingerprint(b)


# --------------------------------------------------------------------------
# observability
# --------------------------------------------------------------------------


def test_summary_reports_counts_a_human_can_act_on() -> None:
    film = build_film(4)
    led = Ledger.in_memory(film, max_attempts=1)
    led.begin(film.shots[0], settings_for(film.shots[0]))
    led.succeed("s01_sh001", "out/a.mp4", seconds=120)
    led.begin(film.shots[1], settings_for(film.shots[1]))
    led.fail("s01_sh002", "CUDA out of memory", kind="vram")
    led.skip("s01_sh003", "removed from script")

    line = led.summary(film)
    for token in ("film=f01", "shots=4", "done=1", "failed=1", "skipped=1"):
        assert token in line
    assert "progress=25.0%" in line
    assert "gpu=2.0min" in line


def test_failures_are_listed_in_a_stable_order() -> None:
    film = build_film(3)
    led = Ledger.in_memory(film)
    for shot in reversed(film.shots):
        led.begin(shot, settings_for(shot))
        led.fail(shot.id, "boom")
    assert [sid for sid, _ in led.failures()] == [
        "s01_sh001", "s01_sh002", "s01_sh003"
    ]


def test_default_max_attempts_matches_the_queue_policy() -> None:
    assert DEFAULT_MAX_ATTEMPTS == 4
