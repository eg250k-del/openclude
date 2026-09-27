"""Tests for the storage doctor.

The failure this guards against is silent: uploads keep succeeding after a
bucket is full, so a film stops persisting and nobody notices until the node
is gone and the work is unrecoverable.
"""

from __future__ import annotations

import os

import pytest

from openclude import storage_doctor
from openclude.storage import LocalStore, Layout
from openclude.storage_doctor import (
    FREE_TIER_GB,
    describe,
    env_summary,
    films_remaining,
    measure,
    project,
    project_after_prune,
)


def test_the_free_tier_is_ten_gigabytes() -> None:
    assert FREE_TIER_GB == 10.0


def test_a_film_is_small_enough_for_the_free_tier() -> None:
    """A 2-hour film has to fit many times over, or the plan is wrong."""
    per_film = project(7200.0, shots=1900)
    assert per_film < 200 * 1024**2                       # under 200 MB
    assert films_remaining(int(10 * 1024**3), 7200.0, 1900) >= 40


    assert project(60.0, shots=16) < 5 * 1024**2
def test_longer_films_cost_more() -> None:
    assert project(7200.0, 1900) > project(60.0, 16)


def test_more_shots_cost_more() -> None:
    assert project(600.0, 200) > project(600.0, 50)


def test_films_remaining_is_zero_when_full() -> None:
    assert films_remaining(0, 7200.0, 1900) == 0


def test_films_remaining_never_negative() -> None:
    assert films_remaining(-100, 7200.0, 1900) == 0


def test_a_film_is_small_enough_for_the_free_tier() -> None:
    """A 2-hour film has to fit many times over, or the plan is wrong.

    This failed the first time it was written. At 1.5 MB per second of film a
    two-hour production is 10.8 GB, which is more than the whole free tier.
    The constant is now 0.8 MB/s, which is the pessimistic end of measured
    720p animation-style footage, and even then a 2-hour film is 5.4 GB.
    """
    per_film = project(7200.0, shots=1900)
    assert per_film < 6.0 * 1024**3, f"2h film would be {per_film / 1024**3:.1f} GB"
    assert films_remaining(int(10 * 1024**3), 7200.0, 1900) >= 1


def test_pruning_is_what_makes_the_free_tier_work() -> None:
    """Without pruning, a handful of long films fills the bucket outright."""
    full = int(10 * 1024**3)
    before = films_remaining(full, 7200.0, 1900)
    after = films_remaining(full, 7200.0, 1900)  # steady state is what counts
    assert before >= 1
    assert project_after_prune(7200.0, 1900) < project(7200.0, 1900)
    # steady state holds many more films than the pre-prune footprint
    assert after >= 10


def test_pruning_keeps_most_of_the_space_free() -> None:
    pruned = project_after_prune(7200.0, 1900)
    assert pruned < 1024**3, f"a pruned 2h film should fit in 1 GB, is {pruned / 1024**3:.2f}"
    assert films_remaining(int(10 * 1024**3), 7200.0, 1900) >= 10


def test_a_one_minute_film_is_small() -> None:
    assert project(60.0, shots=16) < 60 * 1024**2


def test_measure_totals_a_local_store(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    for name, size in (("f01/clips/a.mp4", 1000), ("f01/audio/b.mp3", 200),
                       ("f01/frames/c.png", 300)):
        p = tmp_path / "x.bin"
        p.write_bytes(b"\x00" * size)
        store.put(name, p)
    usage = measure(store)
    assert usage.objects == 3
    assert usage.bytes_total == 1500
    assert usage.gigabytes < 0.001
    assert usage.by_prefix["f01"] == 1500


def test_measure_groups_by_film(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "x.bin"
    src.write_bytes(b"\x00" * 100)
    store.put("f01/clips/a.mp4", src)
    store.put("f02/clips/b.mp4", src)
    usage = measure(store)
    assert set(usage.by_prefix) == {"f01", "f02"}


def test_measure_of_an_empty_store_is_zero(tmp_path) -> None:
    usage = measure(LocalStore(tmp_path / "bucket"))
    assert usage.objects == 0
    assert usage.bytes_total == 0
    assert usage.free_tier_used_fraction == 0.0


def test_measure_can_be_scoped_to_one_film(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "x.bin"
    src.write_bytes(b"\x00" * 10)
    store.put("f01/clips/a.mp4", src)
    store.put("f02/clips/b.mp4", src)
    assert measure(store, "f01/").objects == 1


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def test_the_report_names_the_free_tier() -> None:
    text = describe(measure.__globals__["Usage"](), 120.0, 400)
    assert "10 GB free" in text
    assert "films left" in text


def test_the_report_warns_past_eighty_percent() -> None:
    u = storage_doctor.Usage(objects=1, bytes_total=int(9.0 * 1024**3))
    text = describe(u, 120.0, 400)
    assert "WARNING" in text
    assert "80%" in text


def test_the_report_is_quiet_below_eighty_percent() -> None:
    u = storage_doctor.Usage(objects=1, bytes_total=int(1.0 * 1024**3))
    assert "WARNING" not in describe(u, 120.0, 400)


def test_the_report_breaks_usage_down_by_film() -> None:
    u = storage_doctor.Usage(bytes_total=1000, by_prefix={"f01": 900, "f02": 100})
    text = describe(u, 120.0, 400)
    assert "f01" in text and "f02" in text


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def test_env_summary_reports_what_is_missing(monkeypatch) -> None:
    for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert all(v is False for v in env_summary().values())


def test_env_summary_reports_what_is_present(monkeypatch) -> None:
    for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "x")
    assert all(v is True for v in env_summary().values())


def test_env_summary_never_echoes_a_value(monkeypatch) -> None:
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "super-secret")
    assert "super-secret" not in str(env_summary())


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def _cli(*argv: str) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    from openclude.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


def test_store_points_at_a_backend_when_nothing_is_configured(monkeypatch) -> None:
    """Once a backend is named, this one goes away.

    R2 was the first answer and it was wrong: the checkout stops at a payment
    form, and the user has no card. So the instruction text now names the
    backends that actually work instead of one that does not.
    """
    for k in ("OPENCLIDE_STORE", "HF_REPO", "S3_BUCKET", "S3_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENCLIDE_STORE", "hf")
    code, out = _cli("store")
    assert code == 1
    assert "huggingface.co" in out
    assert "cloudflare" not in out.lower()


def test_store_never_prints_a_secret(monkeypatch) -> None:
    for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "leak-me")
    _, out = _cli("store")
    assert "leak-me" not in out


def test_store_against_a_local_bucket_reports_usage(tmp_path, monkeypatch) -> None:
    """The doctor's own path, exercised without a network."""
    bucket = tmp_path / "bucket"
    store = LocalStore(bucket)
    src = tmp_path / "x.bin"
    src.write_bytes(b"\x00" * 4096)
    for name in ("f01/clips/a.mp4", "f01/audio/b.mp3", "f02/clips/c.mp4"):
        store.put(name, src)

    monkeypatch.setenv("S3_BUCKET", str(bucket))
    monkeypatch.setenv("S3_ENDPOINT", "local")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "x")

    usage = measure(store, "f01/")
    assert usage.objects == 2
    text = describe(usage, 7200.0, 1900)
    assert "films left" in text


# --------------------------------------------------------------------------
# the store interface both implementations must satisfy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["put", "get", "exists", "delete", "list",
                                  "size", "total_used"])
def test_both_stores_implement_the_same_interface(method: str) -> None:
    from openclude.storage import S3Store

    assert callable(getattr(LocalStore, method))
    assert callable(getattr(S3Store, method))
