"""Tests for pruning finished films.

Pruning is the only thing that makes a long film fit in a free tier, which
makes it exactly the code most likely to destroy work by accident. So the
cases that must NOT delete are the ones worth testing.
"""

from __future__ import annotations

import pytest

from openclude.storage import LocalStore, Layout, prunable, prune


def seed(store: LocalStore, layout: Layout, *, with_film: bool, n: int = 3) -> None:
    src = Path("x") if False else None  # placeholder to keep the signature short
    import tempfile
    from pathlib import Path as P

    tmp = P(tempfile.mkdtemp())
    f = tmp / "blob.bin"
    f.write_bytes(b"\x00" * 1000)
    for i in range(n):
        store.put(layout.key_clip(f"s01_sh{i:03d}"), f)
        store.put(layout.key_audio(f"s01_sh{i:03d}"), f)
        store.put(layout.key_frame(f"s01_sh{i:03d}"), f)
    if with_film:
        f2 = tmp / "film.mp4"
        f2.write_bytes(b"\x00" * 5000)
        store.put(layout.key_film(), f2)
    from pathlib import Path as P2

    state = tmp / "ledger.json"
    state.write_text("{}", encoding="utf-8")
    store.put(layout.key_state(), state)


from pathlib import Path  # noqa: E402


# --------------------------------------------------------------------------
# the case that must not delete
# --------------------------------------------------------------------------


def test_an_unfinished_film_is_never_pruned(tmp_path) -> None:
    """Its clips are the only copy of the work. Deleting them loses the film."""
    store = LocalStore(tmp_path / "bucket")
    layout = Layout("f01")
    seed(store, layout, with_film=False)

    freed = prunable(store, layout)
    assert freed["objects"] == 0
    prune(store, layout)
    assert store.list("f01/") != []
    assert any("/clips/" in k for k in store.list("f01/"))


# --------------------------------------------------------------------------
# the case that must delete
# --------------------------------------------------------------------------


def test_a_finished_film_frees_its_clips_and_frames(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    layout = Layout("f01")
    seed(store, layout, with_film=True)

    freed = prunable(store, layout)
    assert freed["objects"] == 9          # 3 shots x clip, audio, frame
    assert freed["clips"] == 3000
    assert freed["audio"] == 3000
    assert freed["frames"] == 3000

    prune(store, layout)
    keys = list(store.list("f01/"))
    assert not any("/clips/" in k for k in keys)
    assert not any("/audio/" in k for k in keys)
    assert not any("/frames/" in k for k in keys)


def test_pruning_never_deletes_the_film(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    layout = Layout("f01")
    seed(store, layout, with_film=True)
    prune(store, layout)
    assert store.exists(layout.key_film())


def test_pruning_never_deletes_the_ledger(tmp_path) -> None:
    """The ledger is what makes a re-run possible, and it is a few kB."""
    store = LocalStore(tmp_path / "bucket")
    layout = Layout("f01")
    seed(store, layout, with_film=True)
    prune(store, layout)
    assert store.exists(layout.key_state())


# --------------------------------------------------------------------------
# safety properties
# --------------------------------------------------------------------------


def test_pruning_one_film_leaves_another_alone(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    a, b = Layout("f01"), Layout("f02")
    seed(store, a, with_film=True)
    seed(store, b, with_film=False)
    prune(store, a)
    assert any("/clips/" in k for k in store.list("f02/"))


def test_pruning_is_idempotent(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    layout = Layout("f01")
    seed(store, layout, with_film=True)
    prune(store, layout)
    second = prune(store, layout)
    assert second["objects"] == 0


def test_pruning_an_empty_store_is_a_noop(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    assert prune(store, Layout("nope"))["objects"] == 0


def test_nothing_outside_the_film_prefix_is_ever_touched(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    import tempfile
    from pathlib import Path as P

    tmp = P(tempfile.mkdtemp())
    f = tmp / "x"
    f.write_bytes(b"\x00" * 10)
    store.put("queue/pending/j1.json", f)
    store.put("other-film/clips/a.mp4", f)

    seed(store, Layout("f01"), with_film=True)
    prune(store, Layout("f01"))
    assert store.exists("queue/pending/j1.json")
    assert store.exists("other-film/clips/a.mp4")
