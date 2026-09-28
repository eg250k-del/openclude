"""Tests for mirroring models between the container and external storage.

This is step three of the order the user described, and it is the step the
whole project is for: you install tools in ComfyUI, and they must be somewhere
that survives the container being stopped.

The cases that matter most are the ones about not doing work twice, because
everything here is measured in gigabytes over a residential connection that
the user is paying for by the hour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MS = Path(__file__).resolve().parents[1] / "tools" / "model_store.py"


@pytest.fixture
def ms():
    spec = importlib.util.spec_from_file_location("model_store", MS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path, spec: dict[str, bytes]) -> None:
    for rel, data in spec.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# --------------------------------------------------------------------------
# walking
# --------------------------------------------------------------------------


def test_it_finds_files_at_every_depth(ms, tmp_path) -> None:
    _tree(tmp_path, {
        "checkpoints/a.safetensors": b"x" * 10,
        "vae/b.safetensors": b"y" * 10,
        "loras/sub/c.safetensors": b"z" * 10,
    })
    found = dict(ms.local_walk(str(tmp_path)))
    assert set(found) == {
        "checkpoints/a.safetensors", "vae/b.safetensors", "loras/sub/c.safetensors"}


def test_a_missing_root_is_reported_not_silently_empty(ms, tmp_path) -> None:
    """An empty listing and an absent folder look the same, and the difference
    is whether anything gets restored."""
    with pytest.raises(FileNotFoundError):
        ms.local_walk(str(tmp_path / "nope"))


def test_pyc_files_are_skipped(ms, tmp_path) -> None:
    """A custom node installs dozens of these and they are never worth copying."""
    _tree(tmp_path, {"__pycache__/x.pyc": b"junk", "node.py": b"code"})
    found = dict(ms.local_walk(str(tmp_path)))
    assert "node.py" in found
    assert not any(f.endswith(".pyc") for f in found)


def test_git_metadata_is_skipped(ms, tmp_path) -> None:
    """A directory literally named ".git", not a repo called "a.git"."""
    _tree(tmp_path, {".git/config": b"x", "node/main.py": b"y"})
    found = dict(ms.local_walk(str(tmp_path)))
    assert not any(".git" in f.split("/") for f in found)
    assert "node/main.py" in found


def test_temp_folders_are_skipped(ms, tmp_path) -> None:
    """ComfyUI writes large intermediates here and they are never needed."""
    _tree(tmp_path, {"temp/blob.bin": b"x" * 100, "vae/real.safetensors": b"y"})
    found = dict(ms.local_walk(str(tmp_path)))
    assert "temp/blob.bin" not in found
    assert "vae/real.safetensors" in found


# --------------------------------------------------------------------------
# not doing the work twice
# --------------------------------------------------------------------------


def test_a_file_already_saved_is_not_copied_again(ms) -> None:
    """30 GB at a time, on a metered connection, while paying by the hour."""
    local = {"a.safetensors": 100, "b.safetensors": 200}
    remote = {"a.safetensors": 100, "b.safetensors": 200}
    p = ms.plan_sync(local, remote)
    assert p["new"] == {}
    assert p["changed"] == {}
    assert p["bytes"] == 0
    assert len(p["unchanged"]) == 2


def test_a_new_file_is_copied(ms) -> None:
    p = ms.plan_sync({"new.safetensors": 500}, {"old.safetensors": 100})
    assert set(p["new"]) == {"new.safetensors"}
    assert p["bytes"] == 500


def test_a_resized_file_is_copied_because_the_old_one_is_wrong(ms) -> None:
    """Same name, different size. Copying nothing would restore a truncated
    model later and fail at load time with a shape error."""
    p = ms.plan_sync({"a.safetensors": 150}, {"a.safetensors": 100})
    assert set(p["changed"]) == {"a.safetensors"}
    assert p["new"] == {}


def test_a_fully_synced_tree_reports_zero_bytes(ms) -> None:
    files = {f"m{i}.safetensors": i * 10 for i in range(50)}
    assert ms.plan_sync(files, dict(files))["bytes"] == 0


# --------------------------------------------------------------------------
# restore
# --------------------------------------------------------------------------


def test_restore_fetches_only_what_is_missing(ms) -> None:
    remote = {"a.safetensors": 100, "b.safetensors": 200}
    local = {"a.safetensors": 100}
    p = ms.plan_restore(remote, local)
    assert set(p["missing"]) == {"b.safetensors"}
    assert set(p["present"]) == {"a.safetensors"}
    assert p["bytes"] == 200


def test_restore_with_nothing_local_fetches_everything(ms) -> None:
    """The ordinary case after a fresh container: the disk is empty."""
    remote = {"a.safetensors": 100, "b.safetensors": 200}
    p = ms.plan_restore(remote, {})
    assert len(p["missing"]) == 2
    assert p["bytes"] == 300


def test_restore_with_a_full_local_tree_fetches_nothing(ms) -> None:
    remote = {"a.safetensors": 100}
    p = ms.plan_restore(remote, {"a.safetensors": 100})
    assert p["missing"] == {}
    assert p["bytes"] == 0


def test_restore_treats_an_absent_folder_as_empty(ms) -> None:
    """Not a crash. A fresh container has no models folder at all yet."""
    assert ms.plan_restore({"a": 1}, {})["bytes"] == 1


# --------------------------------------------------------------------------
# the prefix mapping
# --------------------------------------------------------------------------


class _Store:
    def __init__(self, sizes): self._s = sizes

    def sizes(self): return dict(self._s)


def test_only_the_requested_prefix_is_considered(ms) -> None:
    store = _Store({"models/a": 1, "custom_nodes/b": 2, "outputs/img.png": 3})
    assert ms._remote_sizes(store, "models") == {"a": 1}
    assert ms._remote_sizes(store, "custom_nodes") == {"b": 2}


def test_a_prefix_that_matches_nothing_is_empty_not_everything(ms) -> None:
    store = _Store({"custom_nodes/b": 2})
    assert ms._remote_sizes(store, "models") == {}


def test_an_empty_prefix_is_refused_rather_than_meaning_everything(ms) -> None:
    """A restore with no prefix would pull the user's generated videos out of
    the repo and write them into a models folder. That is quiet, destructive,
    and must be impossible rather than unlikely."""
    store = _Store({"models/a": 1, "outputs/img.png": 2})
    with pytest.raises(ValueError, match="prefix is required"):
        ms._remote_sizes(store, "")
    with pytest.raises(ValueError, match="prefix is required"):
        ms._remote_sizes(store, "   ")
