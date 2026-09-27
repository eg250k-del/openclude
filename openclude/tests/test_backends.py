"""Tests for the storage backends and how one is chosen.

There is one requirement that decided the design: the user has no credit
card. Cloudflare R2, Backblaze B2 and Google Cloud Storage all stop at a
payment form, so none of them work. Hugging Face is free with no card, and
`huggingface_hub` is already installed in the image because the engine uses
it to fetch model weights, so it adds no new dependency.

The consequence is that the pipeline must not care which backend it has.
These tests pin that: identical behaviour, identical layout, one env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openclude.storage import HFStore, Layout, LocalStore, S3Store, StorageError
from openclude.storage_doctor import (
    BACKENDS,
    HF,
    LOCAL,
    S3,
    build_store,
    how_to_configure,
    required_env,
    selected_backend,
)

STORE_METHODS = ("put", "get", "exists", "delete", "list", "size", "total_used")


# --------------------------------------------------------------------------
# every backend implements the same contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LocalStore, S3Store, HFStore])
def test_every_backend_implements_the_whole_interface(cls) -> None:
    for name in STORE_METHODS:
        assert callable(getattr(cls, name)), f"{cls.__name__}.{name}"


def test_there_are_exactly_three_backends() -> None:
    assert set(BACKENDS) == {"local", "s3", "hf"}


# --------------------------------------------------------------------------
# choosing one
# --------------------------------------------------------------------------


def test_nothing_configured_falls_back_to_local(monkeypatch) -> None:
    for k in ("OPENCLIDE_STORE", "HF_REPO", "S3_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    assert selected_backend() == LOCAL


def test_hf_is_chosen_when_a_repo_is_set(monkeypatch) -> None:
    monkeypatch.delenv("OPENCLIDE_STORE", raising=False)
    monkeypatch.setenv("HF_REPO", "someone/something")
    assert selected_backend() == HF


def test_s3_is_chosen_when_a_bucket_is_set(monkeypatch) -> None:
    monkeypatch.delenv("OPENCLIDE_STORE", raising=False)
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.setenv("S3_BUCKET", "openclude")
    assert selected_backend() == S3


def test_an_explicit_choice_wins_over_detection(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLIDE_STORE", "hf")
    monkeypatch.setenv("S3_BUCKET", "openclude")
    assert selected_backend() == HF


def test_a_mistyped_backend_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLIDE_STORE", "dropbox")
    with pytest.raises(RuntimeError, match="not one of"):
        selected_backend()


def test_the_backend_is_never_guessed_when_asked_explicitly() -> None:
    """`local` is the only backend that needs nothing, and it is not durable."""
    assert required_env(LOCAL) == ()
    assert "HF_TOKEN" in required_env(HF)
    assert len(required_env(S3)) == 4


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def test_local_needs_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENCLIDE_STORE", raising=False)
    monkeypatch.setenv("OPENCLIDE_LOCAL", str(tmp_path / "s"))
    store = build_store(LOCAL)
    assert isinstance(store, LocalStore)


def test_building_hf_without_a_token_says_what_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("HF_REPO", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_REPO"):
        build_store(HF)


def test_building_s3_without_keys_says_what_is_missing(monkeypatch) -> None:
    for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="S3_BUCKET"):
        build_store(S3)


# --------------------------------------------------------------------------
# HF specifics
# --------------------------------------------------------------------------


def test_hf_requires_a_username_and_repo_name() -> None:
    with pytest.raises(StorageError, match="username/repo-name"):
        HFStore("just-a-name")


def test_hf_accepts_a_qualified_repo() -> None:
    store = HFStore("eg250k-del/openclude-store")
    assert store.repo_id == "eg250k-del/openclude-store"
    assert store.private is True


def test_hf_defaults_to_a_private_repo() -> None:
    """A film store is not public content."""
    assert HFStore("a/b").private is True


def test_hf_reads_the_token_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    assert HFStore("a/b").token == "hf_xxx"


def test_hf_uses_the_dataset_repo_type() -> None:
    """It is a file store, not a model. The repo type is part of every call."""
    seen: list[tuple] = []

    class FakeClient:
        def file_exists(self, **kw):
            seen.append(kw)
            return False

    store = HFStore("a/b", client=FakeClient())
    store.exists("f01/clips/x.mp4")
    assert seen[0]["repo_type"] == "dataset"
    assert seen[0]["filename"] == "f01/clips/x.mp4"


def test_hf_listing_filters_by_prefix() -> None:
    files = ["f01/clips/a.mp4", "f01/frames/b.png", "f02/clips/c.mp4",
             ".gitattributes"]
    client = _Liveish({p: 10 for p in files})
    store = HFStore("a/b", client=client)
    assert list(store.list("f01/")) == ["f01/clips/a.mp4", "f01/frames/b.png"]
    assert ".gitattributes" not in list(store.list(""))


def test_hf_listing_of_a_missing_repo_is_empty_not_fatal() -> None:
    class FakeClient:
        def list_repo_files(self, **_kw):
            raise RuntimeError("404 repo does not exist")

    assert list(HFStore("a/b", client=FakeClient()).list("")) == []


def test_hf_deleting_something_absent_is_fine() -> None:
    class FakeClient:
        def delete_file(self, **kw):
            raise RuntimeError("404")

    HFStore("a/b", client=FakeClient()).delete("f01/x.mp4")   # must not raise


def test_hf_refuses_a_zero_byte_upload(tmp_path) -> None:
    store = HFStore("a/b")
    empty = tmp_path / "e.bin"
    empty.write_bytes(b"")
    with pytest.raises(StorageError, match="0-byte"):
        store.put("f01/e.bin", empty)


def test_hf_refuses_a_missing_source(tmp_path) -> None:
    with pytest.raises(StorageError, match="no such file"):
        HFStore("a/b").put("f01/x", tmp_path / "nope")


# --------------------------------------------------------------------------
# the imports actually resolve
#
# This block exists because the first live test against a real repo found a
# bug no unit test could see: `hf_hub_upload` was renamed to `upload_file` in
# huggingface_hub 1.x. Every test passed because they used a fake client and
# never imported the library. The symptom was a store that could read but
# never write, which is precisely how a film disappears.
# --------------------------------------------------------------------------


def test_the_modern_hf_functions_exist() -> None:
    """These are the names the code imports. If one vanishes, writes break."""
    hf = pytest.importorskip("huggingface_hub")
    for name in ("upload_file", "hf_hub_download", "HfApi"):
        assert hasattr(hf, name), (
            f"huggingface_hub {getattr(hf, '__version__', '?')} is missing {name}"
        )


def test_at_least_one_upload_name_resolves() -> None:
    """`hf_hub_upload` is legacy and gone in 1.x; `upload_file` replaced it."""
    hf = pytest.importorskip("huggingface_hub")
    assert any(callable(getattr(hf, n, None)) for n in ("upload_file", "hf_hub_upload"))


def test_the_helper_finds_the_modern_name() -> None:
    """Against the real library, so it skips where the library is absent.

    huggingface_hub is a dependency of the engine, not of openclude, so it is
    present in the image and on a developer machine but not on a bare CI
    runner. The first version of this test imported it unconditionally and
    failed the whole build for exactly that reason.
    """
    from openclude.storage import _hf_callable

    pytest.importorskip("huggingface_hub")
    assert callable(_hf_callable("upload_file", "hf_hub_upload"))


def test_the_helper_reports_a_missing_library_clearly(monkeypatch) -> None:
    """On a runner without it, the message has to name the package."""
    import sys

    from openclude.storage import _hf_callable

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(StorageError) as exc:
        _hf_callable("upload_file", "hf_hub_upload")
    assert "huggingface_hub" in str(exc.value)
    assert "engine" in str(exc.value), "the message should say where it comes from"


def _with_module(monkeypatch, name: str, obj: dict) -> None:
    """Swap a module in sys.modules so these tests never touch the real one.

    huggingface_hub re-exports lazily through a module __getattr__, so
    monkeypatch.delattr on it does not stick: the next getattr regenerates the
    name. Injecting a stand-in module is the only reliable way to test the
    rename handling at all.
    """
    import sys
    import types

    fake = types.ModuleType(name)
    for k, v in obj.items():
        setattr(fake, k, v)
    monkeypatch.setitem(sys.modules, name, fake)


def test_the_helper_falls_back_to_the_legacy_name(monkeypatch) -> None:
    """An older image must keep working, so the fallback is load-bearing."""
    from openclude.storage import _hf_callable

    _with_module(monkeypatch, "huggingface_hub", {"hf_hub_upload": lambda **kw: "ok"})
    assert callable(_hf_callable("upload_file", "hf_hub_upload"))


def test_the_helper_prefers_the_modern_name(monkeypatch) -> None:
    from openclude.storage import _hf_callable

    def modern(**kw):
        return "modern"

    _with_module(monkeypatch, "huggingface_hub",
                 {"upload_file": modern, "hf_hub_upload": lambda **kw: "legacy"})
    assert _hf_callable("upload_file", "hf_hub_upload") is modern


def test_the_helper_says_the_version_when_nothing_matches(monkeypatch) -> None:
    from openclude.storage import _hf_callable

    _with_module(monkeypatch, "huggingface_hub", {"__version__": "9.9.9"})
    with pytest.raises(StorageError, match="9.9.9"):
        _hf_callable("upload_file", "hf_hub_upload")


def test_the_helper_explains_a_missing_library(monkeypatch) -> None:
    import sys

    from openclude.storage import _hf_callable

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(StorageError, match="not installed"):
        _hf_callable("upload_file")


class _RepoOk:
    """Just enough of HfApi to get past the create-on-first-write step."""

    def create_repo(self, **_kw):
        return "created"

    def list_repo_files(self, **_kw):
        return []

    def file_info(self, **_kw):
        raise RuntimeError("miss")

    def delete_file(self, **_kw):
        return None


def test_put_reaches_the_library_rather_than_a_stub(tmp_path, monkeypatch) -> None:
    """A fake client must not be able to satisfy put(). The import is the point."""
    seen: dict = {}

    def modern(*, path_or_fileobj, path_in_repo, repo_id, token=None, repo_type=None):
        seen.update(locals())
        return "ok"

    _with_module(monkeypatch, "huggingface_hub", {"upload_file": modern})

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)
    HFStore("a/b", client=_RepoOk()).put("f01/clips/x.mp4", src)

    assert seen["repo_id"] == "a/b"
    assert seen["path_in_repo"] == "f01/clips/x.mp4"
    assert seen["path_or_fileobj"] == str(src)


def test_put_works_against_the_legacy_signature(tmp_path, monkeypatch) -> None:
    """huggingface_hub 0.x is what an older pinned image would have."""
    seen: dict = {}

    def legacy(repo_id, filename, path, token=None, repo_type=None):
        seen.update(locals())
        return "ok"

    _with_module(monkeypatch, "huggingface_hub", {"hf_hub_upload": legacy})

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)
    HFStore("a/b", client=_RepoOk()).put("f01/clips/x.mp4", src)

    assert seen["filename"] == "f01/clips/x.mp4"
    assert seen["path"] == str(src)


def test_the_adapter_picks_the_modern_signature() -> None:
    from openclude.storage import _upload_kwargs

    def modern(*, path_or_fileobj, path_in_repo, repo_id, token=None, repo_type=None):
        ...

    kw = _upload_kwargs(modern, "a/b", "k.mp4", Path("x.mp4"), "t")
    assert kw["path_in_repo"] == "k.mp4"
    assert kw["path_or_fileobj"] == "x.mp4"
    assert "filename" not in kw and "path" not in kw


def test_the_adapter_picks_the_legacy_signature() -> None:
    from openclude.storage import _upload_kwargs

    def legacy(repo_id, filename, path, token=None, repo_type=None):
        ...

    kw = _upload_kwargs(legacy, "a/b", "k.mp4", Path("x.mp4"), "t")
    assert kw["filename"] == "k.mp4"
    assert kw["path"] == "x.mp4"
    assert "path_in_repo" not in kw


def test_the_adapter_always_says_dataset_not_model() -> None:
    """The second live-test bug: without repo_type the write goes to /models.

    The library defaults to the model namespace. This repo is a dataset, so an
    upload without repo_type is addressed to a repo that does not exist, and
    the failure looks like a permissions problem rather than a namespace one.
    """
    from openclude.storage import _upload_kwargs

    def modern(*, path_or_fileobj, path_in_repo, repo_id, token=None, repo_type=None):
        ...

    for fn in (modern, lambda repo_id, filename, path, token=None, repo_type=None: ...):
        assert _upload_kwargs(fn, "a/b", "k", Path("x"), "t")["repo_type"] == "dataset"


def test_a_failing_upload_reports_the_key_not_a_bare_exception(tmp_path, monkeypatch) -> None:
    def boom(**kw):
        raise RuntimeError("network is down")

    _with_module(monkeypatch, "huggingface_hub", {"upload_file": boom})

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)
    with pytest.raises(StorageError) as exc:
        HFStore("a/b", client=_RepoOk()).put("f01/clips/important.mp4", src)
    assert "f01/clips/important.mp4" in str(exc.value)
    assert "network is down" in str(exc.value)


# --------------------------------------------------------------------------
# the metadata methods
#
# The third live-test bug: `HfApi.file_info` was removed in 1.x. Every caller
# swallowed exceptions, so `exists()` answered False for a file that was
# demonstrably there. restore_if_present() uses exists() to decide whether to
# resume, which means a restarted container would have restarted the whole
# film from shot one, silently, forever. A test that only used a fake client
# could not have found this.
# --------------------------------------------------------------------------


class _TreeEntry:
    def __init__(self, path: str, size: int | None, type_: str = "file") -> None:
        self.path, self.size, self.type = path, size, type_


class _TreeFolder:
    """What Hugging Face actually returns for a directory: no size, no type."""

    def __init__(self, path) -> None:
        self.path = path


class _Liveish:
    """A fake HfApi with the 1.x method set, and nothing else."""

    def __init__(self, files: dict[str, int] | None = None) -> None:
        self.files = files or {}
        self.tree_calls = 0
        self.exists_calls: list[str] = []

    def create_repo(self, **_kw):
        return "ok"

    def list_repo_tree(self, repo_id, recursive=False, expand=False,
                       repo_type=None, token=None):
        self.tree_calls += 1
        for p, s in self.files.items():
            if p.endswith("/"):
                yield _TreeFolder(p)          # a directory, exactly as HF sends it
            else:
                yield _TreeEntry(p, s)

    def file_exists(self, repo_id, filename, repo_type=None, token=None):
        self.exists_calls.append(filename)
        return filename in self.files

    def get_paths_info(self, repo_id, paths, expand=False, repo_type=None, token=None):
        return [_TreeEntry(p, self.files[p]) for p in [paths] if p in self.files]

    def list_repo_files(self, **_kw):
        return list(self.files)

    def delete_file(self, repo_id, filename, repo_type=None, token=None):
        self.files.pop(filename, None)

    def file_info(self, *_a, **_kw):  # the removed 0.x method
        raise AttributeError("'HfApi' object has no attribute 'file_info'")

    def upload_file(self, *, path_or_fileobj=None, path_in_repo=None, repo_id=None,
                    token=None, repo_type=None, **kw):
        """Stand in for the real upload so a write can be observed."""
        from pathlib import Path as P

        self.files[path_in_repo] = P(path_or_fileobj).stat().st_size
        return "ok"


def test_exists_is_true_for_a_file_that_is_there() -> None:
    """The bug that would have restarted every film from shot one."""
    c = _Liveish({"f01/clips/a.mp4": 10})
    assert HFStore("a/b", client=c).exists("f01/clips/a.mp4") is True


def test_exists_is_false_for_a_file_that_is_not() -> None:
    c = _Liveish({"f01/clips/a.mp4": 10})
    assert HFStore("a/b", client=c).exists("f01/clips/ghost.mp4") is False


def test_exists_never_uses_the_removed_file_info() -> None:
    """If it went back to file_info, this client would raise and we would get False."""
    c = _Liveish({"f01/clips/a.mp4": 10})
    HFStore("a/b", client=c).exists("f01/clips/a.mp4")
    assert c.exists_calls == ["f01/clips/a.mp4"]


def test_size_reports_the_real_byte_count() -> None:
    c = _Liveish({"f01/clips/a.mp4": 5000})
    assert HFStore("a/b", client=c).size("f01/clips/a.mp4") == 5000


def test_size_of_a_missing_key_is_zero() -> None:
    assert HFStore("a/b", client=_Liveish({})).size("nope") == 0


def test_sizes_returns_everything_in_one_pass() -> None:
    c = _Liveish({"a/1.mp4": 10, "a/2.mp4": 20, "b/3.mp4": 30})
    assert HFStore("a/b", client=c).sizes() == {"a/1.mp4": 10, "a/2.mp4": 20, "b/3.mp4": 30}


def test_the_listing_is_fetched_once_not_once_per_key() -> None:
    """1900 shots must not mean 1900 round trips."""
    files = {f"f01/clips/s{i:04d}.mp4": 100 for i in range(50)}
    c = _Liveish(files)
    store = HFStore("a/b", client=c)
    for i in range(50):
        store.size(f"f01/clips/s{i:04d}.mp4")
    assert c.tree_calls == 1, f"{c.tree_calls} listings for 50 lookups"


def test_total_used_sums_the_listing() -> None:
    c = _Liveish({"a": 10, "b": 20})
    assert HFStore("a/b", client=c).total_used() == 30


def test_a_write_invalidates_the_cached_listing(monkeypatch) -> None:
    """Otherwise a fresh upload would be invisible and resume would restart."""
    c = _Liveish({})
    _with_module(monkeypatch, "huggingface_hub", {"upload_file": c.upload_file})
    store = HFStore("a/b", client=c)
    store.put("f01/clips/new.mp4", _blob(c, 99))
    assert store.exists("f01/clips/new.mp4") is True
    assert store.size("f01/clips/new.mp4") == 99


def test_a_delete_invalidates_the_cached_listing() -> None:
    c = _Liveish({"f01/clips/gone.mp4": 5})
    store = HFStore("a/b", client=c)
    assert store.exists("f01/clips/gone.mp4") is True
    store.delete("f01/clips/gone.mp4")
    assert store.exists("f01/clips/gone.mp4") is False


def _blob(client, size):
    import tempfile
    from pathlib import Path as P

    p = P(tempfile.mkdtemp()) / "b.bin"
    p.write_bytes(b"\x00" * size)
    return p


def test_a_folder_is_not_counted_as_an_object() -> None:
    """The fourth live-test bug: list_repo_tree returns folders too.

    Neither RepoFile nor RepoFolder carries a `type` attribute, so filtering on
    one let every directory through. That inflated the usage report and handed
    folder names to delete().
    """
    client = _Liveish({"f01/": None, "f01/clips/": None, "f01/clips/a.mp4": 10})
    store = HFStore("a/b", client=client)
    assert list(store.list("f01/")) == ["f01/clips/a.mp4"]
    assert store.sizes() == {"f01/clips/a.mp4": 10}
    assert store.total_used() == 10


def test_hf_metadata_files_are_not_ours() -> None:
    """.gitattributes is 2.5 kB of Hugging Face's own bookkeeping."""
    client = _Liveish({".gitattributes": 2504, "f01/clips/a.mp4": 10})
    store = HFStore("a/b", client=client)
    assert list(store.list("")) == ["f01/clips/a.mp4"]
    assert store.total_used() == 10


def test_a_prune_never_asks_to_delete_a_folder(tmp_path) -> None:
    from openclude.storage import Layout, prune

    client = _Liveish({"f01/": None, "f01/clips/": None, "f01/clips/": None,
                       "f01/clips/s.mp4": 10, "f01/film/final.mp4": 99})
    asked: list[str] = []
    client.delete_file = lambda repo_id, filename, **kw: asked.append(filename)
    prune(HFStore("a/b", client=client), Layout("f01"))
    assert asked == ["f01/clips/s.mp4"], asked


def test_measure_uses_the_batched_report(tmp_path) -> None:
    from openclude.storage_doctor import measure

    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "x"
    for name, n in (("f01/a.mp4", 10), ("f01/b.mp4", 20), ("f02/c.mp4", 30)):
        src.write_bytes(b"\x00" * n)
        store.put(name, src)
    u = measure(store)
    assert u.objects == 3
    assert u.bytes_total == 60
    assert u.by_prefix == {"f01": 30, "f02": 30}


def test_measure_falls_back_when_a_store_cannot_batch() -> None:
    """An old store without sizes() must still be measured, not crash."""
    from openclude.storage_doctor import measure

    class Old:
        def list(self, prefix):
            return iter(["f01/a.mp4", "f01/b.mp4"])

        def size(self, key):
            return 5

        def sizes(self):
            raise AttributeError("no")

    u = measure(Old())
    assert u.objects == 2 and u.bytes_total == 10


def test_the_delete_adapter_picks_the_modern_name() -> None:
    from openclude.storage import _delete_kwargs

    def modern(path_in_repo, repo_id, token=None, repo_type=None):
        ...

    kw = _delete_kwargs(modern, "a/b", "k.mp4", "t")
    assert kw["path_in_repo"] == "k.mp4"
    assert "filename" not in kw


def test_the_delete_adapter_picks_the_legacy_name() -> None:
    from openclude.storage import _delete_kwargs

    def legacy(filename, repo_id, token=None, repo_type=None):
        ...

    assert _delete_kwargs(legacy, "a/b", "k.mp4", "t")["filename"] == "k.mp4"


def test_a_real_delete_failure_is_not_swallowed() -> None:
    """The fifth live-test bug: prune reported a saving and deleted nothing."""
    class Client:
        def create_repo(self, **_kw):
            return "ok"

        def delete_file(self, **_kw):
            raise RuntimeError("repository is locked")

    with pytest.raises(StorageError, match="delete of"):
        HFStore("a/b", client=Client()).delete("f01/clips/a.mp4")


def test_deleting_something_absent_is_still_fine() -> None:
    class Client:
        def create_repo(self, **_kw):
            return "ok"

        def delete_file(self, **_kw):
            raise RuntimeError("404 EntryNotFound")

    HFStore("a/b", client=Client()).delete("f01/clips/ghost.mp4")  # no raise


def test_delete_many_uses_one_commit() -> None:
    """1900 clips must not be 1900 commits."""
    calls: list = []

    class Client:
        def create_repo(self, **_kw):
            return "ok"

        def delete_files(self, **kw):
            calls.append(kw)

        def delete_file(self, **_kw):
            raise AssertionError("must not fall back one at a time")

    store = HFStore("a/b", client=Client())
    store.delete_many([f"f01/clips/s{i}.mp4" for i in range(1900)])
    assert len(calls) == 1
    assert len(calls[0]["delete_patterns"]) == 1900


def test_delete_many_of_nothing_does_nothing() -> None:
    class Client:
        def create_repo(self, **_kw):
            return "ok"

        def delete_files(self, **_kw):
            raise AssertionError("should not be called")

    HFStore("a/b", client=Client()).delete_many([])


def test_delete_many_falls_back_when_batch_is_unavailable() -> None:
    seen: list[str] = []

    class Client:
        def create_repo(self, **_kw):
            return "ok"

        def delete_file(self, path_in_repo=None, repo_id=None, token=None, repo_type=None):
            seen.append(path_in_repo)

    HFStore("a/b", client=Client()).delete_many(["a", "b"])
    assert seen == ["a", "b"]


def test_prune_actually_deletes_on_hf() -> None:
    """The end-to-end version of the bug: a reported saving that never happened."""
    from openclude.storage import Layout, prune

    client = _Liveish({
        "f01/clips/": None,
        "f01/clips/s.mp4": 10,
        "f01/audio/": None,
        "f01/audio/s.wav": 5,
        "f01/film/final.mp4": 99,
        "f01/state/ledger.json": 1,
    })
    client.delete_file = lambda path_in_repo=None, repo_id=None, **kw: client.files.pop(
        path_in_repo, None
    )
    store = HFStore("a/b", client=client)
    freed = prune(store, Layout("f01"))
    assert freed["objects"] == 2
    left = {p for p in client.files if not p.endswith("/")}
    assert left == {"f01/film/final.mp4", "f01/state/ledger.json"}


# --------------------------------------------------------------------------
# the layout is backend-agnostic
# --------------------------------------------------------------------------


def test_the_layout_is_identical_for_every_backend() -> None:
    """The pipeline must not branch on which store it has."""
    a = Layout("f01")
    assert a.key_film() == "f01/film/final.mp4"
    assert a.key_state() == "f01/state/ledger.json"
    assert a.key_clip("s01_sh001") == "f01/clips/s01_sh001.mp4"
    assert a.key_prefix() == "f01/"


def test_keys_are_never_absolute() -> None:
    layout = Layout("f01", base="/data/work")
    for key in (layout.key_film(), layout.key_state(), layout.key_prefix(),
                layout.key_audio("s01_sh001"), layout.key_frame("s01_sh001")):
        assert not Path(key).is_absolute(), key


def test_local_paths_are_absolute_even_for_hf() -> None:
    """A container chdirs; a relative path would write to the wrong place."""
    layout = Layout("f01", base="/data/work")
    assert Path(layout.path_film()).is_absolute()


# --------------------------------------------------------------------------
# instructions
# --------------------------------------------------------------------------


def test_hf_instructions_say_no_card_is_needed() -> None:
    text = "\n".join(how_to_configure(HF))
    assert "no credit card" in text
    assert "huggingface.co" in text
    assert "dataset" in text


def test_s3_instructions_name_every_variable() -> None:
    text = "\n".join(how_to_configure(S3))
    for v in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        assert v in text


def test_local_instructions_say_it_is_not_durable() -> None:
    text = "\n".join(how_to_configure(LOCAL))
    assert "survives a container" in text
    assert "no card" in text


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


def test_store_warns_when_nothing_durable_is_configured(monkeypatch) -> None:
    """The local backend works, so this exits 0, but it must say it is fragile.

    Exiting 0 is right: a folder is a perfectly good place to develop against.
    What is not right is a green result that implies the setup will survive a
    SaladCloud container, because it will not.
    """
    for k in ("OPENCLIDE_STORE", "HF_REPO", "S3_BUCKET", "S3_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENCLIDE_LOCAL", "store")
    code, out = _cli("store")
    assert code == 0
    assert "NOT DURABLE" in out
    assert "OPENCLIDE_STORE=hf" in out


def test_store_never_prints_a_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_REPO", "a/b")
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    _, out = _cli("store")
    assert "hf_supersecret" not in out


def test_store_reports_the_chosen_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLIDE_STORE", "hf")
    monkeypatch.setenv("HF_REPO", "a/b")
    monkeypatch.setenv("HF_TOKEN", "x")
    _, out = _cli("store")
    assert "backend" in out
    assert "hf" in out


def test_a_mistyped_backend_is_reported_by_the_cli(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLIDE_STORE", "dropbox")
    code, out = _cli("store")
    assert code == 1
    assert "not one of" in out
