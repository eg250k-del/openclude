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
    calls: list[dict] = []

    class FakeClient:
        def file_info(self, **kw):
            calls.append(kw)
            raise RuntimeError("miss")

    store = HFStore("a/b", client=FakeClient())
    store.exists("f01/clips/x.mp4")
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["filename"] == "f01/clips/x.mp4"


def test_hf_listing_filters_by_prefix() -> None:
    class FakeClient:
        def list_repo_files(self, **_kw):
            return [
                "f01/clips/a.mp4", "f01/frames/b.png", "f02/clips/c.mp4",
                ".gitattributes",
            ]

    store = HFStore("a/b", client=FakeClient())
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
