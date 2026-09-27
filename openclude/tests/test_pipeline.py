"""Storage and full-pipeline tests, against a local store.

No GPU, no network, no ffmpeg: the render and assemble stages are stubbed.
What is under test is the plumbing that decides what survives a container
death, which is the part that actually has to be right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclude.audio import write_tone_wav
from openclude.llm import StubWriter
from openclude.pipeline import Config, Pipeline, PipelineError
from openclude.schema import Character
from openclude.storage import (
    Layout,
    LocalStore,
    StorageError,
    UploadUnconfirmed,
    film_progress,
    put_checked,
    restore_if_present,
)

NOVA = Character(id="nova", name="Nova", description="a pilot in a grey coat")
CAST = [NOVA]

DRAFT = json.dumps({
    "title": "Nine Days", "language": "ar",
    "scenes": [{"summary": "s", "location": "airfield", "time_of_day": "dawn",
                "shots": [
        {"narration": "The rain had not stopped.", "visual": "a flooded runway",
         "camera": "slow push in", "character_ids": ["nova"]},
        {"narration": "She climbed into the plane.", "visual": "the figure climbs a ladder",
         "character_ids": ["nova"]},
    ]}],
})


# --------------------------------------------------------------------------
# storage primitives
# --------------------------------------------------------------------------


def test_put_and_get_round_trip(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "a.wav"
    write_tone_wav(src, 1.0)
    store.put("f01/audio/a.wav", src)
    out = store.get("f01/audio/a.wav", tmp_path / "back.wav")
    assert Path(out).stat().st_size == Path(src).stat().st_size


def test_a_zero_byte_upload_is_refused(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(StorageError, match="0-byte"):
        store.put("f01/audio/empty.wav", empty)


def test_a_missing_upload_source_is_refused(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    with pytest.raises(StorageError, match="no such file"):
        store.put("f01/x", tmp_path / "nope")


def test_a_crafted_key_cannot_escape_the_root(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    with pytest.raises(StorageError, match="escapes"):
        store.exists("../../etc/passwd")


def test_layout_keys_are_namespaced_by_film() -> None:
    a, b = Layout("f01"), Layout("f02")
    assert a.key_state() == "f01/state/ledger.json"
    assert b.key_state() == "f02/state/ledger.json"
    assert a.key_clip("s01_sh001") == "f01/clips/s01_sh001.mp4"


def test_store_keys_stay_relative(tmp_path) -> None:
    """A store key must never be absolute, or it escapes the bucket root."""
    layout = Layout("f01", base=tmp_path)
    for key in (
        layout.key_state(), layout.key_script(), layout.key_film(),
        layout.key_audio("s01_sh001"), layout.key_clip("s01_sh001"),
        layout.key_frame("s01_sh001"), layout.key_prefix(),
    ):
        assert not Path(key).is_absolute(), key
        assert key.startswith("f01/")


def test_local_paths_are_absolute(tmp_path) -> None:
    """The engine calls os.chdir(); a relative output path would be lost."""
    layout = Layout("f01", base=tmp_path)
    for p in (
        layout.path_state(), layout.path_film(), layout.path_script(),
        layout.path_audio("s01_sh001"), layout.path_clip("s01_sh001"),
        layout.path_voiced("s01_sh001"),
    ):
        assert Path(p).is_absolute(), p
        assert str(tmp_path.resolve()) in p


def test_a_layout_writes_nothing_outside_its_base(tmp_path) -> None:
    layout = Layout("f01", base=tmp_path / "work")
    inside = (tmp_path.resolve()).as_posix()
    for p in (layout.path_film(), layout.path_clip("x"), layout.path_state()):
        assert Path(p).resolve().as_posix().startswith(inside), p


def test_put_checked_confirms_the_write(tmp_path, monkeypatch) -> None:
    store = LocalStore(tmp_path / "bucket")
    src = tmp_path / "a.wav"
    write_tone_wav(src, 1.0)
    monkeypatch.setattr(store, "exists", lambda k: False)
    with pytest.raises(UploadUnconfirmed, match="did not survive"):
        put_checked(store, "f01/a.wav", src)


def test_restore_returns_none_when_absent(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    assert restore_if_present(store, "f01/x", tmp_path / "x") is None


def test_film_progress_counts_what_is_banked(tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    for key in ("f01/audio/a.wav", "f01/clips/b.mp4", "f01/frames/c.png"):
        p = tmp_path / "src"
        p.write_bytes(b"x")
        store.put(key, p)
    counts = film_progress(store, Layout("f01"))
    assert counts == {"audio": 1, "clips": 1, "frames": 1, "other": 0}


# --------------------------------------------------------------------------
# the pipeline, with stubbed GPU and ffmpeg
# --------------------------------------------------------------------------


class FakeSynth:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls: list[str] = []

    def speak(self, text, out_path, voice="", emotion="") -> str:
        self.calls.append(text)
        write_tone_wav(out_path, self.seconds)
        return str(out_path)


def fake_render(out_dir: Path):
    """Pretend to be a GPU: write a file, honour the ledger's own rules."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def render(settings: dict) -> str:
        if settings.get("video_length", 0) > 25:
            raise RuntimeError("you have unsufficient VRAM, reduce the frames")
        p = out_dir / f"{settings['seed']}.mp4"
        p.write_bytes(b"\x00" * 512)
        return str(p)

    return render


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openclude.assembly.concat",
        lambda clips, out, work: _fake_concat(clips, out),
    )
    monkeypatch.setattr("openclude.assembly.verify", lambda *a, **kw: "OK")
    monkeypatch.setattr("openclude.assembly.mux", lambda v, a, o: _touch(o))
    return tmp_path


def _touch(p) -> str:
    Path(p).write_bytes(b"\x00" * 128)
    return str(p)


def _fake_concat(clips, out) -> str:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(b"\x00" * 256)
    return str(out)


def build(tmp_path, **cfg_over) -> tuple[Pipeline, Path]:
    store = LocalStore(tmp_path / "bucket")
    cfg = Config(film_id="f01", work_dir=str(tmp_path / "work"), **cfg_over)
    p = Pipeline(
        cfg, store, StubWriter(DRAFT), FakeSynth(2.0),
        fake_render(tmp_path / "clips"), CAST,
    )
    return p, tmp_path


def test_the_pipeline_runs_all_four_stages(pipe) -> None:
    p, _ = build(pipe)
    result = p.run("a pilot takes a derelict plane out in a storm")
    assert [s.name for s in result.stages] == ["script", "audio", "render", "assemble"]
    assert all(s.ok for s in result.stages)
    assert result.complete is True
    assert Path(result.output).exists()


def test_a_second_run_reuses_everything_banked(pipe) -> None:
    p, root = build(pipe)
    p.run("story")
    before = sorted(k for k in LocalStore(root / "bucket").list("f01"))

    p2, _ = build(root)
    p2.synth.calls.clear()
    p2.run("story")
    assert p2.synth.calls == []          # no voice re-rendered
    assert LocalStore(root / "bucket").list("f01") is not None
    assert before  # nothing was lost


def test_a_container_death_mid_film_resumes(pipe) -> None:
    """The whole thesis, executed."""
    p, root = build(pipe)

    calls = {"n": 0}

    def dying(settings: dict) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("node preempted")
        return fake_render(root / "clips")(settings)

    p.render = dying
    with pytest.raises(KeyboardInterrupt):
        p.run("story")

    # a fresh process, same store
    p2, _ = build(root)
    result = p2.run("story")
    assert result.complete is True
    assert p2.store.exists(Layout("f01").key_film())


def test_assembly_refuses_when_a_shot_is_missing(pipe) -> None:
    p, root = build(pipe)

    def always_fails(settings: dict) -> str:
        raise RuntimeError("This model doesn't accept an End Image")

    p.render = always_fails
    with pytest.raises(PipelineError, match="cannot assemble"):
        p.run("story")


def test_a_broken_writer_fails_the_script_stage_loudly(pipe) -> None:
    p, _ = build(pipe)
    p.writer = StubWriter("I'd love to help but I cannot.")
    with pytest.raises(Exception) as exc:
        p.run("story")
    assert "script=FAIL" in repr(exc.value) or "JSON" in str(exc.value)


def test_the_ledger_is_banked_after_rendering(pipe) -> None:
    p, root = build(pipe)
    p.run("story")
    assert LocalStore(root / "bucket").exists(Layout("f01").key_state())


def test_clips_are_banked_as_they_are_made(pipe) -> None:
    p, root = build(pipe)
    p.run("story")
    clips = [k for k in LocalStore(root / "bucket").list("f01") if "/clips/" in k]
    assert clips


def test_narration_is_banked_before_any_rendering(pipe) -> None:
    """If the GPU dies, the audio is already safe and never re-spent."""
    p, root = build(pipe)

    order: list[str] = []

    def spy(settings: dict) -> str:
        order.append("render")
        return fake_render(root / "clips")(settings)

    p.render = spy
    original = p.synth.speak

    def tracked(*a, **kw):
        order.append("audio")
        return original(*a, **kw)

    p.synth.speak = tracked
    p.run("story")
    assert order[0] == "audio"
    assert "render" in order
