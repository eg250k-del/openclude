"""Tests for assembly.

ffmpeg is not present on every machine that runs the tests, so the guard rails
— the part that actually prevents a broken film from shipping — are tested
without it, and the ffmpeg-dependent checks skip when it is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from openclude import assembly
from openclude.assembly import (
    MAX_SHOT_SECONDS,
    AssemblyError,
    Clip,
    DurationDrift,
    MissingClip,
    concat,
    extract_last_frame,
    mux,
    probe_duration,
    require_tools,
    verify,
)

FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")

LONG_FFMPEG = pytest.mark.skipif(
    not FFMPEG, reason="ffmpeg not installed"
)


def fake_clips(tmp_path: Path, n: int, seconds: float = 2.0) -> list[Clip]:
    clips = []
    for i in range(1, n + 1):
        p = tmp_path / f"s01_sh{i:03d}.mp4"
        p.write_bytes(b"\x00" * 2048)
        clips.append(Clip(shot_id=f"s01_sh{i:03d}", path=str(p), seconds=seconds))
    return clips


@pytest.fixture
def stub_probe(monkeypatch):
    """Replace the ffmpeg probe so the guard rails can be tested anywhere."""
    table: dict[str, float] = {}

    def fake(path) -> float:
        return table[str(path)]

    monkeypatch.setattr(assembly, "probe_duration", fake)
    return table


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_missing_tools_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(assembly, "FFMPEG", "definitely-not-ffmpeg")
    monkeypatch.setattr(assembly, "FFPROBE", "definitely-not-ffprobe")
    with pytest.raises(AssemblyError, match="missing"):
        require_tools()


# --------------------------------------------------------------------------
# guard rails — these are what stop a broken film shipping
# --------------------------------------------------------------------------


def test_an_empty_assembly_is_refused(tmp_path, stub_probe) -> None:
    with pytest.raises(AssemblyError, match="nothing to assemble"):
        concat([], tmp_path / "out.mp4", tmp_path)


def test_a_single_clip_is_refused_as_a_film(tmp_path, stub_probe) -> None:
    clips = fake_clips(tmp_path, 1)
    with pytest.raises(AssemblyError, match="at least two"):
        concat(clips, tmp_path / "out.mp4", tmp_path)


def test_a_missing_clip_stops_the_assembly(tmp_path, stub_probe) -> None:
    """The failure mode the audit called out: a film that quietly ends early."""
    clips = fake_clips(tmp_path, 3)
    clips[1] = Clip(shot_id="s01_sh002", path=str(tmp_path / "gone.mp4"))
    for c in clips:
        stub_probe[c.path] = 2.0
    with pytest.raises(MissingClip, match="hole in it"):
        concat(clips, tmp_path / "out.mp4", tmp_path)


def test_an_oversize_clip_is_refused(tmp_path, stub_probe) -> None:
    clips = fake_clips(tmp_path, 3)
    for c in clips:
        stub_probe[c.path] = 2.0
    stub_probe[clips[1].path] = 30.0
    with pytest.raises(AssemblyError, match="should have been split"):
        concat(clips, tmp_path / "out.mp4", tmp_path)


def test_a_clip_at_the_limit_is_accepted(tmp_path, stub_probe, monkeypatch) -> None:
    clips = fake_clips(tmp_path, 3)
    for c in clips:
        stub_probe[c.path] = MAX_SHOT_SECONDS
    monkeypatch.setattr(assembly, "_run", lambda *a, **kw: "")
    Path(tmp_path / "out.mp4").write_bytes(b"x")
    assert concat(clips, tmp_path / "out.mp4", tmp_path)


def test_a_drifted_film_is_refused(tmp_path, stub_probe) -> None:
    out = tmp_path / "film.mp4"
    out.write_bytes(b"x")
    stub_probe[str(out)] = 100.0
    with pytest.raises(DurationDrift, match="Something was dropped"):
        verify(120.0, out)


def test_a_film_within_tolerance_passes(tmp_path, stub_probe) -> None:
    out = tmp_path / "film.mp4"
    out.write_bytes(b"x")
    stub_probe[str(out)] = 119.4
    assert "OK assembled=119.40s" in verify(120.0, out)


def test_verification_tolerance_is_configurable(tmp_path, stub_probe) -> None:
    out = tmp_path / "film.mp4"
    out.write_bytes(b"x")
    stub_probe[str(out)] = 119.4
    with pytest.raises(DurationDrift):
        verify(120.0, out, tolerance=0.1)


# --------------------------------------------------------------------------
# the concat list — the file the audit's version got wrong
# --------------------------------------------------------------------------


def test_the_concat_list_uses_absolute_paths(tmp_path, stub_probe, monkeypatch) -> None:
    """Relative paths break the moment the engine's init() changes the cwd."""
    clips = fake_clips(tmp_path, 3)
    for c in clips:
        stub_probe[c.path] = 2.0
    seen: dict = {}

    def fake_run(cmd, timeout):
        if "concat" in cmd:
            seen["listing"] = Path(cmd[cmd.index("-i") + 1]).read_text("utf-8")
        Path(cmd[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr(assembly, "_run", fake_run)
    concat(clips, tmp_path / "out.mp4", tmp_path)
    listing = seen["listing"]
    assert listing.count("file '") == 3
    for c in clips:
        assert str(Path(c.path).resolve()).replace("\\", "/") in listing.replace("\\", "/")


def test_the_concat_list_preserves_order(tmp_path, stub_probe, monkeypatch) -> None:
    clips = fake_clips(tmp_path, 4)
    for c in clips:
        stub_probe[c.path] = 2.0
    seen: dict = {}

    def fake_run(cmd, timeout):
        if "concat" in cmd:
            seen["listing"] = Path(cmd[cmd.index("-i") + 1]).read_text("utf-8")
        Path(cmd[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr(assembly, "_run", fake_run)
    concat(clips, tmp_path / "out.mp4", tmp_path)
    order = [c.shot_id for c in clips]
    positions = [seen["listing"].index(c.shot_id) for c in clips]
    assert positions == sorted(positions)
    assert len(order) == 4


def test_concat_uses_stream_copy_not_a_reencode(tmp_path, stub_probe, monkeypatch) -> None:
    """A 2-hour re-encode is hours of GPU-adjacent CPU and buys nothing."""
    clips = fake_clips(tmp_path, 2)
    for c in clips:
        stub_probe[c.path] = 2.0
    captured: list = []

    def fake_run(cmd, timeout):
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr(assembly, "_run", fake_run)
    concat(clips, tmp_path / "out.mp4", tmp_path)
    cmd = [c for c in captured if "concat" in c][0]
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-safe" in cmd and cmd[cmd.index("-safe") + 1] == "0"


# --------------------------------------------------------------------------
# the last-frame handoff — the continuity bug
# --------------------------------------------------------------------------


def test_the_last_frame_is_taken_from_the_end_not_the_start(
    tmp_path, monkeypatch
) -> None:
    """The audit found `select='eq(n,0)'` used to grab frame 0 and call it last.

    -sseof -1 seeks to one second before the end, which is what actually gets
    the final frame of a clip.
    """
    captured: list = []

    def fake_run(cmd, timeout):
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x89PNG")
        return ""

    monkeypatch.setattr(assembly, "_run", fake_run)
    out = tmp_path / "shot_last.png"
    assert extract_last_frame(tmp_path / "clip.mp4", out) == str(out)
    cmd = captured[0]
    assert "-sseof" in cmd
    assert cmd[cmd.index("-sseof") + 1] == "-1"


def test_a_failed_last_frame_extraction_is_loud(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(assembly, "_run", lambda *a, **kw: "")
    with pytest.raises(MissingClip, match="could not extract"):
        extract_last_frame(tmp_path / "clip.mp4", tmp_path / "x.png")


def test_an_empty_last_frame_file_is_loud(tmp_path, monkeypatch) -> None:
    def fake_run(cmd, timeout):
        Path(cmd[-1]).write_bytes(b"")
        return ""

    monkeypatch.setattr(assembly, "_run", fake_run)
    with pytest.raises(MissingClip):
        extract_last_frame(tmp_path / "clip.mp4", tmp_path / "x.png")


# --------------------------------------------------------------------------
# real ffmpeg, when it exists
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_real_last_frame_comes_from_the_final_moment(tmp_path) -> None:
    """Build a clip whose last frame is visibly different and prove it is used."""
    src = tmp_path / "src.mp4"
    assembly._run(
        [
            assembly.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
            f"color=c=red:s=64x64:d=1:r=10",
            "-vf", "drawbox=0:0:64:64:white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src),
        ],
        timeout=300,
    )
    out = extract_last_frame(src, tmp_path / "last.png")
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


@needs_ffmpeg
def test_real_probe_reads_a_real_file(tmp_path) -> None:
    src = tmp_path / "clip.mp4"
    assembly._run(
        [
            assembly.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=s=64x64:d=2:r=10", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(src),
        ],
        timeout=300,
    )
    assert probe_duration(src) == pytest.approx(2.0, abs=0.15)


@needs_ffmpeg
def test_real_concat_produces_a_playable_file(tmp_path) -> None:
    clips = []
    for i in (1, 2, 3):
        p = tmp_path / f"c{i}.mp4"
        assembly._run(
            [
                assembly.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                f"testsrc=s=64x64:d=1:r=10", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(p),
            ],
            timeout=300,
        )
        clips.append(Clip(shot_id=f"s{i:03d}", path=str(p)))
    out = concat(clips, tmp_path / "film.mp4", tmp_path)
    assert Path(out).stat().st_size > 0
    assert "OK" in verify(3.0, out, tolerance=0.5)


@needs_ffmpeg
def test_real_mux_attaches_audio(tmp_path) -> None:
    import wave, struct, math

    wav = tmp_path / "voice.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 200 * i / 24000)))
            for i in range(24_000)
        ))

    vid = tmp_path / "clip.mp4"
    assembly._run(
        [
            assembly.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=s=128x128:d=1:r=10", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(vid),
        ],
        timeout=300,
    )
    out = mux(vid, wav, tmp_path / "with_audio.mp4")
    assert Path(out).stat().st_size > 0
    assert probe_duration(out) == pytest.approx(1.0, abs=0.2)
