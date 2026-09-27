"""Tests for the audio stage.

The WAVs written here are real files read back through the same `probe` the
container uses, so these assert measurement behaviour rather than mock
bookkeeping.
"""

from __future__ import annotations

import wave

import pytest

from openclude.audio import (
    AudioError,
    AudioProbeError,
    apply_measurements,
    probe,
    render_narration,
    timing_report,
    voice_for,
    write_tone_wav,
)
from openclude.llm import to_film, parse_draft
from openclude.schema import Character

NOVA = Character(id="nova", name="Nova", description="a pilot", voice_reference="voices/nova.wav")
CAST = [NOVA]

DRAFT = """
{"title":"t","language":"ar","scenes":[{"summary":"s","location":"l","time_of_day":"dawn",
"shots":[
 {"narration":"One two three four five six.","visual":"a figure walks","character_ids":["nova"]},
 {"narration":"Seven eight nine ten.","visual":"the figure stops","character_ids":["nova"],
  "dialogue_speaker":"nova"}
]}]}
"""


class FakeSynth:
    """Writes a real WAV of a length the test chooses, ignoring any estimate."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls: list[tuple[str, str, str, str]] = []

    def speak(self, text, out_path, voice="", emotion="") -> str:
        self.calls.append((text, str(out_path), voice, emotion))
        write_tone_wav(out_path, self.seconds)
        return str(out_path)


# --------------------------------------------------------------------------
# probing real files
# --------------------------------------------------------------------------


def test_a_written_wave_measures_back_exactly(tmp_path) -> None:
    for seconds in (0.5, 1.0, 2.375, 4.0):
        info = write_tone_wav(tmp_path / "a.wav", seconds)
        assert info.seconds == pytest.approx(seconds, abs=1 / info.sample_rate)


def test_the_written_file_is_a_real_wave(tmp_path) -> None:
    p = tmp_path / "a.wav"
    write_tone_wav(p, 1.0)
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24_000
        assert w.getnframes() == 24_000


def test_a_zero_byte_file_is_a_loud_failure(tmp_path) -> None:
    """The bug that lets a whole film's narration come out silent."""
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioProbeError, match="0 bytes"):
        probe(p)


def test_a_missing_file_is_a_loud_failure(tmp_path) -> None:
    with pytest.raises(AudioProbeError, match="does not exist"):
        probe(tmp_path / "nope.wav")


def test_a_truncated_render_is_caught_by_the_floor(tmp_path) -> None:
    """A voice that rendered 4 ms of audio is a failure, not a short line."""
    film = to_film(parse_draft(DRAFT), CAST)

    class Stunted:
        def speak(self, text, out_path, voice="", emotion=""):
            write_tone_wav(out_path, 0.004)
            return str(out_path)

    with pytest.raises(AudioError, match="under the"):
        render_narration(film, Stunted(), tmp_path)


def test_a_synthesiser_returning_the_wrong_type_is_caught(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)

    class Confused:
        def speak(self, text, out_path, voice="", emotion=""):
            write_tone_wav(out_path, 1.0)
            return 42  # a path, but not a string

    with pytest.raises(AudioError, match="expected a path string"):
        render_narration(film, Confused(), tmp_path)


def test_a_synthesiser_may_return_none(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)

    class Quiet:
        def speak(self, text, out_path, voice="", emotion=""):
            write_tone_wav(out_path, 1.0)
            return None

    audio = render_narration(film, Quiet(), tmp_path)
    assert len(audio) == 2


def test_a_non_audio_file_is_rejected_not_misread(tmp_path) -> None:
    p = tmp_path / "notaudio.wav"
    p.write_bytes(b"this is definitely not a RIFF file, not even close" * 3)
    with pytest.raises(AudioProbeError):
        probe(p)


# --------------------------------------------------------------------------
# voices
# --------------------------------------------------------------------------


def test_narration_uses_the_narrator_voice() -> None:
    from openclude.schema import Shot

    shot = Shot(id="s01_sh001", scene_id="s01", index=1,
                narration="hello", visual="a figure", dialogue_speaker="")
    assert voice_for(shot, CAST)[0] == "narrator"


def test_a_character_line_uses_that_character_voice() -> None:
    from openclude.schema import Shot

    shot = Shot(id="s01_sh001", scene_id="s01", index=1,
                narration="hello", visual="a figure", dialogue_speaker="nova")
    assert voice_for(shot, CAST)[0] == "voices/nova.wav"


def test_an_unknown_speaker_is_refused() -> None:
    from openclude.schema import Shot

    shot = Shot(id="s01_sh001", scene_id="s01", index=1, narration="hi",
                visual="a figure", dialogue_speaker="ghost")
    with pytest.raises(AudioError, match="not in the cast"):
        voice_for(shot, CAST)


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


def test_durations_come_from_the_file_not_the_estimate(tmp_path) -> None:
    """The core promise of this stage."""
    film = to_film(parse_draft(DRAFT), CAST)
    synth = FakeSynth(seconds=3.25)   # deliberately not the estimate
    audio = render_narration(film, synth, tmp_path)

    assert len(audio) == 2
    for info in audio.values():
        assert info.seconds == pytest.approx(3.25, abs=0.01)

    updated = apply_measurements(film, audio)
    for shot in updated.shots:
        assert shot.duration_seconds == pytest.approx(3.25, abs=0.01)


def test_measurements_survive_a_resplit(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    audio = render_narration(film, FakeSynth(seconds=30.0), tmp_path)
    updated = apply_measurements(film, audio)

    assert updated.unsplittable() == ()
    assert len(updated.shots) > len(film.shots)
    # the split divides the real measurement; the total is what must survive
    assert updated.total_seconds() == pytest.approx(2 * 30.0, abs=0.05)
    assert all(s.target_frames() <= 121 for s in updated.shots)
    assert all(s.duration_seconds < 30.0 for s in updated.shots)


def test_total_length_is_preserved_through_measurement(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    audio = render_narration(film, FakeSynth(seconds=2.0), tmp_path)
    updated = apply_measurements(film, audio)
    assert updated.total_seconds() == pytest.approx(4.0, abs=0.05)


def test_missing_audio_stops_the_stage(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    audio = render_narration(film, FakeSynth(seconds=1.0), tmp_path)
    audio.pop("s01_sh002")
    with pytest.raises(AudioError, match="no rendered audio"):
        apply_measurements(film, audio)


def test_a_broken_synthesiser_names_the_shot(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)

    class Broken:
        def speak(self, *a, **kw):
            raise RuntimeError("model weights missing")

    with pytest.raises(AudioError, match="s01_sh001: voice synthesis failed"):
        render_narration(film, Broken(), tmp_path)


def test_silent_shots_are_not_sent_to_a_voice(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    synth = FakeSynth(seconds=1.0)
    render_narration(film, synth, tmp_path)
    assert all(text.strip() for text, *_ in synth.calls)


def test_the_report_makes_drift_visible(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    audio = render_narration(film, FakeSynth(seconds=1.9), tmp_path)
    updated = apply_measurements(film, audio)
    report = timing_report(updated, audio)
    assert "total drift" in report
    assert "s01_sh001" in report
    # 1.9 s snaps to 45 frames = 1.875 s, so drift is small but non-zero
    assert "ms" in report


def test_measurements_are_idempotent(tmp_path) -> None:
    film = to_film(parse_draft(DRAFT), CAST)
    audio = render_narration(film, FakeSynth(seconds=2.0), tmp_path)
    once = apply_measurements(film, audio)
    twice = apply_measurements(once, audio)
    assert [s.id for s in once.shots] == [s.id for s in twice.shots]
    assert once.total_seconds() == pytest.approx(twice.total_seconds(), abs=0.01)
