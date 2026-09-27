"""Audio stage: render a voice, then MEASURE it.

This is the single idea the audit found worth stealing, and the one the other
three repos got wrong in different ways:

  * repo B estimated duration from character count (`len(sentence)/4.5`), then
    silently fell back to proportional guessing when silence detection drifted;
  * repo C did `int(duration * fps)`, which truncates and drops a frame per
    shot, and swallowed every exception into a 0-frame result;
  * repo B's TTS could return a zero-byte MP3 and nothing noticed.

So: no estimation. A shot's length is whatever ffprobe says the rendered
audio is. If the audio is broken, the film stops.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .schema import Character, Film, Scene, SchemaError, Shot, framespec_for

SAMPLE_RATE = 24_000
"""Enough for speech, and it keeps files small enough to move around a lot."""

MIN_NARRATION_SECONDS = 0.25
"""Anything shorter is a broken render, not a short line. A silent voice is the
single most common way a whole film comes out mute, and it looks like success
until someone watches it."""


class AudioError(RuntimeError):
    pass


class AudioProbeError(AudioError):
    pass


# --------------------------------------------------------------------------
# measuring — the only source of truth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioInfo:
    path: str
    seconds: float
    sample_rate: int
    channels: int

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise AudioProbeError(
                f"{self.path}: measured {self.seconds}s of audio, which is not "
                f"a usable narration length. An empty or silent render."
            )


def probe(path: str | os.PathLike[str]) -> AudioInfo:
    """Measure a real audio file. No estimation, no fallback, no guessing."""
    p = Path(path)
    if not p.exists():
        raise AudioProbeError(f"{p} does not exist")
    size = p.stat().st_size
    if size == 0:
        raise AudioProbeError(
            f"{p} is 0 bytes. The text-to-speech step produced nothing. "
            f"Never continue past this."
        )

    # ffprobe is the authority in the container; wave is the offline fallback
    # and reads the header itself, so no dependency and no subprocess.
    try:
        return _probe_with_wave(p)
    except (wave.Error, EOFError):
        return _probe_with_ffprobe(p)


def _probe_with_wave(p: Path) -> AudioInfo:
    with wave.open(str(p), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
    if rate <= 0 or frames <= 0:
        raise AudioProbeError(f"{p} has no frames")
    return AudioInfo(str(p), frames / rate, rate, channels)


def _probe_with_ffprobe(p: Path) -> AudioInfo:
    exe = os.environ.get("FFPROBE_BINARY", "ffprobe")
    cmd = [
        exe, "-v", "error",
        "-show_entries", "stream=duration,sample_rate,channels",
        "-of", "json", str(p),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except FileNotFoundError as exc:
        raise AudioProbeError(
            f"{p}: cannot read audio, and ffprobe is not installed"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProbeError(f"{p}: ffprobe timed out after 60s") from exc
    except OSError as exc:
        # a sandbox or a permission problem can block process creation outright;
        # that is a probe failure, not a crash
        raise AudioProbeError(
            f"{p}: could not run ffprobe ({exc}). The file is not a plain WAV."
        ) from exc
    try:
        streams = json.loads(out).get("streams") or []
        s = streams[0]
        return AudioInfo(str(p), float(s["duration"]), int(s["sample_rate"]), int(s["channels"]))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise AudioProbeError(f"{p}: ffprobe returned nothing usable") from exc


# --------------------------------------------------------------------------
# a real, offline voice, so the audio stage is testable without a GPU
# --------------------------------------------------------------------------


def write_tone_wav(
    path: str | os.PathLike[str],
    seconds: float,
    frequency: float = 180.0,
    rate: int = SAMPLE_RATE,
) -> AudioInfo:
    """Write a real WAV whose length is exactly `seconds`.

    Not a mock: a genuine RIFF file that `probe` reads back, which is what
    makes the duration tests meaningful rather than tautological.
    """
    if seconds <= 0:
        raise AudioError(f"seconds must be positive, got {seconds}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * rate))
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            # a quiet decaying tone: real samples, silence-safe
            v = int(6000 * math.sin(2 * math.pi * frequency * i / rate) * math.exp(-i / (rate * 2)))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return probe(p)


# --------------------------------------------------------------------------
# the synthesiser interface
# --------------------------------------------------------------------------


class Synthesiser(Protocol):
    """Renders a line of dialogue to a file. Returns the path."""

    def speak(
        self,
        text: str,
        out_path: str | os.PathLike[str],
        voice: str = "",
        emotion: str = "",
    ) -> str: ...


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotAudio:
    shot_id: str
    path: str
    seconds: float
    voice: str
    emotion: str = ""


def voice_for(shot: Shot, characters: Sequence[Character]) -> tuple[str, str]:
    """Which voice and emotion a shot's line should be spoken with."""
    if not shot.narration.strip():
        return "", ""
    if not shot.dialogue_speaker:
        return "narrator", ""
    index = {c.id: c for c in characters}
    character = index.get(shot.dialogue_speaker)
    if character is None:
        raise AudioError(
            f"shot {shot.id}: dialogue_speaker {shot.dialogue_speaker!r} "
            f"is not in the cast"
        )
    return character.voice_reference or character.id, ""


def render_narration(
    film: Film,
    synth: Synthesiser,
    out_dir: str | os.PathLike[str],
    estimate: Callable[[Shot], float] | None = None,
) -> dict[str, ShotAudio]:
    """Speak every shot, then measure what actually came out.

    `estimate` only sizes the placeholder the synthesiser writes; the returned
    durations always come from `probe`. A synthesiser that ignores the
    estimate entirely still produces correct results.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, ShotAudio] = {}

    for shot in film.shots:
        if not shot.narration.strip():
            continue
        voice, emotion = voice_for(shot, film.characters)
        target = shot.duration_seconds or (estimate(shot) if estimate else 2.0)
        path = out / f"{shot.id}.wav"
        try:
            written = synth.speak(
                shot.narration, path, voice=voice, emotion=emotion
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with shot context
            raise AudioError(f"shot {shot.id}: voice synthesis failed: {exc}") from exc

        # A synthesiser must return the path it wrote. Accept None (wrote to
        # the path we gave it) but reject anything else loudly, because a
        # mistyped return turns into a baffling "does not exist" much later.
        target_path = path
        if isinstance(written, str):
            target_path = Path(written)
        elif written is not None:
            raise AudioError(
                f"shot {shot.id}: synthesiser returned {type(written).__name__}, "
                f"expected a path string. It should return the file it wrote."
            )

        info = probe(target_path)
        if info.seconds < MIN_NARRATION_SECONDS:
            raise AudioError(
                f"shot {shot.id}: rendered audio is {info.seconds:.3f}s, under the "
                f"{MIN_NARRATION_SECONDS}s floor. That is a failed render, not a "
                f"short line."
            )
        produced[shot.id] = ShotAudio(
            shot_id=shot.id,
            path=info.path,
            seconds=info.seconds,
            voice=voice,
            emotion=emotion,
        )
    return produced


def apply_measurements(
    film: Film, audio: dict[str, ShotAudio]
) -> Film:
    """Fold measured durations back into the film and re-cut for fit.

    This is the whole point of the stage: durations stop being estimates and
    start being facts, and any shot that no longer fits a generation window is
    split and re-chained.
    """
    missing = [s.id for s in film.shots if s.narration.strip() and s.id not in audio]
    if missing:
        raise AudioError(
            f"{len(missing)} shots have narration but no rendered audio: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    scenes: list[Scene] = []
    for scene in film.scenes:
        updated = tuple(
            s.measured(audio[s.id].seconds) if s.id in audio else s
            for s in scene.shots
        )
        scenes.append(Scene(
            id=scene.id, index=scene.index, summary=scene.summary,
            location=scene.location, time_of_day=scene.time_of_day, shots=updated,
        ))

    rebuilt = Film(
        id=film.id, title=film.title, language=film.language,
        target_minutes=film.target_minutes, characters=film.characters,
        scenes=tuple(scenes),
    )
    return rebuilt.split_oversized().resolve_continuity()


def timing_report(film: Film, audio: dict[str, ShotAudio]) -> str:
    """One line per shot plus totals, so drift is visible before rendering."""
    lines = [
        f"  {'shot':<12} {'audio':>7} {'video':>7} {'drift':>8}  frames",
        "  " + "-" * 60,
    ]
    total_drift = 0.0
    for shot in film.shots:
        a = audio.get(shot.id)
        if a is None:
            continue
        drift = shot.target_seconds() - a.seconds
        total_drift += drift
        lines.append(
            f"  {shot.id:<12} {a.seconds:>6.2f}s {shot.target_seconds():>6.2f}s "
            f"{drift * 1000:>+7.0f}ms {shot.target_frames():>7}"
        )
    lines.append("  " + "-" * 60)
    lines.append(f"  total drift: {total_drift * 1000:+.0f} ms")
    return "\n".join(lines)
