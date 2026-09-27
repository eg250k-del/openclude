"""Real speech: an OpenAI-compatible TTS endpoint, and the engine's own voices.

Two implementations, because there are two genuinely different situations.

`OpenAICompatSynth` speaks over HTTP. Works with OpenAI, Groq, and anything
else exposing `/v1/audio/speech`. Good for the narration voice.

`EngineSynth` speaks through the GPU engine's in-process API, the same session
the video renderer uses. This is the one that matters for per-character voices:
the engine's Qwen3-TTS and IndexTTS2 clone up to three distinct speakers from
reference audio and take inline `[emotion]` tags, which no HTTP TTS in this
list does. It costs no extra GPU, because the model is already loaded next to
the video pipeline.

What is deliberately absent: `edge-tts`. One audited repo used it and its own
documentation admits the endpoint returns 0-byte files for certain inputs,
which is how a whole film comes out mute and looks like success. Nothing in
this module can produce a 0-byte file and pass it on.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .audio import SAMPLE_RATE, AudioError, Synthesiser
from .schema import Character, Shot

MAX_RETRIES = 4
TIMEOUT = 300

#: Below this, a "rendered" file is a failed render, not a short line.
MIN_SECONDS = 0.25


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class TTSConfig:
    def __init__(self) -> None:
        self.api_key = os.environ.get("TTS_API_KEY", "").strip()
        self.base_url = os.environ.get("TTS_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("TTS_MODEL", "").strip()
        self.voice = os.environ.get("TTS_VOICE", "alloy").strip()
        self.format = os.environ.get("TTS_FORMAT", "wav").strip()
        self.speed = os.environ.get("TTS_SPEED", "").strip()
        self.user_agent = "openclude/0.1"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass
class OpenAICompatSynth:
    """POST /v1/audio/speech. Returns a real wav, verified before it is used."""

    cfg: TTSConfig = field(default_factory=TTSConfig)
    calls: int = 0
    seconds: float = 0.0

    def speak(
        self, text: str, out_path: str | os.PathLike[str], voice: str = "", emotion: str = ""
    ) -> str:
        if not self.cfg.configured:
            raise AudioError(
                "no TTS configured. Set TTS_API_KEY, TTS_BASE_URL and TTS_MODEL, "
                "or use the engine's own voices (TTS_MODE=engine)."
            )
        if not text.strip():
            raise AudioError("refusing to synthesise an empty line")

        chosen = voice or self.cfg.voice
        if chosen in ("narrator", ""):
            chosen = self.cfg.voice

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "voice": chosen,
            "input": _with_emotion(text, emotion),
            "response_format": self.cfg.format,
        }
        if self.cfg.speed:
            payload["speed"] = float(self.cfg.speed)
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.cfg.base_url}/v1/audio/speech"

        last = ""
        for attempt in range(1, MAX_RETRIES + 1):
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {self.cfg.api_key}")
            req.add_header("User-Agent", self.cfg.user_agent)
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    blob = r.read()
                self.calls += 1
                return _write_audio(blob, out_path, self.cfg.format)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                last = f"HTTP {e.code}: {detail}"
                if e.code in (400, 401, 403, 404):
                    raise AudioError(
                        f"{last}\n  configuration problem, not transient. Check "
                        f"TTS_BASE_URL, TTS_API_KEY, TTS_MODEL and the voice name."
                    ) from None
            except urllib.error.URLError as e:
                last = f"network: {e.reason}"
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))

        raise AudioError(f"speech failed after {MAX_RETRIES} attempts: {last}")


def _with_emotion(text: str, emotion: str) -> str:
    """IndexTTS2 and OmniVoice read a leading bracketed tag as an instruction."""
    return f"[{emotion}] {text}" if emotion else text


def _write_audio(blob: bytes, out_path: str | os.PathLike[str], fmt: str) -> str:
    """Write and verify. A zero-byte or unparseable file never leaves here."""
    if not blob:
        raise AudioError("the TTS endpoint returned an empty body")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    if p.stat().st_size == 0:
        raise AudioError(f"wrote {p} and it is 0 bytes")

    if p.suffix.lower() != ".wav":
        return str(p)

    # a .wav that is not a RIFF file is worse than an error, because the
    # prober will fall through to ffprobe and report something unrelated
    try:
        with wave.open(str(p), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
    except (wave.Error, EOFError) as exc:
        p.unlink(missing_ok=True)
        raise AudioError(
            f"the TTS endpoint returned something that is not a valid wav ({exc}). "
            f"Check TTS_FORMAT; it must be 'wav' for a .wav path."
        ) from None
    seconds = frames / rate if rate else 0.0
    if seconds < MIN_SECONDS:
        p.unlink(missing_ok=True)
        raise AudioError(
            f"the TTS endpoint returned {seconds:.3f}s of audio, under the "
            f"{MIN_SECONDS}s floor. That is a failed render, not a short line."
        )
    return str(p)


# --------------------------------------------------------------------------
# the engine's own voices
# --------------------------------------------------------------------------


@dataclass
class EngineSynth:
    """Speaks through the GPU engine, in the same session as the video.

    The engine models declare their own multi-speaker support: Qwen3-TTS takes
    `Speaker 1:` / `Speaker 2:` lines with a reference clip per speaker, and
    IndexTTS2 takes `[emotion]` tags. Reference clips come from each
    character's `voice_reference` path, so a recurring character keeps one voice
    for the whole film.
    """

    session: Any
    model_type: str = "qwen3_tts_base"
    emotion: str = ""
    speaker_label: str = "Speaker 1"
    calls: int = 0
    seconds: float = 0.0

    def speak(
        self, text: str, out_path: str | os.PathLike[str], voice: str = "", emotion: str = ""
    ) -> str:
        if not text.strip():
            raise AudioError("refusing to synthesise an empty line")

        settings: dict[str, Any] = {
            "model_type": self.model_type,
            "prompt": text,
            "alt_prompt": emotion or self.emotion,
            "output_filename": str(Path(out_path).with_suffix("")),
        }
        if voice and voice not in ("narrator", "") and Path(voice).exists():
            # a per-character reference clip is what makes voices distinct
            settings["audio_guide"] = voice
            settings["audio_prompt_type"] = "A"

        started = time.time()
        try:
            result = self.session.run_task(settings)
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"engine speech failed: {exc}") from exc

        if not getattr(result, "success", False):
            errors = getattr(result, "errors", None) or []
            detail = "; ".join(getattr(e, "message", str(e)) for e in errors)
            raise AudioError(f"engine speech failed: {detail or 'no message'}")

        files = list(getattr(result, "generated_files", None) or [])
        if not files:
            raise AudioError("engine reported speech success but produced no file")

        produced = Path(files[0])
        target = Path(out_path)
        if produced.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(produced.read_bytes())
        if target.stat().st_size == 0:
            raise AudioError(f"engine produced a 0-byte file for {target}")

        self.calls += 1
        self.seconds += time.time() - started
        return str(target)

    def line(self) -> str:
        return f"engine-tts calls={self.calls} elapsed={self.seconds:.1f}s"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def synth_from_env(session: Any = None) -> Synthesiser:
    """Pick the speech backend from the environment.

    TTS_MODE=engine   use the GPU engine's own voices (per-character cloning)
    TTS_MODE=http     use an OpenAI-compatible /v1/audio/speech endpoint
    TTS_MODE=none     stay silent, which is a deliberate test mode
    """
    mode = os.environ.get("TTS_MODE", "").strip().lower()
    if not mode:
        mode = "engine" if session is not None else "http"

    if mode == "none":
        return _NullSynth()
    if mode == "engine":
        if session is None:
            raise AudioError(
                "TTS_MODE=engine needs the engine session. Start the engine "
                "before the audio stage, or set TTS_MODE=http."
            )
        return EngineSynth(
            session=session,
            model_type=os.environ.get("TTS_MODEL", "qwen3_tts_base"),
            emotion=os.environ.get("TTS_EMOTION", ""),
        )
    if mode == "http":
        return OpenAICompatSynth()
    raise AudioError(f"unknown TTS_MODE {mode!r}; use engine, http or none")


@dataclass
class _NullSynth:
    """Writes a silent-but-valid wav, so timing can be tested without a voice.

    Deliberately refuses to be used for a real film: `assert_real()` is called
    by the pipeline before assembly.
    """

    def speak(self, text, out_path, voice="", emotion="") -> str:
        seconds = max(1.0, len(text.split()) * 0.42)
        n = int(seconds * SAMPLE_RATE)
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(struct.pack("<h", 0) * n)
        return str(p)
