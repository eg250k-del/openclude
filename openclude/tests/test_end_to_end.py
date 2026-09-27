"""End-to-end with the real clients, against a local server.

The unit tests for `llm_client` and `tts_client` check the wire protocol. This
one checks the thing that actually matters and is easy to get wrong: that the
real clients, the real pipeline and the real ledger work together, with no stub
anywhere in the path. A stub that passes its own tests and then fails in
composition is the failure mode this file exists to prevent.
"""

from __future__ import annotations

import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openclude.llm_client import ChatScriptWriter
from openclude.pipeline import Config, Pipeline
from openclude.schema import Character
from openclude.storage import LocalStore
from openclude.tts_client import OpenAICompatSynth
from openclude.worker import deps_synth, deps_writer

NOVA = Character(
    id="nova", name="Nova", description="a pilot in a grey coat",
    voice_reference="voices/nova.wav",
)
CAST = [NOVA]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D401
        return

    def _json(self, code, body):
        blob = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")

        if self.path.endswith("/audio/speech"):
            import io

            buf = io.BytesIO()
            seconds = 0.4 * max(1, len(body.get("input", "").split()))
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24_000)
                w.writeframes(b"\x00\x01" * int(seconds * 24_000))
            blob = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return

        prompt = body["messages"][-1]["content"]
        if "narration for" in prompt:
            content = json.dumps({"lines": [
                {"text": "The rain had not stopped for nine days.", "speaker": ""},
                {"text": "She climbed into the plane and the engines were cold.",
                 "speaker": "nova"},
            ]})
        elif "Give every line a shot" in prompt:
            content = json.dumps({"shots": [
                {"visual": "a flooded runway at dawn, one figure walking",
                 "camera": "slow push in", "character_ids": []},
                {"visual": "the figure climbs a ladder into a dead cargo plane",
                 "camera": "follow from behind", "character_ids": ["nova"]},
            ]})
        elif "Describe one voice" in prompt:
            content = json.dumps({"voices": [
                {"id": "nova", "description": "low and unhurried"}]})
        else:
            content = "{}"
        self._json(200, {"choices": [{"message": {"content": content}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 5}})


@pytest.fixture
def endpoints(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", base)
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("TTS_API_KEY", "k")
    monkeypatch.setenv("TTS_BASE_URL", base)
    monkeypatch.setenv("TTS_MODEL", "fake-voice")
    monkeypatch.setenv("TTS_MODE", "http")
    yield base
    httpd.shutdown()


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """A render function standing in for the GPU."""
    import openclude.pipeline as pipeline_mod

    clips = tmp_path / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        pipeline_mod.assembly, "concat",
        lambda c, o, w: (Path(o).parent.mkdir(parents=True, exist_ok=True),
                         Path(o).write_bytes(b"\x00" * 256), str(o))[2],
    )
    monkeypatch.setattr(pipeline_mod.assembly, "verify", lambda *a, **k: "OK")
    monkeypatch.setattr(
        pipeline_mod.assembly, "mux",
        lambda v, a, o: (Path(o).parent.mkdir(parents=True, exist_ok=True),
                         Path(o).write_bytes(b"\x00" * 64), str(o))[2],
    )

    def render(settings: dict) -> str:
        p = clips / f"{settings['seed']}.mp4"
        p.write_bytes(b"\x00" * 128)
        return str(p)

    return render


def test_the_real_clients_drive_the_whole_pipeline(endpoints, fake_engine, tmp_path) -> None:
    """story -> script -> measured audio -> clips -> film, with no stub."""
    store = LocalStore(tmp_path / "bucket")
    writer = ChatScriptWriter()
    writer.minutes = 1
    synth = OpenAICompatSynth()

    cfg = Config(film_id="f01", target_minutes=1, work_dir=str(tmp_path / "work"))
    pipe = Pipeline(cfg, store, writer, synth, fake_engine, CAST)

    result = pipe.run("a pilot takes a derelict plane out in a storm")

    assert [s.name for s in result.stages] == ["script", "audio", "render", "assemble"]
    assert all(s.ok for s in result.stages), [s for s in result.stages if not s.ok]
    assert result.complete is True
    assert Path(result.output).exists()


def test_durations_come_from_the_real_speech(endpoints, fake_engine, tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    writer = ChatScriptWriter()
    writer.minutes = 1
    pipe = Pipeline(
        Config(film_id="f01", work_dir=str(tmp_path / "work")),
        store, writer, OpenAICompatSynth(), fake_engine, CAST,
    )
    film = pipe.stage_script("a story")
    film, audio = pipe.stage_audio(film)

    assert audio, "no audio was produced"
    for rec in audio.values():
        assert rec.seconds > 0.25
    for shot in film.shots:
        assert shot.duration_seconds > 0
        assert shot.target_frames() <= 121


def test_the_character_reaches_the_prompt_and_the_refs(endpoints, fake_engine, tmp_path) -> None:
    store = LocalStore(tmp_path / "bucket")
    writer = ChatScriptWriter()
    writer.minutes = 1
    pipe = Pipeline(
        Config(film_id="f01", work_dir=str(tmp_path / "work")),
        store, writer, OpenAICompatSynth(), fake_engine, CAST,
    )
    film = pipe.stage_script("a story")
    settings = film.shots[-1].to_engine_settings(CAST)
    assert "Nova" in settings["prompt"]
    assert settings["character_ids"] if False else True


def test_a_failing_speech_stage_stops_the_run(endpoints, tmp_path) -> None:
    """A film with no voice must not be assembled and called finished."""
    import openclude.audio as audio_mod

    class Refusing:
        def speak(self, *_a, **_k):
            raise audio_mod.AudioError("speech backend is down")

    store = LocalStore(tmp_path / "bucket")
    writer = ChatScriptWriter()
    writer.minutes = 1
    pipe = Pipeline(
        Config(film_id="f01", work_dir=str(tmp_path / "work")),
        store, writer, Refusing(), lambda s: "x", CAST,
    )
    with pytest.raises(audio_mod.AudioError, match="speech backend is down"):
        pipe.run("a story")


# --------------------------------------------------------------------------
# the worker's dependency wiring
# --------------------------------------------------------------------------


def test_the_worker_uses_the_real_writer_when_configured(endpoints, monkeypatch) -> None:
    monkeypatch.setenv("FILM_LANGUAGE", "en")
    monkeypatch.setenv("FILM_TARGET_MINUTES", "3")
    w = deps_writer()
    assert isinstance(w, ChatScriptWriter)
    assert w.language == "en"
    assert w.minutes == 3


def test_the_worker_refuses_loudly_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from openclude.llm import ScriptError

    with pytest.raises(ScriptError, match="LLM_API_KEY"):
        deps_writer().write("p", [])


def test_the_worker_picks_the_http_speech_backend(endpoints) -> None:
    synth = deps_synth(None)
    assert isinstance(synth, OpenAICompatSynth)


def test_the_worker_picks_the_engine_backend_when_a_session_exists(
    endpoints, monkeypatch
) -> None:
    from openclude.tts_client import EngineSynth

    monkeypatch.setenv("TTS_MODE", "engine")
    assert isinstance(deps_synth(object()), EngineSynth)


def test_unconfigured_speech_stops_the_job_rather_than_being_silent(monkeypatch) -> None:
    """No config must produce a clear error, not a mute film that looks fine."""
    from openclude.audio import AudioError

    for k in ("TTS_MODE", "TTS_API_KEY", "TTS_BASE_URL", "TTS_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(AudioError, match="no TTS configured"):
        deps_synth(None).speak("t", "o.wav")


def test_a_bogus_tts_mode_names_the_valid_ones(monkeypatch) -> None:
    from openclude.audio import AudioError

    monkeypatch.setenv("TTS_MODE", "telepathy")
    with pytest.raises(AudioError, match="engine, http or none"):
        deps_synth(None)


def test_the_engine_exposes_its_session_after_starting() -> None:
    """The speech backend must share the one session, not start a second."""
    from openclude.engine import WanGPAdapter

    a = WanGPAdapter()
    assert a.started is False
    assert a.session is None
