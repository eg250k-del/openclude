"""Tests for the real LLM and TTS clients.

Both are exercised against a local HTTP server that speaks the real wire
protocol, so the assertions are about the request and response handling, not
about a mock's bookkeeping. Nothing here touches the network.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openclude import llm_client, tts_client
from openclude.audio import probe
from openclude.llm import ScriptError, TransientScriptError
from openclude.llm_client import LLMConfig, Usage, chat, writer_from_env
from openclude.tts_client import TTSConfig, OpenAICompatSynth, synth_from_env
from openclude.audio import AudioError
from openclude.schema import Character

NOVA = Character(id="nova", name="Nova", description="a pilot")
CAST = [NOVA]


# --------------------------------------------------------------------------
# a local server that speaks the real protocol
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    script: list[dict] = []
    fail_times = 0
    seen: list[dict] = []

    def log_message(self, *a):  # noqa: D401
        return

    def _json(self, code: int, body: dict) -> None:
        blob = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        Handler.seen.append({"path": self.path, "body": body,
                             "auth": self.headers.get("Authorization")})

        if Handler.fail_times > 0:
            Handler.fail_times -= 1
            self._json(429, {"error": "rate limited"})
            return

        if self.path.endswith("/audio/speech"):
            model = body.get("model", "")
            if model == "broken-wav":
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", "12")
                self.end_headers()
                self.wfile.write(b"not a riff!!")
                return
            if model == "silent":
                blob = _wav(0.01)
            elif model == "empty":
                blob = b""
            else:
                blob = _wav(2.0 + len(body.get("input", "")) * 0.001)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return

        prompt = body["messages"][-1]["content"]
        self._json(200, {
            "choices": [{"message": {"content": _answer(prompt)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        })


def _wav(seconds: float) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes(b"\x00\x01" * int(seconds * 24_000))
    return buf.getvalue()


def _answer(prompt: str) -> str:
    """Reply in the shape each pass asked for."""
    if "narration for" in prompt:
        return json.dumps({"lines": [
            {"text": "The rain had not stopped for nine days.", "speaker": ""},
            {"text": "She climbed into the plane.", "speaker": "nova"},
            {"text": "The engines were cold.", "speaker": ""},
        ]})
    if "Give every line a shot" in prompt:
        return json.dumps({"shots": [
            {"visual": "a flooded runway at dawn", "camera": "slow push in",
             "character_ids": []},
            {"visual": "the figure climbs a ladder", "camera": "follow behind",
             "character_ids": ["nova"]},
            {"visual": "close on gloved hands", "camera": "close up",
             "character_ids": ["nova"]},
        ]})
    if "Describe one voice" in prompt:
        return json.dumps({"voices": [
            {"id": "nova", "description": "low, steady, unhurried"}]})
    return "```json\n" + json.dumps({"ok": True}) + "\n```"


@pytest.fixture
def server():
    Handler.script = []
    Handler.fail_times = 0
    Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def llm(server, monkeypatch) -> LLMConfig:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", server)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    return LLMConfig()


@pytest.fixture
def tts(server, monkeypatch) -> TTSConfig:
    monkeypatch.setenv("TTS_API_KEY", "k")
    monkeypatch.setenv("TTS_BASE_URL", server)
    monkeypatch.setenv("TTS_MODEL", "test-voice")
    return TTSConfig()


# --------------------------------------------------------------------------
# LLM transport
# --------------------------------------------------------------------------


def test_a_completion_comes_back(llm) -> None:
    usage = Usage()
    out = chat(llm, [{"role": "user", "content": "hello"}], usage)
    assert "ok" in out
    assert usage.calls == 1
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20


def test_the_bearer_token_is_sent(llm) -> None:
    chat(llm, [{"role": "user", "content": "hi"}], Usage())
    assert Handler.seen[-1]["auth"] == "Bearer k"


def test_the_request_goes_to_the_chat_route(llm) -> None:
    chat(llm, [{"role": "user", "content": "hi"}], Usage())
    assert Handler.seen[-1]["path"].endswith("/v1/chat/completions")


def test_temperature_defaults_to_zero(llm, monkeypatch) -> None:
    """The same story must produce the same script."""
    chat(llm, [{"role": "user", "content": "hi"}], Usage())
    assert Handler.seen[-1]["body"]["temperature"] == 0


def test_a_rate_limit_is_retried(server, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", server)
    Handler.fail_times = 2
    usage = Usage()
    out = chat(LLMConfig(), [{"role": "user", "content": "hi"}], usage)
    assert "ok" in out
    assert usage.calls == 1


def test_a_401_stops_immediately(server, monkeypatch) -> None:
    """A bad key is a configuration error; retrying it wastes minutes.

    The handler is swapped with monkeypatch rather than assigned by hand: a bare
    `del Handler.do_POST` removes the method from the class permanently, and
    every later test in the file then gets a 501 it dutifully retries, which is
    how a one-second test file turned into a four-minute one.
    """
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", server)

    def refuse(self):
        blob = b'{"error":"nope"}'
        self.send_response(401)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    monkeypatch.setattr(Handler, "do_POST", refuse)
    with pytest.raises(ScriptError, match="configuration problem"):
        chat(LLMConfig(), [{"role": "user", "content": "hi"}], Usage())


def test_an_unconfigured_client_says_what_to_set(llm, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    w = llm_client.ChatScriptWriter()
    with pytest.raises(ScriptError, match="LLM_API_KEY"):
        w.write("a story", CAST)


def test_the_fingerprint_changes_with_the_prompt(llm) -> None:
    a, b = Usage(), Usage()
    chat(llm, [{"role": "user", "content": "one"}], a)
    chat(llm, [{"role": "user", "content": "two"}], b)
    assert a.last_fingerprint != b.last_fingerprint


# --------------------------------------------------------------------------
# the four passes
# --------------------------------------------------------------------------


def test_the_dialogue_pass_returns_lines(llm) -> None:
    lines = llm_client.pass_dialogue(llm, "a pilot", CAST, 1)
    assert len(lines) == 3
    assert lines[1]["speaker"] == "nova"


def test_the_shot_pass_must_align_with_the_lines(llm) -> None:
    """A misaligned shot pass silently corrupts the whole film."""
    body = "\n".join(f"{i + 1}. line {i}" for i in range(3))
    with pytest.raises(ScriptError, match="Alignment"):
        llm_client.pass_shots(llm, "s", CAST, [{"text": f"line {i}"} for i in range(4)])


def test_the_voice_pass_returns_a_description_per_character(llm) -> None:
    voices = llm_client.pass_voices(llm, CAST)
    assert voices["nova"].startswith("low, steady")


def test_a_fenced_response_is_parsed(llm) -> None:
    assert llm_client._json_from('```json\n{"a":1}\n```', "test") == {"a": 1}


def test_a_response_with_prose_around_json_is_parsed(llm) -> None:
    raw = 'Sure! Here you go:\n{"a": 1}\nHope that helps.'
    assert llm_client._json_from(raw, "test") == {"a": 1}


def test_a_truncated_response_is_reported_as_retryable() -> None:
    with pytest.raises(TransientScriptError, match="truncated"):
        llm_client._json_from('{"scenes": [{"shots": [{"narration": "' + "x" * 5000,
                          "shot")


def test_prose_with_no_json_is_a_clear_error() -> None:
    with pytest.raises(ScriptError, match="could not find JSON"):
        llm_client._json_from("I would love to help with that!", "dialogue")


# --------------------------------------------------------------------------
# the writer end to end
# --------------------------------------------------------------------------


def test_the_writer_produces_a_parsable_draft(llm) -> None:
    from openclude.llm import parse_draft

    w = llm_client.ChatScriptWriter(cfg=llm)
    w.minutes = 1
    draft = parse_draft(w.write("a pilot takes a plane out in a storm", CAST))
    assert draft.title
    assert len(draft.shots()) == 3
    assert draft.shots()[1].character_ids == ("nova",)
    assert draft.shots()[1].dialogue_speaker == "nova"


def test_the_writer_runs_four_passes(llm) -> None:
    w = llm_client.ChatScriptWriter(cfg=llm)
    w.write("a story", CAST)
    # dialogue, shots, voices
    assert len(Handler.seen) == 3


def test_the_writer_keeps_a_speaker_change_as_a_scene_break(llm) -> None:
    from openclude.llm import parse_draft

    w = llm_client.ChatScriptWriter(cfg=llm)
    draft = parse_draft(w.write("a story", CAST))
    # line 2 has a speaker, lines 1 and 3 do not -> at least two scenes
    assert len(draft.scenes) >= 2


def test_the_usage_line_is_readable(llm) -> None:
    w = llm_client.ChatScriptWriter(cfg=llm)
    w.write("a story", CAST)
    assert "llm calls=" in w.usage.line()


def test_writer_from_env_reads_the_language_and_length(monkeypatch) -> None:
    monkeypatch.setenv("FILM_LANGUAGE", "en")
    monkeypatch.setenv("FILM_TARGET_MINUTES", "20")
    w = writer_from_env()
    assert w.language == "en"
    assert w.minutes == 20


# --------------------------------------------------------------------------
# TTS over HTTP
# --------------------------------------------------------------------------


def test_speech_is_written_and_measured(tts, tmp_path) -> None:
    out = tmp_path / "a.wav"
    synth = OpenAICompatSynth(cfg=tts)
    path = synth.speak("hello there", out)
    assert Path(path).stat().st_size > 0
    assert probe(path).seconds >= 0.25


def test_the_voice_is_sent(tts, tmp_path) -> None:
    OpenAICompatSynth(cfg=tts).speak("hi", tmp_path / "a.wav", voice="echo")
    assert Handler.seen[-1]["body"]["voice"] == "echo"


def test_narrator_falls_back_to_the_default_voice(tts, tmp_path) -> None:
    OpenAICompatSynth(cfg=tts).speak("hi", tmp_path / "a.wav", voice="narrator")
    assert Handler.seen[-1]["body"]["voice"] == tts.voice


def test_an_emotion_becomes_a_leading_tag(tts, tmp_path) -> None:
    OpenAICompatSynth(cfg=tts).speak("I am afraid", tmp_path / "a.wav", emotion="fear")
    assert Handler.seen[-1]["body"]["input"] == "[fear] I am afraid"


def test_a_wav_that_is_not_a_wav_is_refused(tts, tmp_path) -> None:
    tts.model = "broken-wav"
    with pytest.raises(AudioError, match="not a valid wav"):
        OpenAICompatSynth(cfg=tts).speak("hi", tmp_path / "a.wav")


def test_an_empty_body_is_refused(tts, tmp_path) -> None:
    tts.model = "empty"
    with pytest.raises(AudioError, match="empty body"):
        OpenAICompatSynth(cfg=tts).speak("hi", tmp_path / "a.wav")


def test_a_sub_second_render_is_refused(tts, tmp_path) -> None:
    tts.model = "silent"
    with pytest.raises(AudioError, match="under the"):
        OpenAICompatSynth(cfg=tts).speak("hi", tmp_path / "a.wav")


def test_a_failed_render_leaves_no_file_behind(tts, tmp_path) -> None:
    """A 0-byte wav on disk looks like success to everything downstream."""
    tts.model = "silent"
    out = tmp_path / "a.wav"
    with pytest.raises(AudioError):
        OpenAICompatSynth(cfg=tts).speak("hi", out)
    assert not out.exists()


def test_an_empty_line_is_refused(tts, tmp_path) -> None:
    with pytest.raises(AudioError, match="empty line"):
        OpenAICompatSynth(cfg=tts).speak("   ", tmp_path / "a.wav")


def test_an_unconfigured_tts_says_what_to_set(monkeypatch, tmp_path) -> None:
    for k in ("TTS_API_KEY", "TTS_BASE_URL", "TTS_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(AudioError, match="TTS_API_KEY"):
        OpenAICompatSynth().speak("hi", tmp_path / "a.wav")


# --------------------------------------------------------------------------
# the engine's own voices
# --------------------------------------------------------------------------


class FakeEngineSession:
    """Stands in for the engine's session object.

    `produce_nothing` is a separate flag rather than a mutation of `files`,
    because the run_task stub always writes a real file; clearing the list
    afterwards changes the object but not what the call returns, which is how
    this test once passed without testing anything.
    """

    def __init__(self, ok: bool = True, produce_nothing: bool = False) -> None:
        self.ok = ok
        self.produce_nothing = produce_nothing
        self.calls: list[dict] = []

    def run_task(self, settings: dict):
        self.calls.append(settings)
        if not self.ok:
            return type("R", (), {
                "success": False, "generated_files": [],
                "errors": [type("E", (), {"message": "model missing"})()],
            })()
        if self.produce_nothing:
            return type("R", (), {
                "success": True, "generated_files": [], "errors": [],
            })()
        path = Path(settings["output_filename"]).with_suffix(".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_wav(2.0))
        return type("R", (), {
            "success": True, "generated_files": [str(path)], "errors": [],
        })()


def test_the_engine_synth_speaks_through_the_session(tmp_path) -> None:
    session = FakeEngineSession()
    synth = tts_client.EngineSynth(session=session)
    out = synth.speak("hello", tmp_path / "a.wav")
    assert Path(out).exists()
    assert session.calls[0]["model_type"] == "qwen3_tts_base"


def test_the_engine_synth_attaches_a_character_voice(tmp_path) -> None:
    ref = tmp_path / "nova.wav"
    ref.write_bytes(_wav(1.0))
    session = FakeEngineSession()
    tts_client.EngineSynth(session=session).speak("hi", tmp_path / "a.wav", voice=str(ref))
    assert session.calls[0]["audio_guide"] == str(ref)
    assert session.calls[0]["audio_prompt_type"] == "A"


def test_the_engine_synth_ignores_a_voice_that_is_not_a_file(tmp_path) -> None:
    session = FakeEngineSession()
    tts_client.EngineSynth(session=session).speak("hi", tmp_path / "a.wav", voice="narrator")
    assert "audio_guide" not in session.calls[0]


def test_a_failing_engine_raises_with_its_message(tmp_path) -> None:
    with pytest.raises(AudioError, match="model missing"):
        tts_client.EngineSynth(session=FakeEngineSession(ok=False)).speak(
            "hi", tmp_path / "a.wav"
        )


def test_an_engine_that_produces_nothing_is_not_success(tmp_path) -> None:
    """`success` with no file is the failure mode that leaves a hole in a film."""
    with pytest.raises(AudioError, match="no file"):
        tts_client.EngineSynth(session=FakeEngineSession(produce_nothing=True)).speak(
            "hi", tmp_path / "a.wav"
        )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_http_mode_is_chosen_without_a_session(monkeypatch) -> None:
    monkeypatch.setenv("TTS_MODE", "http")
    assert isinstance(synth_from_env(), OpenAICompatSynth)


def test_engine_mode_needs_a_session(monkeypatch) -> None:
    monkeypatch.setenv("TTS_MODE", "engine")
    with pytest.raises(AudioError, match="needs the engine session"):
        synth_from_env(None)


def test_engine_mode_is_used_when_a_session_exists(monkeypatch) -> None:
    monkeypatch.setenv("TTS_MODE", "engine")
    assert isinstance(synth_from_env(FakeEngineSession()), tts_client.EngineSynth)


def test_an_unknown_mode_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("TTS_MODE", "telepathy")
    with pytest.raises(AudioError, match="unknown TTS_MODE"):
        synth_from_env()


def test_none_mode_writes_a_valid_silent_wav(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TTS_MODE", "none")
    out = synth_from_env().speak("one two three four", tmp_path / "a.wav")
    assert probe(out).seconds == pytest.approx(1.68, abs=0.05)
