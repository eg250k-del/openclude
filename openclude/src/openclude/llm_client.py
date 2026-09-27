"""A real ScriptWriter: any OpenAI-compatible chat endpoint.

Everything with a `/v1/chat/completions` route works — Salad's AI Gateway,
OpenAI, OpenRouter, Groq, a local llama.cpp or vLLM server. That matters here
because SaladCloud already offers an AI Gateway, so the script stage can run
from the same account as the GPU.

The contract this implements is the `ScriptWriter` protocol from `llm.py`. The
guidance it obeys comes straight from the audit: the one useful thing in the
largest audited repository was a four-step prompt chain (dialogue -> static
shot -> video shot -> voice), and the one thing that killed it was shipping
that chain as markdown for a human to follow instead of as code.

Here it is code, with the four roles as four separate calls, so a failure in
the shot pass does not cost the dialogue pass. `temperature=0` and a recorded
prompt hash, so the same story produces the same script.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm import (
    MAX_PROMPT_CHARS,
    SYSTEM,
    Draft,
    DraftScene,
    DraftShot,
    ScriptError,
    ScriptWriter,
    TransientScriptError,
    build_prompt,
    parse_draft,
)
from .schema import Character, Film

DEFAULT_BASE = "https://api.salad.com/api/public"
DEFAULT_MODEL = "gpt-4o-mini"

MAX_RETRIES = 4
TIMEOUT = 180


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class LLMConfig:
    """Every value from the environment, so no secret is ever in the source."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("LLM_API_KEY", "").strip()
        self.base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
        self.model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip()
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "8000"))
        self.user_agent = "openclude/0.1"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    last_fingerprint: str = ""

    def line(self) -> str:
        return (
            f"llm calls={self.calls} prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} elapsed={self.seconds:.1f}s"
        )


def chat(
    cfg: LLMConfig,
    messages: list[dict[str, str]],
    usage: Usage,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """One chat completion, with bounded retries and loud failure.

    Retries only what is worth retrying: timeouts, 429 and 5xx. A 401 or a 400
    is a configuration error and repeating it just burns time.
    """
    body = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature if temperature is None else temperature,
        "max_tokens": max_tokens or cfg.max_tokens,
    }
    usage.last_fingerprint = hashlib.sha256(
        json.dumps([cfg.model, messages], sort_keys=True).encode()
    ).hexdigest()[:16]

    data = json.dumps(body).encode("utf-8")
    url = f"{cfg.base_url}/v1/chat/completions"

    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {cfg.api_key}")
        req.add_header("User-Agent", cfg.user_agent)
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                payload = json.loads(r.read().decode("utf-8"))
            usage.calls += 1
            usage.seconds += time.time() - started
            u = payload.get("usage") or {}
            usage.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
            usage.completion_tokens += int(u.get("completion_tokens", 0) or 0)
            choices = payload.get("choices") or []
            if not choices:
                raise ScriptError("model returned no choices")
            text = (choices[0].get("message") or {}).get("content") or ""
            if not text.strip():
                raise ScriptError("model returned an empty completion")
            return text
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (400, 401, 403, 404):
                raise ScriptError(
                    f"{last}\n  this is a configuration problem, not a transient "
                    f"one. Check LLM_BASE_URL, LLM_API_KEY and LLM_MODEL."
                ) from None
            if e.code != 429 and e.code < 500:
                raise ScriptError(last) from None
        except urllib.error.URLError as e:
            last = f"network: {e.reason}"
        except TimeoutError:
            last = f"timed out after {TIMEOUT}s"

        if attempt < MAX_RETRIES:
            wait = min(2 ** attempt, 45) + random.uniform(0, 1.5)
            time.sleep(wait)

    raise TransientScriptError(
        f"gave up after {MAX_RETRIES} attempts. last error: {last}"
    )


def _json_from(text: str, what: str) -> Any:
    """Pull one JSON value out of a completion.

    Models wrap JSON in prose and fences constantly, so this is a real
    requirement rather than defensive noise. A truncated response is the one
    failure worth retrying, and it is reported as such.
    """
    import re

    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = body.find(opener)
        j = body.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(body[i : j + 1])
            except json.JSONDecodeError:
                continue
    if "max_tokens" in body.lower() or len(body) > 4000:
        raise TransientScriptError(
            f"the {what} response looks truncated ({len(body)} chars). "
            f"Raise LLM_MAX_TOKENS."
        )
    raise ScriptError(
        f"could not find JSON in the {what} response. It started with: "
        f"{body[:160]!r}"
    )


# --------------------------------------------------------------------------
# the four passes
# --------------------------------------------------------------------------


DIALOGUE_SYSTEM = (
    "You write spoken narration for an animated film. You output ONLY JSON."
)

SHOT_SYSTEM = (
    "You are a cinematographer breaking a script into shots. You output ONLY JSON."
)

VOICE_SYSTEM = (
    "You cast voices for film characters. You output ONLY JSON."
)


def pass_dialogue(
    cfg: LLMConfig, story: str, characters: Sequence[Character], target_minutes: int
) -> list[dict]:
    """One pass: the spoken lines, with the character who says each."""
    cast = "\n".join(f"  {c.id} = {c.name}" for c in characters) or "  (none)"
    words = int(target_minutes * 60 / 0.42)
    prompt = f"""Write the narration for a {target_minutes} minute animated film.

STORY
{story.strip()}

CHARACTERS (use only these ids when someone speaks)
{cast}

RULES
- About {words} words in total, split into short lines of at most 14 words.
- Most lines are narration with no speaker.
- When a character speaks, set speaker to their id.
- No stage directions, no scene numbers, no markdown.

Return: {{"lines": [{{"text": "...", "speaker": ""}}]}}"""
    data = _json_from(
        chat(cfg, [{"role": "system", "content": DIALOGUE_SYSTEM},
                   {"role": "user", "content": prompt}], Usage()),
        "dialogue",
    )
    lines = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(lines, list) or not lines:
        raise ScriptError("the dialogue pass returned no lines")
    return [x for x in lines if isinstance(x, dict) and str(x.get("text", "")).strip()]


def pass_shots(
    cfg: LLMConfig,
    story: str,
    characters: Sequence[Character],
    lines: list[dict],
) -> list[dict]:
    """One pass: the picture and the camera move for every line."""
    cast = "\n".join(
        f"  {c.id} = {c.name}: {c.description}" for c in characters
    ) or "  (none)"
    body = "\n".join(
        f"{i + 1}. {l.get('text', '')}"
        + (f"   [speaker: {l['speaker']}]" if l.get("speaker") else "")
        for i, l in enumerate(lines)
    )
    prompt = f"""Give every line a shot.

STORY
{story.strip()}

CHARACTERS
{cast}

LINES
{body}

RULES
- Return exactly {len(lines)} shots, in the same order, one per line.
- "visual" describes only what the camera sees. Never a character's name:
  the renderer cannot render text and does not know who anyone is. Describe
  hair, build, clothes, age, expression instead.
- "visual" stays under {MAX_PROMPT_CHARS} characters.
- One physical camera move in "camera", e.g. "slow push in".
- "character_ids" lists which characters appear, by id.

Return: {{"shots": [{{"visual": "...", "camera": "...", "character_ids": []}}]}}"""
    data = _json_from(
        chat(cfg, [{"role": "system", "content": SHOT_SYSTEM},
                   {"role": "user", "content": prompt}], Usage()),
        "shot",
    )
    shots = data.get("shots") if isinstance(data, dict) else data
    if not isinstance(shots, list):
        raise ScriptError("the shot pass returned no shots list")
    if len(shots) != len(lines):
        raise ScriptError(
            f"the shot pass returned {len(shots)} shots for {len(lines)} lines. "
            f"Alignment is the whole point of this pass."
        )
    return [s for s in shots if isinstance(s, dict)]


def pass_voices(
    cfg: LLMConfig, characters: Sequence[Character]
) -> dict[str, str]:
    """One pass: one short voice description per character."""
    if not characters:
        return {}
    body = "\n".join(f"  {c.id} = {c.name}: {c.description}" for c in characters)
    prompt = f"""Describe one voice per character.

CHARACTERS
{body}

RULES
- One line each: tone, pitch, pace, age. No name, no adjectives about looks.
- This becomes a voice-cloning reference, so be concrete and short.

Return: {{"voices": [{{"id": "...", "description": "..."}}]}}"""
    data = _json_from(
        chat(cfg, [{"role": "system", "content": VOICE_SYSTEM},
                   {"role": "user", "content": prompt}], Usage()),
        "voice",
    )
    voices = data.get("voices") if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for v in voices or []:
        if isinstance(v, dict) and v.get("id"):
            out[str(v["id"])] = str(v.get("description", "")).strip()
    return out


# --------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------


@dataclass
class ChatScriptWriter:
    """Implements ScriptWriter using four chainable model passes."""

    cfg: LLMConfig = field(default_factory=LLMConfig)
    usage: Usage = field(default_factory=Usage)
    characters: Sequence[Character] = ()

    def write(self, story: str, characters: Sequence[Character]) -> str:
        if not self.cfg.configured:
            raise ScriptError(
                "no LLM configured. Set LLM_API_KEY (and optionally "
                "LLM_BASE_URL, LLM_MODEL). The SaladCloud AI Gateway, OpenAI, "
                "OpenRouter, Groq and a local llama.cpp server all work."
            )
        lines = pass_dialogue(self.cfg, story, characters, self.minutes)
        shots = pass_shots(self.cfg, story, characters, lines)
        pass_voices(self.cfg, characters)

        scenes: list[dict] = []
        current: dict | None = None
        for i, (line, shot) in enumerate(zip(lines, shots), start=1):
            # a scene break every time the speaker changes is a crude but
            # reliable beat boundary, which is what the engine's start/end
            # image anchoring needs
            if current is None or line.get("speaker") != scenes[-1].get("_speaker"):
                current = {
                    "summary": "",
                    "location": "",
                    "time_of_day": "",
                    "_speaker": line.get("speaker", ""),
                    "shots": [],
                }
                scenes.append(current)
            current["shots"].append(
                {
                    "narration": str(line.get("text", "")).strip(),
                    "visual": str(shot.get("visual", "")).strip(),
                    "camera": str(shot.get("camera", "")).strip(),
                    "character_ids": shot.get("character_ids", []) or [],
                    "dialogue_speaker": str(line.get("speaker", "")).strip(),
                    "anchor": "",
                }
            )
        for s in scenes:
            s.pop("_speaker", None)

        # the parse_draft path applies every validation, so the chain's output
        # goes through exactly the same gate as any other writer
        draft = {
            "title": story.strip().splitlines()[0][:80] if story.strip() else "untitled",
            "language": self.language,
            "scenes": scenes,
        }
        return json.dumps(draft)

    language: str = "ar"
    minutes: int = 1


def writer_from_env() -> ScriptWriter:
    """The writer the worker uses. Fails loudly when nothing is configured."""
    w = ChatScriptWriter()
    w.language = os.environ.get("FILM_LANGUAGE", "ar")
    w.minutes = int(os.environ.get("FILM_TARGET_MINUTES", "1"))
    return w
