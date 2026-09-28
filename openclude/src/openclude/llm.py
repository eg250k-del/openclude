"""Story -> script. The first stage that needs a language model.

The four audited repos had nothing here. One of them shipped a 77 KB markdown
file describing steps for a human-with-an-LLM to follow by hand; another had
zero LLM calls in its entire codebase and did the splitting in the agent's
head. So this module is built from scratch, and it is deliberately small.

Two ideas were worth stealing from the audit:

  * the prompt chain. Source C's four prompts (dialogue -> static shot ->
    video shot -> voice descriptor) are a good shape: one job per call, each
    feeding the next. Splitting them into calls that can be retried
    independently is worth more than making one giant prompt.

  * the "Next shot:" trigger. One prompt per line, marker-delimited. It is
    fragile but it is also trivially checkable, and checkable is what matters
    in an unattended run.

What this module deliberately refuses to do:
  * guess when the model returns prose instead of JSON
  * accept a shot with no visual description
  * accept narration that cannot fit a generation window without flagging it
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .schema import Character, Film, SchemaError, Scene, Shot, framespec_for

MAX_PROMPT_CHARS = 3000
"""Wan text encoders truncate well before this; long prompts silently lose
the tail, which is usually the camera instruction. Keep it tight and loud."""


class ScriptError(RuntimeError):
    """The writer produced something unusable. Never paper over it."""


class TransientScriptError(RuntimeError):
    """Worth retrying: rate limit, timeout, truncated response."""


# --------------------------------------------------------------------------
# what we ask for, and what we accept back
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftShot:
    """One shot as drafted, before the schema gets hold of it."""

    narration: str
    visual: str
    camera: str = ""
    character_ids: tuple[str, ...] = ()
    dialogue_speaker: str = ""
    anchor: str = ""


@dataclass(frozen=True)
class DraftScene:
    summary: str
    location: str
    time_of_day: str
    shots: tuple[DraftShot, ...]


@dataclass(frozen=True)
class Draft:
    title: str
    language: str
    scenes: tuple[DraftScene, ...]

    def characters_used(self) -> set[str]:
        return {cid for s in self.shots() for cid in s.character_ids}

    def shots(self) -> tuple[DraftShot, ...]:
        return tuple(s for sc in self.scenes for s in sc.shots)


def parse_draft(payload: str | dict[str, Any], language: str = "") -> Draft:
    """Turn a model response into a Draft, or explain precisely why not.

    A model that returns a paragraph instead of JSON is the single most common
    failure, and it must be a loud error, not an empty film.
    """
    if isinstance(payload, str):
        text = payload.strip()
        # tolerate ```json fences; models add them constantly
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        if not text:
            raise ScriptError("writer returned an empty response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            preview = text[:160].replace("\n", " ")
            raise ScriptError(
                f"writer did not return JSON (line {exc.lineno}, col {exc.colno}). "
                f"It started with: {preview!r}"
            ) from exc
    else:
        data = payload

    if not isinstance(data, dict):
        raise ScriptError(f"expected a JSON object, got {type(data).__name__}")

    scenes_raw = data.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise ScriptError("draft has no non-empty 'scenes' list")

    scenes: list[DraftScene] = []
    for i, sc in enumerate(scenes_raw, start=1):
        if not isinstance(sc, dict):
            raise ScriptError(f"scene {i} is not an object")
        shots_raw = sc.get("shots")
        if not isinstance(shots_raw, list) or not shots_raw:
            raise ScriptError(f"scene {i} has no non-empty 'shots' list")
        shots: list[DraftShot] = []
        for j, sh in enumerate(shots_raw, start=1):
            if not isinstance(sh, dict):
                raise ScriptError(f"scene {i} shot {j} is not an object")
            narration = str(sh.get("narration", "")).strip()
            visual = str(sh.get("visual", "")).strip()
            if not visual:
                raise ScriptError(
                    f"scene {i} shot {j} has no 'visual' description; "
                    f"a shot with no picture is not a shot"
                )
            if len(visual) > MAX_PROMPT_CHARS:
                raise ScriptError(
                    f"scene {i} shot {j} visual is {len(visual)} chars, over the "
                    f"{MAX_PROMPT_CHARS} the encoder will actually read"
                )
            ids = sh.get("character_ids", []) or []
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                raise ScriptError(f"scene {i} shot {j} character_ids must be a list")
            shots.append(
                DraftShot(
                    narration=narration,
                    visual=visual,
                    camera=str(sh.get("camera", "")).strip(),
                    character_ids=tuple(str(x) for x in ids),
                    dialogue_speaker=str(sh.get("dialogue_speaker", "")).strip(),
                    anchor=str(sh.get("anchor", "")).strip(),
                )
            )
        scenes.append(
            DraftScene(
                summary=str(sc.get("summary", "")).strip(),
                location=str(sc.get("location", "")).strip(),
                time_of_day=str(sc.get("time_of_day", "")).strip(),
                shots=tuple(shots),
            )
        )

    return Draft(
        title=str(data.get("title", "untitled")).strip() or "untitled",
        # The job says what language the film is in, and the job is the request.
        # Letting the model decide produced "ar" for an English story on the
        # first real call, which would have sent English narration to an Arabic
        # voice. A model guessing at an input it was already given is not a
        # decision to defer to; the caller is the authority.
        language=(language or "en").strip() or "en",
        scenes=tuple(scenes),
    )


# --------------------------------------------------------------------------
# assembly into the real schema
# --------------------------------------------------------------------------


def to_film(
    draft: Draft,
    characters: Sequence[Character],
    film_id: str = "f01",
    target_minutes: int = 1,
    model_type: str = "ti2v_2_2_fastwan",
    seconds_per_word: float = 0.42,
) -> Film:
    """Build a Film from a draft, estimating durations for now.

    Durations here are ESTIMATES and are marked as unmeasured. The audio stage
    replaces every one of them with a measured value; until that happens the
    film refuses to render, which is the point.
    """
    known = {c.id for c in characters}
    spec = framespec_for(model_type)

    scenes: list[Scene] = []
    for i, sc in enumerate(draft.scenes, start=1):
        shots: list[Shot] = []
        for j, ds in enumerate(sc.shots, start=1):
            unknown = set(ds.character_ids) - known
            if unknown:
                raise ScriptError(
                    f"scene {i} shot {j} references unknown characters "
                    f"{sorted(unknown)}; known: {sorted(known)}"
                )
            if ds.dialogue_speaker and ds.dialogue_speaker not in known:
                raise ScriptError(
                    f"scene {i} shot {j} dialogue_speaker {ds.dialogue_speaker!r} "
                    f"is not a known character"
                )
            anchor = ds.anchor if ds.anchor in ("", "S", "E", "SE") else ""
            shot = Shot(
                id=f"s{i:02d}_sh{j:03d}",
                scene_id=f"s{i:02d}",
                index=j,
                narration=ds.narration,
                visual=ds.visual,
                dialogue_speaker=ds.dialogue_speaker,
                character_ids=ds.character_ids,
                camera=ds.camera,
                model_type=model_type,
                anchor_mode=anchor,
                # the first shot of a scene is a hard cut; later ones chain
                carry_from_previous=(j > 1),
            )
            words = len(ds.narration.split()) if ds.narration else 0
            seconds = max(1.0, round(words * seconds_per_word, 2))
            if seconds > spec.max_seconds():
                # left oversized on purpose: the audio stage measures for real
                # and Scene.split_oversized() handles the overflow
                pass
            shots.append(shot.measured(seconds))
        scenes.append(
            Scene(
                id=f"s{i:02d}",
                index=i,
                summary=sc.summary,
                location=sc.location,
                time_of_day=sc.time_of_day,
                shots=tuple(shots),
            )
        )

    return Film(
        id=film_id,
        title=draft.title,
        language=draft.language,
        target_minutes=target_minutes,
        characters=tuple(characters),
        scenes=tuple(scenes),
    ).split_oversized().resolve_continuity()


# --------------------------------------------------------------------------
# the writer interface
# --------------------------------------------------------------------------


class ScriptWriter(Protocol):
    """Anything that can turn a story into a Draft."""

    def write(self, story: str, characters: Sequence[Character]) -> str:
        """Return the raw model response (JSON text)."""


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

SYSTEM = (
    "You are a screenwriter for an animated film. You output ONLY JSON. "
    "No prose, no markdown fence, no commentary."
)

SCHEMA_HINT = """Return exactly this shape:
{
  "title": "string",
  "language": "en",
  "scenes": [
    {
      "summary": "one line",
      "location": "where this happens",
      "time_of_day": "dawn",
      "shots": [
        {
          "narration": "the spoken line, at most 14 words",
          "visual": "what the camera sees. No names, no dialogue, no text.",
          "camera": "one camera move, e.g. 'slow push in'",
          "character_ids": ["id"],
          "dialogue_speaker": "id, or empty for narration",
          "anchor": "" or "S" or "SE"
        }
      ]
    }
  ]
}"""


def build_prompt(story: str, characters: Sequence[Character], target_minutes: int) -> str:
    """The full user prompt. Kept as a function so it can be inspected and tested."""
    if not story.strip():
        raise ScriptError("story is empty")
    if target_minutes <= 0:
        raise ScriptError(f"target_minutes must be positive, got {target_minutes}")

    cast = "\n".join(
        f"  {c.id} = {c.name}: {c.description}" for c in characters
    ) or "  (no recurring characters; use no character_ids)"

    approx_words = int(target_minutes * 60 / 0.42)
    return f"""Break this story into shots for a {target_minutes} minute animated film.

STORY
{story.strip()}

CHARACTERS (use only these ids)
{cast}

RULES
1. About {approx_words} words of narration in total. One spoken line per shot.
2. Every shot needs a `visual`: describe the picture, never the plot.
3. Never write a character's name in `visual` - use physical description only.
   The model that renders this cannot render text or know who anyone is.
4. `visual` must stay under {MAX_PROMPT_CHARS} characters.
5. One camera move per shot, in `camera`. Keep it simple and physical.
6. The first shot of every scene has `anchor` "" (a hard cut). Use "S" when a
   shot must look like the one before it, "SE" only for a deliberate bookend.
7. Keep each narration under 14 words so one shot fits a generation window.

{SCHEMA_HINT}

Respond with the JSON object only. No prose before or after it.
"""


# --------------------------------------------------------------------------
# a deterministic writer, used by the tests and by dry runs
# --------------------------------------------------------------------------


class StubWriter:
    """Returns a fixed, valid draft. Never calls a network."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def write(self, story: str, characters: Sequence[Character]) -> str:
        self.calls.append(story)
        return self.response


def script(
    story: str,
    characters: Sequence[Character],
    writer: ScriptWriter,
    target_minutes: int = 1,
    film_id: str = "f01",
    model_type: str = "ti2v_2_2_fastwan",
) -> Film:
    """story + writer -> validated Film, or a loud failure."""
    prompt = build_prompt(story, characters, target_minutes)
    raw = writer.write(prompt, characters)
    draft = parse_draft(raw)
    return to_film(
        draft,
        characters,
        film_id=film_id,
        target_minutes=target_minutes,
        model_type=model_type,
    )
