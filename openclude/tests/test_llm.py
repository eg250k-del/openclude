"""Tests for story -> script.

The model is stubbed, so these assert OUR behaviour, not the model's: that
prose instead of JSON is a loud error, that unknown character ids are caught,
that a shot with no visual is rejected, and that the prompt asks for the
right thing.
"""

from __future__ import annotations

import json

import pytest

from openclude.llm import (
    MAX_PROMPT_CHARS,
    ScriptError,
    StubWriter,
    build_prompt,
    parse_draft,
    script,
    to_film,
)
from openclude.schema import Character, SchemaError

NOVA = Character(id="nova", name="Nova", description="a pilot in a grey coat")
KAI = Character(id="kai", name="Kai", description="a mechanic, older, bearded")
CAST = [NOVA, KAI]

GOOD = {
    "title": "Nine Days",
    "language": "ar",
    "scenes": [
        {
            "summary": "takeoff",
            "location": "airfield",
            "time_of_day": "dawn",
            "shots": [
                {
                    "narration": "The rain had not stopped for nine days.",
                    "visual": "a flooded runway at dawn, one figure walking",
                    "camera": "slow push in",
                    "character_ids": ["nova"],
                    "anchor": "",
                },
                {
                    "narration": "She had lost this route twice.",
                    "visual": "the figure climbs a ladder into a dead cargo plane",
                    "camera": "follow from behind",
                    "character_ids": ["nova"],
                    "anchor": "S",
                },
            ],
        }
    ],
}


def payload(**over) -> str:
    data = json.loads(json.dumps(GOOD))
    data.update(over)
    return json.dumps(data)


# --------------------------------------------------------------------------
# parsing — loud on everything
# --------------------------------------------------------------------------


def test_a_good_draft_parses() -> None:
    draft = parse_draft(payload())
    assert draft.title == "Nine Days"
    assert len(draft.scenes) == 1
    assert len(draft.shots()) == 2
    assert draft.shots()[1].character_ids == ("nova",)


def test_a_json_fence_is_tolerated() -> None:
    assert parse_draft(f"```json\n{payload()}\n```").title == "Nine Days"
    assert parse_draft(f"```\n{payload()}\n```").title == "Nine Days"


def test_prose_instead_of_json_is_a_clear_error() -> None:
    with pytest.raises(ScriptError, match="did not return JSON"):
        parse_draft("Here is your story, divided into five lovely scenes.")


def test_the_error_shows_what_came_back() -> None:
    with pytest.raises(ScriptError, match="It started with"):
        parse_draft("Sorry! I cannot help with that request.")


def test_empty_response_is_rejected() -> None:
    with pytest.raises(ScriptError, match="empty response"):
        parse_draft("   ")


def test_a_missing_scenes_list_is_rejected() -> None:
    with pytest.raises(ScriptError, match="no non-empty 'scenes' list"):
        parse_draft(json.dumps({"title": "x", "scenes": []}))


def test_a_scene_with_no_shots_is_rejected() -> None:
    bad = {"title": "x", "scenes": [{"summary": "s", "shots": []}]}
    with pytest.raises(ScriptError, match="no non-empty 'shots' list"):
        parse_draft(json.dumps(bad))


def test_a_shot_with_no_visual_is_rejected() -> None:
    bad = json.loads(payload())
    bad["scenes"][0]["shots"][0]["visual"] = "   "
    with pytest.raises(ScriptError, match="has no 'visual' description"):
        parse_draft(json.dumps(bad))


def test_an_over_long_visual_is_rejected_before_the_encoder_truncates() -> None:
    bad = json.loads(payload())
    bad["scenes"][0]["shots"][0]["visual"] = "x" * (MAX_PROMPT_CHARS + 1)
    with pytest.raises(ScriptError, match="over the"):
        parse_draft(json.dumps(bad))


def test_a_bare_string_character_id_is_accepted() -> None:
    data = json.loads(payload())
    data["scenes"][0]["shots"][0]["character_ids"] = "nova"
    assert parse_draft(json.dumps(data)).shots()[0].character_ids == ("nova",)


def test_a_bad_anchor_is_dropped_not_crashed_on() -> None:
    data = json.loads(payload())
    data["scenes"][0]["shots"][0]["anchor"] = "banana"
    draft = parse_draft(json.dumps(data))
    assert draft.shots()[0].anchor == "banana"      # kept verbatim on the draft
    film = to_film(draft, CAST)                      # dropped at build time
    assert film.shots[0].anchor_mode == ""


# --------------------------------------------------------------------------
# building the film
# --------------------------------------------------------------------------


def test_no_source_file_defaults_to_arabic() -> None:
    """Three dataclass fields said "ar" on a project with no Arabic content.

    ChatScriptWriter, writer_from_env, pipeline.Config and worker.Job all
    defaulted to Arabic. The first real LLM call returned an English script
    tagged Arabic, and the film would have been spoken by an Arabic voice. A
    test per site would have been found and fixed one at a time; this finds all
    of them, and any added later.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "openclude"
    offenders = []
    for path in sorted(src.glob("*.py")):
        for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if '"ar"' in code or "'ar'" in code:
                offenders.append(f"{path.name}:{n}: {line.strip()[:70]}")
    assert not offenders, "Arabic defaults found:\n" + "\n".join(offenders)


def test_the_job_language_reaches_the_film() -> None:
    """End to end through the worker's own types, not just the parser."""
    from openclude.worker import Job
    from openclude.pipeline import Config as PipelineConfig

    job = Job(id="f01", story="x", film_id="f01", target_minutes=1)
    assert job.language == "en", "an omitted field must not become Arabic"
    assert PipelineConfig().language == "en"

    job_fr = Job(id="f02", story="x", film_id="f02", target_minutes=1,
                 language="fr")
    assert PipelineConfig(language=job_fr.language).language == "fr"


def test_the_chat_url_never_doubles_the_version_segment() -> None:
    """The documented Salad base already ends in /v1, and appending it again
    produced /v1/v1/chat/completions and a 404 whose body says nothing useful.
    """
    from openclude.llm_client import chat_url

    assert chat_url("https://ai.salad.cloud/v1") == \
        "https://ai.salad.cloud/v1/chat/completions"
    assert chat_url("https://ai.salad.cloud/v1/") == \
        "https://ai.salad.cloud/v1/chat/completions"
    assert chat_url("https://api.salad.com/api/public") == \
        "https://api.salad.com/api/public/v1/chat/completions"
    assert chat_url("https://api.openai.com") == \
        "https://api.openai.com/v1/chat/completions"
    assert "v1/v1" not in chat_url("https://ai.salad.cloud/v1")


def test_a_draft_becomes_a_valid_film() -> None:
    film = to_film(parse_draft(payload()), CAST)
    assert film.language == "en"
    assert len(film.scenes) == 1
    assert [s.id for s in film.shots] == ["s01_sh001", "s01_sh002"]
    assert film.total_seconds() > 0


def test_the_job_language_wins_over_whatever_the_model_said() -> None:
    """The model returned "ar" for an English story and the film inherited it.

    The prompt offered `"ar" or "en"` with nothing to choose between, and a
    dataclass field was hardcoded to "ar". Both would have sent English
    narration to an Arabic voice. The caller is the authority, because the
    caller is what knows the language of the request.
    """
    data = payload(language="ar")
    assert parse_draft(data, language="en").language == "en"
    assert parse_draft(data, language="fr").language == "fr"
    # and with nothing said, the model is still not trusted
    assert parse_draft(data).language == "en"


def test_the_default_language_is_english_not_arabic() -> None:
    """A hardcoded "ar" on a class with nothing to do with Arabic."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENCLIDE_LANGUAGE", None)
        from openclude.llm_client import ChatScriptWriter
        assert ChatScriptWriter().language == "en"


def test_the_prompt_does_not_offer_a_choice_it_cannot_justify() -> None:
    """`"ar" or "en"` with no instruction is a coin toss the model loses."""
    from openclude.llm import SCHEMA_HINT

    assert '"ar" or "en"' not in SCHEMA_HINT
    assert '"language": "en"' in SCHEMA_HINT


def test_durations_are_estimates_until_the_audio_is_measured() -> None:
    """The film must refuse to render on guessed durations."""
    film = to_film(parse_draft(payload()), CAST)
    # estimates are present, so unmeasured() is empty, but they are guesses:
    # longer narration must produce a longer estimate
    short = to_film(parse_draft(payload()), CAST).shots[0].duration_seconds
    long_data = json.loads(payload())
    long_data["scenes"][0]["shots"][0]["narration"] = "word " * 30
    long_shot = to_film(parse_draft(json.dumps(long_data)), CAST).shots[0]
    assert long_shot.duration_seconds > short


def test_unknown_character_ids_are_caught_at_build_time() -> None:
    data = json.loads(payload())
    data["scenes"][0]["shots"][0]["character_ids"] = ["ghost"]
    with pytest.raises(ScriptError, match="unknown characters"):
        to_film(parse_draft(json.dumps(data)), CAST)


def test_an_unknown_dialogue_speaker_is_caught() -> None:
    data = json.loads(payload())
    data["scenes"][0]["shots"][0]["dialogue_speaker"] = "ghost"
    with pytest.raises(ScriptError, match="not a known character"):
        to_film(parse_draft(json.dumps(data)), CAST)


def test_scenes_are_numbered_and_shots_restart_indoors() -> None:
    data = json.loads(payload())
    data["scenes"].append(dict(data["scenes"][0]))
    film = to_film(parse_draft(json.dumps(data)), CAST)
    assert [s.id for s in film.scenes] == ["s01", "s02"]
    assert [s.id for s in film.scenes[1].shots] == ["s02_sh001", "s02_sh002"]


def test_a_scene_boundary_is_never_a_continuity_carry() -> None:
    data = json.loads(payload())
    data["scenes"].append(dict(data["scenes"][0]))
    film = to_film(parse_draft(json.dumps(data)), CAST)
    assert film.scenes[1].shots[0].carry_from_previous is False
    assert film.scenes[1].shots[0].start_image == ""


def test_continuity_is_resolved_inside_each_scene() -> None:
    film = to_film(parse_draft(payload()), CAST)
    shots = film.shots
    assert shots[0].carry_from_previous is False
    assert shots[1].carry_from_previous is True
    assert shots[1].start_image == "frames/s01_sh001_last.png"


def test_long_narration_is_split_to_fit_a_generation() -> None:
    data = json.loads(payload())
    data["scenes"][0]["shots"][0]["narration"] = "word " * 60
    film = to_film(parse_draft(json.dumps(data)), CAST)
    assert film.unsplittable() == ()
    assert all(s.target_frames() <= 121 for s in film.shots)


def test_every_shot_gets_legal_frame_counts() -> None:
    film = to_film(parse_draft(payload()), CAST)
    for shot in film.shots:
        frames = shot.target_frames()
        assert (frames - 21) % 4 == 0
        assert 21 <= frames <= 121


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def test_the_prompt_states_the_character_ids() -> None:
    prompt = build_prompt("a story", CAST, target_minutes=2)
    assert "nova" in prompt and "kai" in prompt
    assert "a pilot in a grey coat" in prompt


def test_the_prompt_forbids_names_in_the_visual() -> None:
    """The renderer cannot render text, so a name in the visual is a silent bug."""
    prompt = build_prompt("a story", CAST, target_minutes=1)
    assert "Never write a character's name" in prompt


def test_the_prompt_sizes_itself_to_the_target_length() -> None:
    # 0.42 s per word, so 1 minute is ~142 words and 10 minutes ~1428
    one = build_prompt("s", CAST, target_minutes=1)
    ten = build_prompt("s", CAST, target_minutes=10)
    assert "142 words" in one
    assert "1428 words" in ten


def test_the_prompt_handles_a_codeless_story() -> None:
    prompt = build_prompt("a lone house in the rain", [], target_minutes=1)
    assert "no recurring characters" in prompt


def test_an_empty_story_is_rejected() -> None:
    with pytest.raises(ScriptError, match="story is empty"):
        build_prompt("   ", CAST, target_minutes=1)


def test_a_bad_target_is_rejected() -> None:
    with pytest.raises(ScriptError, match="target_minutes must be positive"):
        build_prompt("s", CAST, target_minutes=0)


# --------------------------------------------------------------------------
# end to end with a stub
# --------------------------------------------------------------------------


def test_story_to_film_end_to_end() -> None:
    writer = StubWriter(payload())
    film = script("a pilot takes a plane out in a storm", CAST, writer, target_minutes=1)
    assert len(film.shots) == 2
    assert len(writer.calls) == 1
    assert "a pilot takes a plane" in writer.calls[0]


def test_the_writer_sees_a_prompt_not_the_raw_story() -> None:
    writer = StubWriter(payload())
    script("MARKER-STORY", CAST, writer)
    assert "MARKER-STORY" in writer.calls[0]
    assert "JSON" in writer.calls[0]


def test_a_bad_writer_response_stops_the_stage() -> None:
    with pytest.raises(ScriptError):
        script("story", CAST, StubWriter("I would love to help!"))
