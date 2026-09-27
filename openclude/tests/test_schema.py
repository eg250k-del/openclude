"""Offline proof that the film schema holds.

No GPU, no network, no engine. If this file passes, the contract is sound and
every later stage can be built against it.  Run:

    python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from openclude.schema import (
    WAN22_5B,
    Character,
    CharacterView,
    Film,
    FrameSpec,
    Scene,
    SchemaError,
    Shot,
    framespec_for,
)


# --------------------------------------------------------------------------
# frame arithmetic — the rule every audited repo got wrong
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (1.0, 25),    # 24 -> 25   (25 = 21+4)
        (2.0, 49),    # 48 -> 49
        (3.0, 73),    # 72 -> 73
        (5.0, 121),   # capped at the model's 121 max
        (0.5, 21),    # below the minimum
        (9.0, 121),   # capped, never exceeds maximum
    ],
)
def test_frames_snap_to_legal_lattice(seconds: float, expected: int) -> None:
    assert WAN22_5B.frames_for_duration(seconds) == expected


def test_every_snapped_count_is_legal() -> None:
    """21 + 4n for n >= 0, and never above the model's maximum."""
    for frames in range(1, 200):
        snapped = WAN22_5B.snap(frames)
        assert snapped >= WAN22_5B.minimum
        assert (snapped - WAN22_5B.minimum) % WAN22_5B.step == 0
        assert snapped <= 121


def test_snapping_never_loses_a_whole_frame_downward() -> None:
    """The bug: int(duration*fps) truncates and drops a frame per shot.

    1,300 shots x 1 lost frame = 78 s of drift.  Snapping must round to
    NEAREST, so it can go up or down but never silently discard the tail.
    """
    for frames in range(21, 122):
        snapped = WAN22_5B.snap(frames)
        assert abs(snapped - frames) <= 2  # at most one step of 4, halved


def test_duration_roundtrip_is_stable() -> None:
    """frames -> seconds -> frames must be a fixed point."""
    for frames in (21, 25, 49, 81, 121):
        seconds = WAN22_5B.duration_for_frames(frames)
        assert WAN22_5B.frames_for_duration(seconds) == frames


def test_model_rejects_impossible_spec() -> None:
    with pytest.raises(SchemaError, match="frames_maximum"):
        FrameSpec(minimum=121, maximum=21)


def test_unknown_model_fails_loud() -> None:
    with pytest.raises(SchemaError, match="unknown model_type"):
        framespec_for("wan2_2_nope")


def test_zero_duration_fails_loud() -> None:
    with pytest.raises(SchemaError, match="duration must be positive"):
        WAN22_5B.frames_for_duration(0)


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------


def test_character_requires_description() -> None:
    with pytest.raises(SchemaError, match="description is required"):
        Character(id="nova", name="Nova", description="   ")


def test_character_rejects_duplicate_angles() -> None:
    view = CharacterView(angle="front", image_path="a.png")
    with pytest.raises(SchemaError, match="duplicate angle"):
        Character(
            id="nova",
            name="Nova",
            description="a pilot",
            views=(view, view),
        )


def test_reference_image_must_be_an_image() -> None:
    with pytest.raises(SchemaError, match="png/jpg/webp"):
        CharacterView(angle="front", image_path="notes.txt")


# --------------------------------------------------------------------------
# shots
# --------------------------------------------------------------------------


def make_shot(**overrides) -> Shot:
    base = dict(
        id="s01_sh001",
        scene_id="s01",
        index=1,
        narration="She opened the door and the wind took her hat.",
        visual="A woman in a grey coat steps onto a flooded balcony",
        character_ids=("nova",),
        camera="slow dolly in",
    )
    base.update(overrides)
    return Shot(**base)


def test_shot_id_format_is_enforced() -> None:
    with pytest.raises(SchemaError, match="shot id must look like"):
        make_shot(id="shot1")


def test_shot_needs_content() -> None:
    with pytest.raises(SchemaError, match="at least narration or visual"):
        make_shot(narration="", visual="")


def test_unmeasured_shot_refuses_to_size_itself() -> None:
    """The single most important guard: never estimate duration from text."""
    with pytest.raises(SchemaError, match="duration not measured yet"):
        make_shot().target_frames()


def test_measured_duration_drives_length() -> None:
    shot = make_shot().measured(2.4)
    assert shot.target_frames() == 57          # 2.4s * 24 = 57.6 -> 57 -> snap 57
    assert shot.target_seconds() == pytest.approx(57 / 24)


def test_non_positive_measurement_is_rejected() -> None:
    with pytest.raises(SchemaError, match="non-positive duration"):
        make_shot().measured(0.0)


def test_anchored_shot_needs_a_start_image() -> None:
    with pytest.raises(SchemaError, match="needs start_image or carry"):
        make_shot(anchor_mode="S")


def test_end_anchor_needs_an_end_image() -> None:
    with pytest.raises(SchemaError, match="requires end_image"):
        make_shot(anchor_mode="SE", start_image="a.png")


def test_carry_from_previous_is_a_legal_anchor() -> None:
    shot = make_shot(anchor_mode="S", carry_from_previous=True)
    assert shot.carry_from_previous is True


# --------------------------------------------------------------------------
# splitting — the silent-truncation bug
# --------------------------------------------------------------------------


def test_max_generation_is_5_seconds() -> None:
    assert make_shot().max_generation_seconds() == pytest.approx(5.042, abs=0.01)


def test_long_audio_is_flagged_for_splitting() -> None:
    assert make_shot().measured(12.0).needs_split() is True


def test_short_audio_needs_no_split() -> None:
    assert make_shot().measured(3.0).needs_split() is False


def test_oversized_shot_refuses_to_render() -> None:
    with pytest.raises(SchemaError, match="one generation of"):
        make_shot().measured(12.0).assert_single_generation()


def test_split_preserves_total_duration() -> None:
    shot = make_shot().measured(12.0)
    pieces = shot.split()
    assert len(pieces) == 3
    total = sum(p.duration_seconds for p in pieces)
    assert total == pytest.approx(12.0, abs=0.01)


def test_every_split_piece_fits_one_generation() -> None:
    for piece in make_shot().measured(12.0).split():
        assert piece.target_frames() <= 121
        piece.assert_single_generation()


def test_split_chains_continuity_head_to_tail() -> None:
    pieces = make_shot(
        anchor_mode="SE", start_image="a.png", end_image="z.png"
    ).measured(12.0).split()
    head, *middle, tail = pieces
    # head keeps the real start anchor, tail keeps the real end anchor
    assert head.anchor_mode == "S" and head.start_image == "a.png"
    assert tail.end_image == "z.png"
    assert tail.anchor_mode == "SE"
    # every middle piece is carried from whatever came before it
    assert all(p.carry_from_previous for p in middle)
    assert all(p.anchor_mode == "S" for p in middle)


def test_no_split_piece_ever_claims_an_anchor_it_does_not_have() -> None:
    """Regression: a head piece used to inherit 'SE' with no end image."""
    for anchor, start, end in [
        ("SE", "a.png", "z.png"),
        ("S", "a.png", ""),
        ("", "", ""),
    ]:
        for piece in make_shot(anchor_mode=anchor, start_image=start, end_image=end).measured(30.0).split():
            if "S" in piece.anchor_mode:
                assert piece.start_image or piece.carry_from_previous
            if "E" in piece.anchor_mode:
                assert piece.end_image


def test_split_of_a_short_shot_is_a_noop() -> None:
    shot = make_shot().measured(3.0)
    assert shot.split() == (shot,)


# --------------------------------------------------------------------------
# scene-level renumbering — ids must never collide
# --------------------------------------------------------------------------


def one_scene(*shots: Shot) -> Scene:
    return Scene(
        id="s01",
        index=1,
        summary="x",
        location="y",
        time_of_day="dawn",
        shots=shots,
    )


def test_scene_renumbers_after_splitting() -> None:
    scene = one_scene(
        make_shot(index=1, id="s01_sh001").measured(12.0),  # splits into 3
        make_shot(index=2, id="s01_sh002").measured(3.0),
    )
    fixed = scene.split_oversized()
    assert [s.id for s in fixed.shots] == [
        "s01_sh001",
        "s01_sh002",
        "s01_sh003",
        "s01_sh004",
    ]
    assert [s.index for s in fixed.shots] == [1, 2, 3, 4]
    assert all(s.scene_id == "s01" for s in fixed.shots)


def test_scene_ids_stay_unique_and_sortable() -> None:
    scene = one_scene(*[make_shot(index=i, id=f"s01_sh{i:03d}").measured(12.0) for i in range(1, 6)])
    ids = [s.id for s in scene.split_oversized().shots]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_scene_split_is_idempotent() -> None:
    scene = one_scene(make_shot(index=1).measured(12.0))
    once = scene.split_oversized()
    assert once.split_oversized() == once


def test_scene_split_preserves_total_time() -> None:
    scene = one_scene(
        make_shot(index=1).measured(12.0),
        make_shot(index=2).measured(4.0),
    )
    assert scene.split_oversized().total_seconds() == pytest.approx(16.0, abs=0.01)


def test_film_split_clears_every_overflow() -> None:
    nova = Character(id="nova", name="Nova", description="a pilot")
    scene = one_scene(
        *[make_shot(index=i, id=f"s01_sh{i:03d}").measured(12.0) for i in range(1, 4)]
    )
    film = Film(
        id="f01",
        title="t",
        language="ar",
        target_minutes=1,
        characters=(nova,),
        scenes=(scene,),
    )
    fixed = film.split_oversized()
    assert fixed.unsplittable() == ()
    assert len(fixed.shots) == 9  # 3 shots x 3 pieces


# --------------------------------------------------------------------------
# continuity resolution — must happen AFTER splitting
# --------------------------------------------------------------------------


def chain(*shots: Shot) -> Scene:
    """A scene where every shot after the first carries from the one before."""
    return Scene(
        id="s01",
        index=1,
        summary="x",
        location="y",
        time_of_day="dawn",
        shots=tuple(shots),
    )


def test_continuity_points_at_the_immediately_previous_shot() -> None:
    scene = chain(
        make_shot(index=1, id="s01_sh001", carry_from_previous=False),
        make_shot(index=2, id="s01_sh002", anchor_mode="S", carry_from_previous=True),
        make_shot(index=3, id="s01_sh003", anchor_mode="S", carry_from_previous=True),
    )
    shots = scene.resolve_continuity().shots
    assert shots[0].start_image == ""
    assert shots[1].start_image == "frames/s01_sh001_last.png"
    assert shots[2].start_image == "frames/s01_sh002_last.png"


def test_continuity_resolution_is_idempotent() -> None:
    scene = chain(
        make_shot(index=1, id="s01_sh001"),
        make_shot(index=2, id="s01_sh002", anchor_mode="S", carry_from_previous=True),
    )
    once = scene.resolve_continuity()
    assert once.resolve_continuity() == once


def test_continuity_is_resolved_against_post_split_ids() -> None:
    """The bug: paths baked in before splitting point at the wrong frames."""
    scene = chain(
        make_shot(index=1, id="s01_sh001", carry_from_previous=False),
        make_shot(index=2, id="s01_sh002", anchor_mode="S", carry_from_previous=True).measured(12.0),
    )
    fixed = scene.split_oversized().resolve_continuity()
    # shot 2 became 3 pieces; piece 2 must chain off piece 1, not off sh001
    assert fixed.shots[1].start_image == "frames/s01_sh001_last.png"
    assert fixed.shots[2].start_image == "frames/s01_sh002_last.png"
    assert fixed.shots[3].start_image == "frames/s01_sh003_last.png"


def test_scene_cannot_carry_from_a_previous_scene() -> None:
    scene = chain(make_shot(index=1, id="s01_sh001", anchor_mode="S", carry_from_previous=True))
    with pytest.raises(SchemaError, match="first shot in the scene"):
        scene.resolve_continuity()


def test_a_scene_boundary_is_a_hard_cut() -> None:
    """The first shot of every scene must not carry a stale start image."""
    a = Scene(
        id="s01", index=1, summary="x", location="y", time_of_day="dawn",
        shots=(make_shot(index=1, id="s01_sh001", character_ids=()),),
    )
    b = Scene(
        id="s02", index=2, summary="y", location="z", time_of_day="day",
        shots=(make_shot(index=1, id="s02_sh001", scene_id="s02", character_ids=()),),
    )
    film = Film(
        id="f01", title="t", language="ar", target_minutes=1, scenes=(a, b)
    ).resolve_continuity()
    assert film.scenes[1].shots[0].start_image == ""


# --------------------------------------------------------------------------
# determinism — the reproducibility bug
# --------------------------------------------------------------------------


def test_seed_is_deterministic_across_processes() -> None:
    """Same content -> same seed, always. No timestamps, no randomness."""
    a = make_shot().resolved_seed()
    b = make_shot().resolved_seed()
    assert a == b
    assert 0 < a <= 0x7FFFFFFF


def test_seed_changes_when_visual_changes() -> None:
    a = make_shot().resolved_seed()
    b = make_shot(visual="a different scene entirely").resolved_seed()
    assert a != b


def test_explicit_seed_wins() -> None:
    assert make_shot(seed=1234).resolved_seed() == 1234


def test_two_runs_produce_identical_settings(cast: list[Character]) -> None:
    """The property that makes A/B testing possible at all."""
    shot = make_shot(anchor_mode="S", start_image="start.png").measured(2.0)
    assert shot.to_engine_settings(cast) == shot.to_engine_settings(cast)


def test_missing_cast_fails_loud_instead_of_silently_dropping_the_character() -> None:
    """Regression guard: a shot must never render without its character sheet."""
    shot = make_shot().measured(2.0)
    with pytest.raises(SchemaError, match="unknown character 'nova'"):
        shot.to_engine_settings([])


# --------------------------------------------------------------------------
# engine bridge
# --------------------------------------------------------------------------


@pytest.fixture
def cast() -> list[Character]:
    return [
        Character(
            id="nova",
            name="Nova",
            description="Female, 25, athletic, jet black undercut hair, "
            "charcoal grey tactical shirt",
            views=(
                CharacterView(angle="front", image_path="nova_front.png"),
                CharacterView(angle="side", image_path="nova_side.png"),
            ),
        )
    ]


def test_settings_bridge_is_complete(cast: list[Character]) -> None:
    shot = make_shot(anchor_mode="SE", start_image="s.png", end_image="e.png").measured(2.0)
    s = shot.to_engine_settings(cast)

    assert s["model_type"] == "ti2v_2_2_fastwan"
    assert s["image_prompt_type"] == "SE"
    assert s["image_start"] == "s.png"
    assert s["image_end"] == "e.png"
    assert s["video_length"] == 49
    assert s["force_fps"] == 24
    assert s["seed"] == shot.resolved_seed()

    # character identity is injected into the prompt AND as reference images
    assert "Nova" in s["prompt"]
    assert "slow dolly in" in s["prompt"]
    assert s["image_refs"] == ["nova_front.png", "nova_side.png"]
    assert s["video_prompt_type"] == "I"


def test_unknown_character_fails_loud(cast: list[Character]) -> None:
    shot = make_shot(character_ids=("ghost",)).measured(2.0)
    with pytest.raises(SchemaError, match="unknown character"):
        shot.to_engine_settings(cast)


# --------------------------------------------------------------------------
# film
# --------------------------------------------------------------------------


def build_film() -> Film:
    nova = Character(id="nova", name="Nova", description="a pilot")
    shots = tuple(
        Shot(
            id=f"s01_sh{i:03d}",
            scene_id="s01",
            index=i,
            narration=f"Line {i}",
            visual=f"Shot {i}",
        ).measured(1.5)
        for i in range(1, 6)
    )
    scene = Scene(
        id="s01",
        index=1,
        summary="Opening",
        location="airfield",
        time_of_day="dawn",
        shots=shots,
    )
    return Film(
        id="f01",
        title="Test",
        language="ar",
        target_minutes=2,
        characters=(nova,),
        scenes=(scene,),
    )


def test_film_totals() -> None:
    film = build_film()
    assert len(film.shots) == 5
    assert film.total_seconds() == pytest.approx(7.5)
    assert film.unmeasured() == ()


def test_film_reports_unmeasured_shots() -> None:
    film = build_film()
    scenes = (
        Scene(
            id="s01",
            index=1,
            summary="x",
            location="y",
            time_of_day="dawn",
            shots=(film.shots[0],),
        ),
    )
    partial = Film(
        id="f01",
        title="Test",
        language="ar",
        target_minutes=2,
        characters=film.characters,
        scenes=scenes,
    )
    assert len(partial.unmeasured()) == 0  # the one shot we kept is measured


def test_shot_cannot_claim_the_wrong_scene() -> None:
    with pytest.raises(SchemaError, match="claims scene"):
        Scene(
            id="s01",
            index=1,
            summary="x",
            location="y",
            time_of_day="dawn",
            shots=(make_shot(scene_id="s02"),),
        )


def test_lookup_missing_shot_fails_loud() -> None:
    with pytest.raises(SchemaError, match="no shot"):
        build_film().shot("s01_sh999")
