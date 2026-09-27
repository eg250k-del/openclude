"""Shows what the machine would actually send to the GPU.

Run:
    python demo.py

This is the whole first milestone in one screen: a story in, five ready-to-run
shots out, with every number the engine needs and no guesswork.
"""

from __future__ import annotations

from openclude.schema import Character, CharacterView, Film, Scene, Shot


def build() -> Film:
    """A 60-second, two-shot-scene, one-character test film."""

    nova = Character(
        id="nova",
        name="Nova",
        description=(
            "Female, 25, athletic build, jet black undercut hair, "
            "charcoal grey tactical shirt, black cargo pants"
        ),
        views=(
            CharacterView(angle="front", image_path="refs/nova_front.png"),
            CharacterView(angle="side", image_path="refs/nova_side.png"),
            CharacterView(angle="back", image_path="refs/nova_back.png"),
        ),
        voice_reference="voices/nova.wav",
    )

    # Durations below are the values we EXPECT from the audio. In the real run
    # they are measured from the rendered voice file, never typed by hand.
    plan = [
        (
            "The rain had not stopped for nine days.",
            "A flooded airfield at dawn, one figure walking toward a dead cargo plane",
            "",
            "wide establishing, slow push in",
        ),
        (
            "She had flown this route twice before, and lost both times.",
            "Nova climbs the cargo plane ladder, hand on the rail",
            "S",
            "medium shot, follow from behind",
        ),
        (
            "The engines were cold. That was the good part.",
            "Close on Nova's face as she checks the instrument panel",
            "S",
            "close up, slight handheld",
        ),
        (
            "Somewhere below, the ground crew had already cleared the field.",
            "Ground crew silhouettes scatter across the wet tarmac",
            "S",
            "low angle, fast whip pan",
        ),
        (
            "This time the storm was on her side.",
            "Nova in the cockpit, engines spinning up, rain on the glass",
            "SE",
            "wide, then slow tilt up to the sky",
        ),
    ]

    shots = []
    for i, (narration, visual, anchor, camera) in enumerate(plan, start=1):
        shot = Shot(
            id=f"s01_sh{i:03d}",
            scene_id="s01",
            index=i,
            narration=narration,
            visual=visual,
            character_ids=("nova",),
            camera=camera,
            # start_image is deliberately left EMPTY for carried shots — the
            # chain is resolved after splitting, when the real ids are final.
            anchor_mode="" if i == 1 else anchor,
            carry_from_previous=(i in (2, 3, 4, 5)),
            end_image="refs/shot005_end.png" if i == 5 else "",
        )
        shots.append(shot.measured(11.0 if i == 1 else 12.0))

    return Film(
        id="f01",
        title="Nine Days of Rain",
        language="ar",
        target_minutes=1,
        characters=(nova,),
        scenes=(
            Scene(
                id="s01",
                index=1,
                summary="A pilot takes a derelict plane out in a storm",
                location="abandoned airfield",
                time_of_day="dawn",
                shots=tuple(shots),
            ),
        ),
    ).split_oversized().resolve_continuity()


def main() -> None:
    film = build()
    cast = list(film.characters)

    print("=" * 78)
    print(f"  {film.title}   |   {film.language}   |   target {film.target_minutes} min")
    print("=" * 78)
    print(f"  characters : {len(film.characters)}  ->  "
          f"{', '.join(c.id for c in film.characters)}")
    print(f"  scenes     : {len(film.scenes)}")
    print(f"  shots      : {len(film.shots)}")
    print(f"  length     : {film.total_seconds():.2f} s "
          f"({film.total_seconds() / 60:.2f} min)")
    print(f"  unmeasured : {len(film.unmeasured())}  <- must be 0 or the run stops")
    print(f"  oversized  : {len(film.unsplittable())}  <- must be 0, no clipped dialogue")
    print()

    for shot in film.shots:
        fs = shot.framespec
        seconds = shot.duration_seconds
        frames = shot.target_frames()
        actual = shot.target_seconds()

        drift_ms = (actual - seconds) * 1000

        print("-" * 78)
        print(f"  {shot.id}   anchor={shot.anchor_mode or '-':<3}   "
              f"seed={shot.resolved_seed()}")
        print(f"    narration : {shot.narration}")
        print(f"    visual    : {shot.visual}")
        print(f"    camera    : {shot.camera}")
        if shot.start_image:
            src = "carried from previous shot" if shot.carry_from_previous else "fixed"
            print(f"    start img : {shot.start_image}   ({src})")
        if shot.end_image:
            print(f"    end   img : {shot.end_image}")
        print(f"    length    : {seconds:.2f}s audio -> {frames} frames "
              f"-> {actual:.3f}s actual  (drift {drift_ms:+.0f} ms)")
        print(f"    char sheet: {', '.join(shot.character_ids) or '-'}")

    print("=" * 78)
    print("  ENGINE COMMANDS  (one job per shot, retried alone)")
    print("=" * 78)
    for shot in film.shots:
        s = shot.to_engine_settings(cast)
        print(f"\n  --- {shot.id} ---")
        for key, value in s.items():
            if key == "prompt":
                print(f"    {key:>18} = {value[:96]}...")
            else:
                print(f"    {key:>18} = {value!r}")

    print()
    print("=" * 78)
    print("  CONTINUITY CHAIN")
    print("=" * 78)
    for shot in film.shots:
        if shot.start_image:
            chain = shot.start_image
            if shot.carry_from_previous:
                chain += "   <- final frame of previous shot"
            else:
                chain += "   <- fixed reference"
            print(f"  {shot.id}  {chain}")
    print()


if __name__ == "__main__":
    main()
