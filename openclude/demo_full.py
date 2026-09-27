"""The whole machine, one screen, with a fake GPU so it runs anywhere.

    python demo_full.py

story -> script -> audio -> shots -> render -> film
Every stage shown with real numbers, including what a 2-hour target would
actually cost on a rented 4090.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openclude.audio import write_tone_wav
from openclude.llm import StubWriter
from openclude.pipeline import Config, Pipeline
from openclude.schema import Character
from openclude.storage import LocalStore

DRAFT = json.dumps({
    "title": "Nine Days of Rain",
    "language": "ar",
    "scenes": [
        {
            "summary": "A pilot takes a derelict plane out in a storm",
            "location": "abandoned airfield",
            "time_of_day": "dawn",
            "shots": [
                {"narration": "The rain had not stopped for nine days.",
                 "visual": "a flooded runway at dawn, one figure walking toward a dead cargo plane",
                 "camera": "wide establishing, slow push in", "character_ids": ["nova"]},
                {"narration": "She had flown this route twice, and lost both times.",
                 "visual": "the figure climbs a ladder into a dead cargo plane, hand on the rail",
                 "camera": "medium shot, follow from behind", "character_ids": ["nova"]},
                {"narration": "The engines were cold. That was the good part.",
                 "visual": "close on the face as a gloved hand checks an instrument panel",
                 "camera": "close up, slight handheld", "character_ids": ["nova"]},
                {"narration": "Somewhere below, the ground crew had cleared the field.",
                 "visual": "silhouettes scatter across wet tarmac under floodlights",
                 "camera": "low angle, fast whip pan", "character_ids": ["nova"]},
            ],
        }
    ],
})


class FakeSynth:
    def speak(self, text, out_path, voice="", emotion=""):
        # 0.42 s per word, like real speech
        seconds = max(0.6, len(text.split()) * 0.42)
        write_tone_wav(out_path, seconds)
        return str(out_path)


def fake_gpu(clips: Path):
    clips.mkdir(parents=True, exist_ok=True)

    def render(settings: dict) -> str:
        # 24 s per generation is roughly a 5B model at 4 steps on a 4090
        import time
        time.sleep(0.02)
        p = clips / f"{settings['seed']}.mp4"
        p.write_bytes(b"\x00" * 2048)
        return str(p)

    return render


def patch_assembly(monkey):
    import openclude.assembly as assembly

    def fake_concat(clips, out, work):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"\x00" * 4096)
        return str(out)

    assembly.concat = fake_concat
    assembly.verify = lambda *a, **kw: "OK assembled=59.0s expected=59.0s drift=+0.0s"
    assembly.mux = lambda v, a, o: (Path(o).write_bytes(b"\x00" * 256), str(o))[1]


def line(char="-", n=76):
    return char * n


def build_pipeline(store: LocalStore, root: Path) -> Pipeline:
    """Everything the demo needs, in one place. Writes only under `root`."""
    nova = Character(
        id="nova", name="Nova",
        description="Female, 25, athletic, jet black undercut hair, charcoal grey tactical shirt",
        voice_reference="voices/nova.wav",
    )
    cfg = Config(film_id="f01", language="ar", target_minutes=1,
                 work_dir=str(root / "work"))
    return Pipeline(cfg, store, StubWriter(DRAFT), FakeSynth(),
                    fake_gpu(root / "clips"), [nova])


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="openclude_full_"))
    patch_assembly(None)

    store = LocalStore(root / "bucket")
    p = build_pipeline(store, root)
    cfg = p.cfg

    print(line("="))
    print("  OPENCLIDE  -  the whole machine")
    print(line("="))
    print("  story : a pilot takes a derelict plane out in a storm")
    print("  cast  : nova (voice: voices/nova.wav)")
    print(f"  target: {cfg.target_minutes} minute, {cfg.language}")
    print(f"  store : {store.root}")

    result = p.run("a pilot takes a derelict plane out in a storm")

    print(line("="))
    print("  OPENCLIDE  -  the whole machine")
    print(line("="))
    print(f"  story : a pilot takes a derelict plane out in a storm")
    print(f"  cast  : nova (voice: voices/nova.wav)")
    print(f"  target: {cfg.target_minutes} minute, {cfg.language}")
    print(f"  store : {store.root}")

    result = p.run("a pilot takes a derelict plane out in a storm")

    print()
    print(line())
    print("  STAGES")
    print(line())
    for s in result.stages:
        print(f"    {s.name:<10} {'ok' if s.ok else 'FAIL':<5} {s.seconds:>6.2f}s"
              f"   {s.detail}")

    film = p.stage_script("a pilot takes a derelict plane out in a storm")
    film, audio = p.stage_audio(film)

    print()
    print(line())
    print("  WHAT THE FILM LOOKS LIKE")
    print(line())
    print(f"  {'shot':<12} {'audio':>7} {'video':>8} {'frames':>7}  anchor")
    for shot in film.shots:
        a = audio.get(shot.id)
        print(f"  {shot.id:<12} {a.seconds if a else 0:>6.2f}s "
              f"{shot.target_seconds():>7.2f}s {shot.target_frames():>7}  "
              f"{shot.anchor_mode or '-'}")

    print()
    print(line())
    print("  CONTINUITY")
    print(line())
    for shot in film.shots:
        if shot.start_image:
            print(f"  {shot.id}  <- {shot.start_image}")
    print(f"  (carried frame: {sum(1 for s in film.shots if s.carry_from_previous)} shots)")

    print()
    print(line())
    print("  BANKED IN THE STORE")
    print(line())
    keys = sorted(store.list("f01/"))
    for kind in ("audio", "frames", "clips", "state", "script", "film"):
        n = sum(1 for k in keys if f"/{kind}/" in k or k.endswith(f"/{kind}.json")
                or k.endswith(f"/{kind}.mp4"))
        if n:
            print(f"    {kind:<8} {n:>4} files")

    print()
    print(line())
    print("  WHAT A REAL FILM WOULD COST  (RTX 4090 on SaladCloud)")
    print(line())
    sec_per_shot = film.total_seconds() / len(film.shots)
    shots_per_min = 60 / sec_per_shot
    # Two tiers, because "Wan 2.2" is not one thing. These are wall-clock
    # estimates for 121 frames at 720p on a 4090, and they are the single
    # biggest driver of cost, so they are stated as ranges, not point values.
    TIERS = (
        ("5B  FastWan  3-step  720p", 60, 90),
        ("14B A14B Lightning 4-step", 150, 300),
    )
    for minutes in (1, 20, 60, 120):
        shots = minutes * shots_per_min
        print(f"    {minutes:>4} min = {shots:>7.0f} shots")
        for name, lo, hi in TIERS:
            h_lo = shots * lo / 3600
            h_hi = shots * hi / 3600
            print(
                f"            {name:<26} {h_lo:>6.1f}-{h_hi:>6.1f} GPU-h"
                f"   ${h_lo * 0.16:>6.2f} - ${h_hi * 0.27:>6.2f}"
            )
    print()
    print(f"    measured shot length here: {sec_per_shot:.2f}s of film per shot")
    print("    prices: $0.16 (lowest) - $0.27 (medium priority) per 4090-hour")
    print(line("="))
    print()


if __name__ == "__main__":
    main()
