"""Shows the machine surviving a disaster mid-film.

Run:
    python demo_crash.py

It runs a 5-shot film against a deliberately hostile fake GPU, kills the
process at shot 3, then starts a second process that finishes the job. The
state file is plain JSON you can open and read.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from openclude.runner import run_film
from openclude.schema import Character, Film, Scene, Shot
from openclude.state import Ledger

logging.basicConfig(
    level=logging.INFO, format="    %(levelname)-7s %(message)s"
)


def build_film() -> Film:
    nova = Character(id="nova", name="Nova", description="a pilot in a grey coat")
    shots = tuple(
        Shot(
            id=f"s01_sh{i:03d}", scene_id="s01", index=i,
            narration=f"Narration line {i}", visual=f"Visual {i}",
            character_ids=("nova",),
        ).measured(2.0)
        for i in range(1, 6)
    )
    return Film(
        id="f01", title="Crash Test", language="ar", target_minutes=1,
        characters=(nova,),
        scenes=(
            Scene(id="s01", index=1, summary="x", location="y",
                   time_of_day="dawn", shots=shots),
        ),
    )


class Preempted(BaseException):
    """Not an Exception, so nothing can catch it. Like a real SIGKILL."""


def hostile_gpu(settings: dict) -> str:
    """Oversized jobs blow VRAM, then the node disappears."""
    n = int(settings["prompt"].split("Visual ")[1][0])
    if settings["video_length"] > 25:
        raise RuntimeError(
            "The generation of the video has encountered an error: it is likely "
            "that you have unsufficient VRAM and you should therefore reduce the "
            "video resolution or its number of frames."
        )
    if n == 3:
        raise Preempted("container preempted by the host")
    return f"out/s01_sh{n:03d}.mp4"


def healthy_gpu(settings: dict) -> str:
    n = int(settings["prompt"].split("Visual ")[1][0])
    return f"out/s01_sh{n:03d}.mp4"


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def show_state(path: Path) -> None:
    data = json.loads(path.read_text("utf-8"))
    print()
    print(f"  {'shot':<12} {'status':<9} {'tries':>5} {'frames':>7}  output")
    print("  " + "-" * 68)
    for shot_id, rec in sorted(data["records"].items()):
        tries = len(rec["attempts"])
        print(
            f"  {shot_id:<12} {rec['status']:<9} {tries:>5} "
            f"{rec['frames']:>7}  {rec['output'] or '-'}"
        )


def main() -> None:
    film = build_film()
    state = Path(tempfile.gettempdir()) / "openclude_demo_state.json"
    state.unlink(missing_ok=True)

    banner("RUN 1 - the node dies at shot 3")
    led = Ledger.open(state, film, max_attempts=4)
    try:
        report = run_film(film, led, hostile_gpu)
        print(f"\n  {report.line()}")
    except Preempted:
        print("\n  *** process killed. nothing was lost. ***")
    print(f"  {led.summary(film)}")
    show_state(state)

    banner("RUN 2 - fresh process, same state file")
    led2 = Ledger.open(state, film, max_attempts=4)
    print(f"  resuming from: {[s.id for s in led2.workable(film)]}")
    report2 = run_film(film, led2, healthy_gpu)
    print(f"\n  {report2.line()}")
    print(f"  {led2.summary(film)}")
    show_state(state)

    banner("RUN 3 - a finished film does no work")
    calls = {"n": 0}

    def counting(settings: dict) -> str:
        calls["n"] += 1
        return "x"

    report3 = run_film(film, Ledger.open(state, film), counting)
    print(f"  renders attempted: {calls['n']}")
    print(f"  {report3.line()}")

    print()
    print(f"  state file: {state}")
    print("  it is plain JSON - open it any time to see where the film stands.")
    print()


if __name__ == "__main__":
    main()
