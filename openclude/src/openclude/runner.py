"""The render loop: schema + ledger + retry ladder, wired together.

Deliberately engine-agnostic.  `render` is any callable that takes engine
settings and returns an output path; it raises on failure.  That means the
whole orchestration — resume, per-shot isolation, bounded degradation — is
testable without a GPU, and swapping in the real engine later changes nothing
here.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .retry import LadderExhausted, classify, next_attempt
from .schema import Film, Shot
from .state import Ledger

log = logging.getLogger("openclude.runner")

RenderFn = Callable[[dict], str]
"""(engine settings) -> output path.  Raises to signal failure."""


class RenderResult(Protocol):  # pragma: no cover - documentation only
    path: str


@dataclass
class RunReport:
    """What one invocation of the loop achieved."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_done: int = 0
    degraded: int = 0
    seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"attempted={self.attempted} ok={self.succeeded} failed={self.failed} "
            f"already_done={self.skipped_done} degraded={self.degraded} "
            f"elapsed={self.seconds:.1f}s"
        )


def render_shot(
    shot: Shot,
    ledger: Ledger,
    render: RenderFn,
    film: Film,
) -> tuple[str, int]:
    """Render one shot, degrading and retrying within the attempt budget.

    Returns (output_path, degradations_applied).

    Raises nothing: a shot that cannot be saved is recorded in the ledger and
    reported through the ledger, so the rest of the film keeps moving.  That is
    the whole point of the brief.
    """
    settings = shot.to_engine_settings(film.characters)
    used_rungs: set[str] = set()
    degradations = 0
    last_error = ""

    while True:
        ledger.begin(shot, settings)
        started = time.time()
        try:
            output = render(settings)
        except Exception as exc:  # noqa: BLE001 - classified, then degraded
            # KeyboardInterrupt / SystemExit / GeneratorExit are NOT caught
            # here on purpose: a container being preempted or an operator
            # pressing Ctrl+C must stop the process, not burn a retry. The
            # ledger is already on disk, so the next run resumes cleanly.
            elapsed = time.time() - started
            last_error = f"{type(exc).__name__}: {exc}"
            kind = classify(last_error)
            ledger.fail(shot.id, last_error, kind=kind)
            log.warning(
                "shot=%s attempt=%d failed kind=%s after %.1fs: %s",
                shot.id,
                ledger.record(shot.id).attempt_count,
                kind,
                elapsed,
                last_error,
            )
            try:
                step = next_attempt(settings, kind, frozenset(used_rungs))
            except LadderExhausted:
                return "", degradations
            used_rungs.add(step.rung)
            settings = step.settings
            degradations += 1
            log.info(
                "shot=%s degrading via %s (%s)", shot.id, step.rung, step.note
            )
            continue

        seconds = time.time() - started
        frames = int(settings.get("video_length", 0) or 0)
        ledger.succeed(shot.id, output, seconds=seconds, frames=frames)
        log.info("shot=%s ok in %.1fs -> %s", shot.id, seconds, output)
        return output, degradations


def run_film(
    film: Film,
    ledger: Ledger,
    render: RenderFn,
    *,
    sleep_between: float = 0.0,
) -> RunReport:
    """Work through every shot the ledger still needs.

    Safe to call repeatedly and safe to interrupt at any point: the ledger is
    written before and after every shot, so the next call picks up where this
    one stopped.
    """
    report = RunReport()
    todo = ledger.workable(film)
    report.skipped_done = len(film.shots) - len(todo) - len(ledger.blocked(film))
    started = time.time()

    for shot in todo:
        report.attempted += 1
        try:
            output, degradations = render_shot(shot, ledger, render, film)
        except Exception as exc:  # noqa: BLE001
            # render_shot already recorded it; keep the film moving
            report.failed += 1
            report.failures.append((shot.id, f"{type(exc).__name__}: {exc}"))
            log.exception("shot=%s broke the loop", shot.id)
            continue
        report.degraded += degradations
        if output:
            report.succeeded += 1
        else:
            report.failed += 1
            report.failures.append((shot.id, ledger.record(shot.id).error))
        if sleep_between:
            time.sleep(sleep_between)

    report.seconds = time.time() - started
    return report
