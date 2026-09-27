"""Degradation policy: what to try next when a shot fails.

The engine reports a failure; it does not recover.  Its own message says
"reduce the video resolution or its number of frames" and then stops.  Left
alone, an unattended run burns its retries re-sending the exact same oversized
job and the shot dies for good.

This module decides the next attempt.  It is a pure function over settings, so
the whole policy is testable without a GPU.

Two rules govern everything here:
  * only degrade what the failure class actually implicates
  * a failure that no amount of shrinking will fix (bad settings, missing
    model) is never retried — it is raised, not swallowed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schema import FrameSpec, framespec_for

# failure classes
VRAM = "vram"
OOM_RAM = "ram"
ENGINE = "engine"
VALIDATION = "validation"
MISSING = "missing"
DOWNLOAD = "download"
CANCELLED = "cancelled"
UNKNOWN = "unknown"

#: Failures that shrinking the job cannot possibly fix.
TERMINAL = frozenset({VALIDATION, MISSING, CANCELLED})

#: Failures where the same request is likely to work on a retry.
TRANSIENT = frozenset({DOWNLOAD, ENGINE, UNKNOWN})

#: Failures that mean "the job was too big for this card".
RESOURCE = frozenset({VRAM, OOM_RAM})


class LadderExhausted(RuntimeError):
    """Nothing left to try for this shot."""


def classify(message: str) -> str:
    """Map an engine error string onto a failure class.

    Keyed off the engine's own wording, which is stable in practice because it
    is the same code path that produced it.  Order matters: the engine's VRAM
    message also contains "error" and "reduce", so resource checks come first.
    """
    m = message.lower()

    if "too many resources" in m:
        return OOM_RAM
    if "insufficient vram" in m or "unsufficient vram" in m or "out of vram" in m:
        return VRAM
    if "out of memory" in m:
        # CUDA's own OOM is RAM-side; the engine's VRAM message says "VRAM"
        return OOM_RAM if "vram" not in m else VRAM
    if "no such file" in m or "not found" in m or "does not exist" in m:
        return MISSING
    # the engine's own validation wording: "doesn't accept", "must provide",
    # "must be", "unsupported", "invalid", "the number of X should be"
    if (
        "accept" in m
        or "unsupported" in m
        or "must " in m
        or "invalid" in m
        or "should be" in m
    ):
        return VALIDATION
    if "download" in m or "connection" in m or "timed out" in m or "timeout" in m:
        return DOWNLOAD
    if "cancel" in m or "abort" in m:
        return CANCELLED
    return UNKNOWN


def should_retry(kind: str) -> bool:
    return kind not in TERMINAL


def _halve_length(settings: dict[str, Any], spec: FrameSpec) -> dict[str, Any]:
    """Shorter video, on the legal lattice, never below the model's floor.

    Returns the input untouched when the length is already at the floor, so
    the ladder can tell "this rung did nothing" from "this rung helped" and
    move on instead of burning a retry.
    """
    current = int(settings.get("video_length", spec.maximum))
    target = max(spec.minimum, current // 2)
    snapped = spec.snap(target)
    if snapped >= current:
        return settings
    out = dict(settings)
    out["video_length"] = snapped
    if "sliding_window_size" in out:
        out["sliding_window_size"] = min(int(out["sliding_window_size"]), snapped)
    return out


def _shrink_resolution(settings: dict[str, Any]) -> dict[str, Any]:
    """Halve the frame area, keeping both sides on a multiple of 16."""
    raw = str(settings.get("resolution", "1280x720"))
    try:
        w, h = (int(x) for x in raw.lower().split("x", 1))
    except ValueError:
        raise LadderExhausted(f"cannot parse resolution {raw!r}")
    nw = max(256, ((w // 2) // 16) * 16)
    nh = max(144, ((h // 2) // 16) * 16)
    if (nw, nh) >= (w, h):
        return settings
    out = dict(settings)
    out["resolution"] = f"{nw}x{nh}"
    return out


def _fewer_steps(settings: dict[str, Any]) -> dict[str, Any]:
    steps = int(settings.get("num_inference_steps", 0) or 0)
    if steps <= 2:
        return settings
    out = dict(settings)
    out["num_inference_steps"] = max(2, steps // 2)
    return out


def _drop_quality_extras(settings: dict[str, Any]) -> dict[str, Any]:
    """Drop conditioning that costs memory: reference images first.

    Losing the character sheet is a real quality loss, so it is the LAST rung,
    after length, steps and resolution have all been tried.
    """
    if not settings.get("image_refs"):
        return settings
    out = dict(settings)
    out.pop("image_refs", None)
    out.pop("video_prompt_type", None)
    return out


#: Ordered rungs, cheapest quality loss first.
LADDER: tuple[tuple[str, Callable[[dict[str, Any], FrameSpec], dict[str, Any]]], ...] = (
    ("length", _halve_length),
    ("steps", _fewer_steps),
    ("resolution", _shrink_resolution),
    ("quality", _drop_quality_extras),
)


@dataclass(frozen=True)
class Degradation:
    """The next thing to try, and what changed."""

    settings: dict[str, Any]
    rung: str
    note: str

    def describe(self) -> str:
        return f"{self.rung}: {self.note}"


def next_attempt(
    settings: dict[str, Any], kind: str, used_rungs: frozenset[str] = frozenset()
) -> Degradation:
    """Return degraded settings for the next try, or raise.

    `used_rungs` is the set of rungs already applied to this shot, so the
    ladder never repeats itself and always terminates.
    """
    if not should_retry(kind):
        raise LadderExhausted(
            f"{kind} failures are terminal; retrying cannot help"
        )

    spec = framespec_for(str(settings.get("model_type", "")))
    before = dict(settings)

    if kind in TRANSIENT:
        # same job, just try again — but only once per rung so it terminates
        if "resend" in used_rungs:
            raise LadderExhausted("transient failure already retried unchanged")
        return Degradation(before, "resend", "retrying the identical job")

    for name, step in LADDER:
        if name in used_rungs:
            continue
        candidate = step(before, spec) if name == "length" else step(before)
        if candidate == before:
            continue  # this rung had nothing left to give
        changed = ", ".join(
            f"{k}: {before[k]!r} -> {candidate[k]!r}"
            for k in sorted(set(before) & set(candidate))
            if before[k] != candidate[k]
        )
        return Degradation(candidate, name, changed or "no comparable fields")

    raise LadderExhausted(
        "every rung is spent; the shot cannot be shrunk any further"
    )


def settings_produced_by(
    settings: dict[str, Any], kind: str, attempts: int
) -> dict[str, Any]:
    """Apply the ladder up to `attempts` times, stopping when it runs out.

    Planning helper: never raises, so a caller can ask "what would 6 retries
    produce" and get the floor rather than an exception.
    """
    out = dict(settings)
    used: set[str] = set()
    for _ in range(max(0, attempts)):
        try:
            step = next_attempt(out, kind, frozenset(used))
        except LadderExhausted:
            break
        out = step.settings
        used.add(step.rung)
    return out
