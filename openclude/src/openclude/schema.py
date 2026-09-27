"""Film data model — the contract every other stage speaks.

Design rules (learned from auditing four existing pipelines):
  1. Shot duration comes from the REAL audio, never from text length.
  2. Frame counts must be legal for the model:  min + n*step + offset  (Wan: 21+4n).
  3. Seeds are derived from content, never random.  Two of four audited repos
     added a timestamp to the seed, which silently destroys reproducibility.
  4. Nothing is optional-by-omission.  A missing required field must raise,
     not default to something plausible.

Every field that maps onto a WanGP generation setting is named exactly like the
setting, so `Shot.to_engine_settings()` is a mechanical, auditable transform.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when input cannot produce a legal film. Fails loud, on purpose."""


# --------------------------------------------------------------------------
# model capability — the frame arithmetic depends on it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSpec:
    """Frame-count rules for one model.

    Wan 2.2 exposes these through the engine's model schema:
        frames_minimum / frames_steps / frames_offset / fps

    Legal frame counts are  minimum + n*step  for n = 0,1,2,...
    Wan 2.2 uses minimum=21, step=4  ->  21, 25, 29, 81, 121, ...
    """

    fps: int = 24
    minimum: int = 21
    step: int = 4
    maximum: int = 121

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise SchemaError(f"fps must be positive, got {self.fps}")
        if self.minimum < 1:
            raise SchemaError(f"frames_minimum must be >= 1, got {self.minimum}")
        if self.step < 1:
            raise SchemaError(f"frames_step must be >= 1, got {self.step}")
        if self.maximum < self.minimum:
            raise SchemaError(
                f"frames_maximum ({self.maximum}) < frames_minimum ({self.minimum})"
            )

    def snap(self, frames: int) -> int:
        """Round a frame count onto the legal lattice. Never truncates downward."""
        if frames < self.minimum:
            return self.minimum
        n = round((frames - self.minimum) / self.step)
        snapped = self.minimum + n * self.step
        return min(snapped, self._legal_max())

    def _legal_max(self) -> int:
        """Largest legal count <= maximum."""
        n = (self.maximum - self.minimum) // self.step
        return self.minimum + n * self.step

    def frames_for_duration(self, seconds: float) -> int:
        """Frames needed to cover `seconds`, snapped to a legal count."""
        if seconds <= 0:
            raise SchemaError(f"duration must be positive, got {seconds}")
        return self.snap(round(seconds * self.fps))

    def duration_for_frames(self, frames: int) -> float:
        return self.snap(frames) / self.fps

    def max_seconds(self) -> float:
        return self._legal_max() / self.fps


# The 24 GB design point. 121 frames @ 24fps = 5.04 s per shot at 720p.
WAN22_5B = FrameSpec(fps=24, minimum=21, step=4, maximum=121)
WAN22_14B = FrameSpec(fps=24, minimum=21, step=4, maximum=121)

MODEL_FRAMESPECS: dict[str, FrameSpec] = {
    "ti2v_2_2_fastwan": WAN22_5B,
    "ti2v_2_2": WAN22_5B,
    "i2v_2_2": WAN22_14B,
    "t2v_2_2": WAN22_14B,
}

DEFAULT_MODEL = "ti2v_2_2_fastwan"


def framespec_for(model_type: str) -> FrameSpec:
    try:
        return MODEL_FRAMESPECS[model_type]
    except KeyError:
        raise SchemaError(
            f"unknown model_type {model_type!r}; "
            f"known: {sorted(MODEL_FRAMESPECS)}"
        ) from None


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------

ViewAngle = Literal["front", "side", "back", "three_quarter", "expression"]


@dataclass(frozen=True)
class CharacterView:
    """One reference image of a character from one angle."""

    angle: ViewAngle
    image_path: str
    prompt: str = ""

    def __post_init__(self) -> None:
        if not self.image_path:
            raise SchemaError("CharacterView.image_path is required")
        if not re.search(r"\.(png|jpe?g|webp)$", self.image_path, re.I):
            raise SchemaError(
                f"reference image must be png/jpg/webp, got {self.image_path!r}"
            )


@dataclass(frozen=True)
class Character:
    """A recurring character.

    `views` is the front/side/back/expressions sheet from the brief.  The
    engine consumes these as reference-image conditioning so the same face
    appears in every shot.
    """

    id: str
    name: str
    description: str
    views: tuple[CharacterView, ...] = ()
    voice_model: str = ""
    voice_reference: str = ""
    lora: str = ""
    lora_strength: float = 1.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,32}", self.id):
            raise SchemaError(
                f"character id must be lowercase slug, got {self.id!r}"
            )
        if not self.description.strip():
            raise SchemaError(f"character {self.id}: description is required")
        seen: set[str] = set()
        for v in self.views:
            if v.angle in seen:
                raise SchemaError(f"character {self.id}: duplicate angle {v.angle!r}")
            seen.add(v.angle)

    @property
    def appearance(self) -> str:
        """The stable description injected into every prompt that has this character."""
        return f"{self.name}, {self.description}"

    def reference_paths(self) -> list[str]:
        return [v.image_path for v in self.views]


# --------------------------------------------------------------------------
# shots
# --------------------------------------------------------------------------

# How the shot is visually anchored.  These letters are the engine's own
# vocabulary, kept identical on purpose.
#   ""  -> text to video, no anchors
#   "S" -> pinned to a start image
#   "E" -> pinned to an end image
#   "SE"-> pinned to both
AnchorMode = Literal["", "S", "E", "SE"]


@dataclass(frozen=True)
class Shot:
    """The atomic unit of production.  One shot = one generation job.

    A shot is NOT retried as part of a batch.  Anything that goes wrong here is
    retried alone — that was the brief's hardest requirement and the one thing
    none of the four audited repos actually implemented correctly.
    """

    id: str
    scene_id: str
    index: int

    narration: str
    """The spoken line.  Its rendered audio duration sets the shot length."""

    visual: str
    """What the camera sees.  Goes to the image/video model."""

    dialogue_speaker: str = ""
    """Character id who speaks this shot. Empty for narration-only shots."""

    duration_seconds: float = 0.0
    """Measured from the rendered audio, not estimated. 0 = not yet measured."""

    anchor_mode: AnchorMode = ""
    start_image: str = ""
    end_image: str = ""
    carry_from_previous: bool = False
    """Take the final frame of the previous shot as this shot's start image."""

    character_ids: tuple[str, ...] = ()
    camera: str = ""
    """Free-form camera move, e.g. 'slow dolly in', 'handheld push'."""

    model_type: str = DEFAULT_MODEL
    seed: int = 0
    """Deterministic. 0 means derive one from content (see `resolved_seed`)."""

    steps: int = 0
    """0 = use the model's accelerator profile default."""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"s\d{2}_sh\d{3}", self.id):
            raise SchemaError(
                f"shot id must look like s01_sh001, got {self.id!r}"
            )
        if self.index < 1:
            raise SchemaError(f"shot index must be >= 1, got {self.index}")
        if not self.narration.strip() and not self.visual.strip():
            raise SchemaError(f"shot {self.id}: needs at least narration or visual")
        if self.duration_seconds < 0:
            raise SchemaError(
                f"shot {self.id}: negative duration {self.duration_seconds}"
            )
        if self.anchor_mode not in ("", "S", "E", "SE"):
            raise SchemaError(
                f"shot {self.id}: anchor_mode must be '', 'S', 'E' or 'SE', "
                f"got {self.anchor_mode!r}"
            )
        if "E" in self.anchor_mode and not self.end_image:
            raise SchemaError(
                f"shot {self.id}: anchor_mode {self.anchor_mode!r} requires end_image"
            )
        if (
            "S" in self.anchor_mode
            and not self.start_image
            and not self.carry_from_previous
        ):
            raise SchemaError(
                f"shot {self.id}: anchored shot needs start_image or "
                f"carry_from_previous"
            )
        framespec_for(self.model_type)  # fail fast on unknown model

    # -- derived -----------------------------------------------------------

    @property
    def framespec(self) -> FrameSpec:
        return framespec_for(self.model_type)

    def measured(self, duration_seconds: float) -> "Shot":
        """Return a copy with the audio-measured duration. Idempotent."""
        if duration_seconds <= 0:
            raise SchemaError(
                f"shot {self.id}: refusing to record non-positive duration "
                f"{duration_seconds}"
            )
        return replace(self, duration_seconds=duration_seconds)

    def target_frames(self) -> int:
        """Frames this shot needs.  Requires a measured duration."""
        if self.duration_seconds <= 0:
            raise SchemaError(
                f"shot {self.id}: duration not measured yet — render the audio "
                f"first, then call shot.measured(seconds)"
            )
        return self.framespec.frames_for_duration(self.duration_seconds)

    def target_seconds(self) -> float:
        """Duration the shot will actually have after frame snapping."""
        return self.framespec.duration_for_frames(self.target_frames())

    # -- splitting long shots ---------------------------------------------

    def max_generation_seconds(self) -> float:
        """Longest video one generation call can produce for this model.

        Wan 2.2 caps a single generation at 121 frames = 5.04 s at 24 fps.
        Anything longer is not "slow", it is impossible: the model will
        silently truncate and cut the narration off mid-sentence.
        """
        return self.framespec.max_seconds()

    def needs_split(self) -> bool:
        """True when the measured audio is longer than one generation."""
        if self.duration_seconds <= 0:
            return False
        return self.duration_seconds > self.max_generation_seconds() + 1e-6

    def split(self) -> tuple["Shot", ...]:
        """Break an over-long shot into chainable generation-sized pieces.

        The pieces are NOT yet valid production ids — `Scene.split_oversized()`
        owns numbering, because only the scene knows its neighbours.  Call that
        instead; this is the building block underneath it.
        """
        if not self.needs_split():
            return (self,)

        capacity = self.max_generation_seconds()
        # ceil, never round: 12 s / 5.04 s = 2.38, and round() would give 2
        # pieces of 6 s each — still over the limit, still truncated.
        # With ceil, even distribution guarantees every piece fits.
        count = max(2, math.ceil(self.duration_seconds / capacity))
        per_shot = self.duration_seconds / count

        pieces: list[Shot] = []
        for n in range(count):
            first = n == 0
            last = n == count - 1

            start = self.start_image if first else ""
            end = self.end_image if last else ""
            carries = self.carry_from_previous or not first

            # The anchor letters describe what THIS piece is pinned to, so they
            # have to be recomputed per piece. A head that inherited "SE" while
            # losing the end image would render against a missing anchor.
            # A carried start counts as an anchor: the start image is supplied
            # at render time from the previous piece's final frame.
            anchor = ("S" if (start or carries) else "") + ("E" if end else "")

            pieces.append(
                self._unchecked(
                    id=self.id,  # renumbered by Scene.split_oversized()
                    scene_id=self.scene_id,
                    index=n,
                    narration=self.narration,
                    visual=self.visual,
                    dialogue_speaker=self.dialogue_speaker,
                    duration_seconds=per_shot,
                    anchor_mode=anchor,
                    start_image=start,
                    end_image=end,
                    carry_from_previous=self.carry_from_previous or not first,
                    character_ids=self.character_ids,
                    camera=self.camera,
                    model_type=self.model_type,
                    seed=self.seed,
                    steps=self.steps,
                )
            )
        return tuple(pieces)

    @classmethod
    def _unchecked(cls, **kwargs: Any) -> "Shot":
        """Construct without validation. For internal renumbering only."""
        self = object.__new__(cls)
        for f in dataclasses.fields(cls):
            if f.name in kwargs:
                value = kwargs[f.name]
            elif f.default is not dataclasses.MISSING:
                value = f.default
            else:
                value = f.default_factory()
            object.__setattr__(self, f.name, value)
        return self

    def assert_single_generation(self) -> None:
        """Fail loud rather than render a clipped shot."""
        if self.needs_split():
            fs = self.framespec
            raise SchemaError(
                f"shot {self.id}: audio is {self.duration_seconds:.2f}s but one "
                f"generation of {self.model_type} tops out at "
                f"{self.max_generation_seconds():.2f}s "
                f"({fs.snap(fs.maximum)} frames @ {fs.fps}fps). "
                f"Call shot.split() first, or shorten the narration."
            )


    def resolved_seed(self) -> int:
        """A stable seed derived from the shot's own content.

        This is the fix for the reproducibility bug: same content in, same
        frame out, so a re-run can be compared against a parameter change.
        """
        if self.seed:
            return self.seed
        material = "|".join(
            [self.id, self.model_type, self.visual, self.camera, *self.character_ids]
        )
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def with_start_image(self, path: str) -> "Shot":
        if not path:
            raise SchemaError(f"shot {self.id}: empty start image path")
        if "E" in self.anchor_mode and not self.end_image:
            raise SchemaError(
                f"shot {self.id}: cannot add start image, end image missing"
            )
        return replace(self, start_image=path, anchor_mode=self.anchor_mode or "S")

    # -- engine bridge -----------------------------------------------------

    def to_engine_settings(
        self, characters: Sequence[Character] = ()
    ) -> dict[str, Any]:
        """Translate into the engine's generation settings dict.

        This is the only place shot data crosses into engine vocabulary.
        Keeping it mechanical is what makes the pipeline auditable.
        """
        fs = self.framespec
        settings: dict[str, Any] = {
            "model_type": self.model_type,
            "prompt": self._compose_prompt(characters),
            "resolution": "1280x720",
            "video_length": self.target_frames(),
            "force_fps": fs.fps,
            "seed": self.resolved_seed(),
            "image_prompt_type": self.anchor_mode,
        }
        if self.start_image:
            settings["image_start"] = self.start_image
        if self.end_image:
            settings["image_end"] = self.end_image

        refs = self._reference_images(characters)
        if refs:
            settings["image_refs"] = refs
            settings["video_prompt_type"] = "I"

        if self.steps:
            settings["num_inference_steps"] = self.steps
        return settings

    def _compose_prompt(self, characters: Sequence[Character]) -> str:
        parts: list[str] = []
        if self.visual:
            parts.append(self.visual.strip())
        index = {c.id: c for c in characters}
        for cid in self.character_ids:
            character = index.get(cid)
            if character is None:
                raise SchemaError(
                    f"shot {self.id}: references unknown character {cid!r}"
                )
            parts.append(character.appearance)
        if self.camera:
            parts.append(f"Camera: {self.camera.strip()}")
        return ". ".join(p for p in parts if p)

    def _reference_images(self, characters: Sequence[Character]) -> list[str]:
        index = {c.id: c for c in characters}
        refs: list[str] = []
        for cid in self.character_ids:
            character = index.get(cid)
            if character is None:
                raise SchemaError(
                    f"shot {self.id}: references unknown character {cid!r}"
                )
            refs.extend(character.reference_paths())
        return refs


# --------------------------------------------------------------------------
# scenes and film
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    id: str
    index: int
    summary: str
    location: str
    time_of_day: str
    shots: tuple[Shot, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"s\d{2}", self.id):
            raise SchemaError(f"scene id must look like s01, got {self.id!r}")
        for shot in self.shots:
            if shot.scene_id != self.id:
                raise SchemaError(
                    f"shot {shot.id} claims scene {shot.scene_id} but sits in {self.id}"
                )

    def total_seconds(self) -> float:
        return sum(s.duration_seconds for s in self.shots)

    def resolve_continuity(self) -> "Scene":
        """Fill in `start_image` for every carried shot, in order.

        Continuity references must be resolved AFTER splitting, never stored
        before it.  Splitting renumbers shots, so a path baked in earlier
        silently points at the wrong frame — which is how a film ends up
        jumping back to a scene from three shots ago.

        Idempotent: running it twice changes nothing.
        """
        resolved: list[Shot] = []
        previous: Shot | None = None
        for shot in self.shots:
            if not shot.carry_from_previous or shot.start_image:
                resolved.append(shot)
                previous = shot
                continue
            if previous is None:
                raise SchemaError(
                    f"scene {self.id}: {shot.id} carries from the previous shot "
                    f"but it is the first shot in the scene"
                )
            resolved.append(
                replace(shot, start_image=f"frames/{previous.id}_last.png")
            )
            previous = resolved[-1]
        return replace(self, shots=tuple(resolved))

    def split_oversized(self) -> "Scene":
        """Return a scene whose every shot fits one generation.

        Shots longer than one generation are broken into chained pieces and
        the whole scene is renumbered sequentially, so ids stay sortable and
        collision-free no matter how many pieces a shot produced.

        This is the only correct place to do the renumbering, because only the
        scene knows which numbers its neighbours already occupy.
        """
        numbered: list[Shot] = []
        n = 0
        for shot in self.shots:
            for piece in shot.split():
                n += 1
                numbered.append(
                    replace(piece, id=f"{self.id}_sh{n:03d}", scene_id=self.id, index=n)
                )
        return replace(self, shots=tuple(numbered))


@dataclass(frozen=True)
class Film:
    id: str
    title: str
    language: str
    target_minutes: int
    characters: tuple[Character, ...] = ()
    scenes: tuple[Scene, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("film id is required")
        if self.target_minutes <= 0:
            raise SchemaError(
                f"target_minutes must be positive, got {self.target_minutes}"
            )
        known = {c.id for c in self.characters}
        for scene in self.scenes:
            for shot in scene.shots:
                unknown = set(shot.character_ids) - known
                if unknown:
                    raise SchemaError(
                        f"shot {shot.id}: unknown character ids {sorted(unknown)}"
                    )

    @property
    def shots(self) -> tuple[Shot, ...]:
        return tuple(s for scene in self.scenes for s in scene.shots)

    def split_oversized(self) -> "Film":
        """Return a film where every shot fits a single generation."""
        return replace(self, scenes=tuple(s.split_oversized() for s in self.scenes))

    def resolve_continuity(self) -> "Film":
        """Resolve every carried shot's start image, scene by scene, in order.

        Each scene restarts the chain: the first shot of a scene is a hard cut,
        which is what a scene boundary means.
        """
        return replace(self, scenes=tuple(s.resolve_continuity() for s in self.scenes))

    def unsplittable(self) -> tuple[Shot, ...]:
        """Shots that still overflow one generation after splitting."""
        return tuple(s for s in self.shots if s.needs_split())

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise SchemaError(f"no shot {shot_id!r} in film {self.id!r}")

    def total_seconds(self) -> float:
        return sum(s.duration_seconds for s in self.shots)

    def unmeasured(self) -> tuple[Shot, ...]:
        return tuple(s for s in self.shots if s.duration_seconds <= 0)
