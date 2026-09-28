"""The whole pipeline, end to end.

story -> script -> audio -> shots -> clips -> film

Every stage is resumable at its own boundary, because on this infrastructure
the container is assumed to die without warning. The orchestrator holds no
state of its own: it asks the store what already exists and picks up there.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import assembly
from .audio import (
    AudioError,
    ShotAudio,
    apply_measurements,
    probe,
    render_narration,
    timing_report,
)
from .llm import ScriptError, ScriptWriter, script
from .runner import run_film
from .schema import Character, Film
from .state import Ledger
from .storage import Layout, Store, put_checked, restore_if_present

log = logging.getLogger("openclude.pipeline")


class PipelineError(RuntimeError):
    pass


@dataclass
class Config:
    film_id: str = "f01"
    # English. This was "ar" in three places across the codebase, on a project
    # with no Arabic content, and the first real LLM call inherited it and would
    # have sent English narration to an Arabic voice. A default should be the
    # common case, not an accident of whoever wrote the file first.
    language: str = "en"
    target_minutes: int = 1
    model_type: str = "ti2v_2_2_fastwan"
    max_attempts: int = 4
    work_dir: str = "work"
    mux_audio: bool = True
    verify_tolerance: float = 1.0


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float = 0.0
    detail: str = ""


@dataclass
class PipelineResult:
    stages: list[StageResult] = field(default_factory=list)
    film_seconds: float = 0.0
    output: str = ""
    complete: bool = False

    def line(self) -> str:
        return " | ".join(
            f"{s.name}={'ok' if s.ok else 'FAIL'}({s.seconds:.1f}s)"
            for s in self.stages
        )


class Pipeline:
    def __init__(
        self,
        config: Config,
        store: Store,
        writer: ScriptWriter,
        synth: Any,
        render: Callable[[dict], str],
        characters: Sequence[Character] = (),
    ) -> None:
        self.cfg = config
        self.store = store
        self.writer = writer
        self.synth = synth
        self.render = render
        self.characters = tuple(characters)
        self.work = Path(config.work_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        # the layout is rooted at the work dir, so every path this pipeline
        # writes is absolute and survives the engine's os.chdir
        self.layout = Layout(config.film_id, base=self.work)
        self._make_dirs()

    def _make_dirs(self) -> None:
        """Create every directory this pipeline writes to, once.

        Doing it here rather than trusting each backend to mkdir is what keeps
        a stubbed or swapped-out backend from failing on a missing parent.
        """
        for d in ("state", "script", "audio", "frames", "clips", "film"):
            Path(self.layout.path(f"{self.cfg.film_id}/{d}")).mkdir(
                parents=True, exist_ok=True
            )

    # -- stage 1: script ---------------------------------------------------

    def stage_script(self, story: str) -> Film:
        """Draft the film, unless a good one is already banked."""
        cached = restore_if_present(
            self.store, self.layout.key_script(), self.work / "script.json"
        )
        if cached:
            data = json.loads(Path(cached).read_text("utf-8"))
            if data.get("model_type") == self.cfg.model_type:
                log.info("script: reusing the banked draft")
                return self._film_from_dict(data)
            log.info("script: banked draft is for a different model, redrafting")

        film = script(
            story,
            self.characters,
            self.writer,
            target_minutes=self.cfg.target_minutes,
            film_id=self.cfg.film_id,
            model_type=self.cfg.model_type,
        )
        put_checked(self.store, self.layout.key_script(), self._dump(film))
        return film

    def _film_from_dict(self, data: dict) -> Film:
        from .llm import Draft, parse_draft

        return self._rebuild(parse_draft(json.dumps(data["draft"])), data.get("seconds"))

    def _rebuild(self, draft, seconds: dict[str, float] | None = None):
        from .llm import to_film

        film = to_film(
            draft,
            self.characters,
            film_id=self.cfg.film_id,
            target_minutes=self.cfg.target_minutes,
            model_type=self.cfg.model_type,
        )
        if not seconds:
            return film
        from .llm import Draft as D

        scenes = []
        for scene in film.scenes:
            shots = tuple(
                s.measured(seconds[s.id]) if s.id in seconds else s
                for s in scene.shots
            )
            from .schema import Scene

            scenes.append(
                Scene(
                    id=scene.id, index=scene.index, summary=scene.summary,
                    location=scene.location, time_of_day=scene.time_of_day,
                    shots=shots,
                )
            )
        from .schema import Film as F

        rebuilt = F(
            id=film.id, title=film.title, language=film.language,
            target_minutes=film.target_minutes, characters=film.characters,
            scenes=tuple(scenes),
        )
        return rebuilt.split_oversized().resolve_continuity()

    def _dump(self, film: Film) -> str:
        from .llm import Draft, DraftScene, DraftShot

        draft = Draft(
            title=film.title,
            language=film.language,
            scenes=tuple(
                DraftScene(
                    summary=s.summary, location=s.location, time_of_day=s.time_of_day,
                    shots=tuple(
                        DraftShot(
                            narration=sh.narration, visual=sh.visual,
                            camera=sh.camera, character_ids=sh.character_ids,
                            dialogue_speaker=sh.dialogue_speaker, anchor=sh.anchor_mode,
                        )
                        for sh in s.shots
                    ),
                )
                for s in film.scenes
            ),
        )
        p = self.work / "script.json"
        p.write_text(
            json.dumps(
                {
                    "draft": json.loads(json.dumps(draft.__dict__ | {
                        "scenes": [
                            {**s.__dict__, "shots": [
                                {**sh.__dict__, "character_ids": list(sh.character_ids)}
                                for sh in s.shots
                            ]} for s in draft.scenes
                        ]
                    })),
                    "model_type": self.cfg.model_type,
                    "seconds": {s.id: s.duration_seconds for s in film.shots},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(p)

    # -- stage 2: audio ----------------------------------------------------

    def stage_audio(self, film: Film) -> tuple[Film, dict[str, ShotAudio]]:
        """Speak every line, measure it, and re-cut the film to match."""
        audio: dict[str, ShotAudio] = {}
        for shot in film.shots:
            if not shot.narration.strip():
                continue
            local = self.work / "audio" / f"{shot.id}.wav"
            remote = restore_if_present(
                self.store, self.layout.key_audio(shot.id), local
            )
            if remote:
                info = probe(remote)
                voice, emotion = "", ""
            else:
                written = self.synth.speak(
                    shot.narration, local, voice=shot.dialogue_speaker, emotion=""
                )
                info = probe(written or local)
                put_checked(self.store, self.layout.key_audio(shot.id), info.path)
                voice, emotion = shot.dialogue_speaker, ""
            audio[shot.id] = ShotAudio(
                shot_id=shot.id, path=info.path, seconds=info.seconds,
                voice=voice, emotion=emotion,
            )

        if not audio:
            log.warning("audio: no narration in this film")
            return film, audio
        return apply_measurements(film, audio), audio

    # -- stage 3: render ---------------------------------------------------

    def stage_render(self, film: Film, audio: dict[str, ShotAudio]) -> Ledger:
        """Render every shot, banking each clip the moment it exists."""
        state_path = self.work / "ledger.json"
        restored = restore_if_present(self.store, self.layout.key_state(), state_path)
        if restored is None:
            # a banked ledger is the authority; download it before rendering
            pass

        ledger = Ledger.open(state_path, film, max_attempts=self.cfg.max_attempts)

        def wrapped(settings: dict) -> str:
            settings = dict(settings)
            out = self.render(settings)
            if out:
                put_checked(self.store, self.layout.key_clip(Path(out).stem), out)
            return out

        report = run_film(film, ledger, wrapped)
        log.info("render: %s", report.line())
        put_checked(self.store, self.layout.key_state(), state_path)
        return ledger

    # -- stage 4: assemble -------------------------------------------------

    def stage_assemble(
        self, film: Film, ledger: Ledger, audio: dict[str, ShotAudio]
    ) -> str:
        """Mux, concatenate, verify. Refuses to ship a film with a hole."""
        clips: list[assembly.Clip] = []
        for shot in film.shots:
            rec = ledger.record(shot.id)
            if not rec.is_done:
                raise PipelineError(
                    f"cannot assemble: {shot.id} is {rec.status.value}. "
                    f"Fix or reset it before building a film."
                )
            local = Path(rec.output)
            if not local.exists():
                remote = restore_if_present(
                    self.store, self.layout.key_clip(shot.id), self.work / local.name
                )
                if remote is None:
                    raise PipelineError(
                        f"cannot assemble: {shot.id} has no clip locally or upstream"
                    )
                local = Path(remote)
            clips.append(assembly.Clip(shot_id=shot.id, path=str(local)))

        if self.cfg.mux_audio and audio:
            muxed: list[assembly.Clip] = []
            for clip in clips:
                rec = audio.get(clip.shot_id)
                if rec is None:
                    muxed.append(clip)
                    continue
                out = self.layout.path_voiced(clip.shot_id)
                assembly.mux(clip.path, rec.path, out)
                muxed.append(assembly.Clip(clip.shot_id, str(out)))
            clips = muxed

        out = assembly.concat(clips, self.layout.path_film(), self.work)
        note = assembly.verify(film.total_seconds(), out, self.cfg.verify_tolerance)
        log.info("assemble: %s", note)
        put_checked(self.store, self.layout.key_film(), out)
        return out

    # -- the whole thing ---------------------------------------------------

    def run(self, story: str) -> PipelineResult:
        result = PipelineResult()

        def stage(name: str, fn):
            t0 = time.time()
            try:
                value = fn()
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                result.stages.append(
                    StageResult(name, False, time.time() - t0, f"{type(exc).__name__}: {exc}")
                )
                raise
            result.stages.append(StageResult(name, True, time.time() - t0))
            return value

        film = stage("script", lambda: self.stage_script(story))
        film, audio = stage("audio", lambda: self.stage_audio(film))
        if audio:
            log.info("audio: timing\n%s", timing_report(film, audio))
        ledger = stage("render", lambda: self.stage_render(film, audio))
        output = stage("assemble", lambda: self.stage_assemble(film, ledger, audio))

        result.film_seconds = film.total_seconds()
        result.output = output
        result.complete = ledger.done_count() == len(film.shots)
        return result
