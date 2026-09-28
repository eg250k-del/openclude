"""The real engine adapter.

Everything above this line is pure data. This is where it becomes a GPU job.

The engine is driven through its own importable Python session rather than a
web UI, because:
  * the UI has no API (no `api_name` anywhere), so a browser-driven approach
    would mean driving a Gradio page, which is not survivable unattended;
  * the in-process API validates a task before loading the model, so a typo in
    a 1,300-shot manifest is caught in milliseconds instead of after a
    15-minute model load;
  * one session keeps the last model warm across shots, which is most of the
    difference between a film costing 20 dollars and one costing 80.

The adapter is the ONLY place that imports the engine. That keeps the rest of
the codebase testable on a laptop with no GPU, and it keeps the engine
upgrade path to a single file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .audio import ShotAudio
from .schema import Character, Film, Shot
from .state import Ledger

DEFAULT_ROOT = "/opt/WanGP"
"""Where the engine is baked into the container image."""


class EngineError(RuntimeError):
    pass


class EngineUnavailable(EngineError):
    """The engine could not be imported. Not retryable by degrading."""


@dataclass(frozen=True)
class EnginePaths:
    """Where everything lives. One object so a test can redirect all of it."""

    root: str = DEFAULT_ROOT
    config: str = "/data/config"
    output: str = "/data/outputs"
    frames: str = "/data/frames"
    audio: str = "/data/audio"

    def ensure(self) -> "EnginePaths":
        for p in (self.config, self.output, self.frames, self.audio):
            Path(p).mkdir(parents=True, exist_ok=True)
        return self

    def last_frame(self, shot_id: str) -> str:
        return str(Path(self.frames) / f"{shot_id}_last.png")

    def clip(self, shot_id: str) -> str:
        return str(Path(self.output) / f"{shot_id}.mp4")


@dataclass
class WanGPAdapter:
    """Drives the engine's Python session. One instance per process.

    The engine keeps aggressive global state: a single runtime singleton, a
    process-wide generation lock, a chdir on init, and module-level globals.
    So: one adapter, one root, one shot at a time. Scaling is more containers,
    never more threads in here.
    """

    paths: EnginePaths = field(default_factory=EnginePaths)
    attention: str = "sage2"
    profile: int = 4
    preload_mb: int = 0
    verbose: bool = False
    _session: Any = field(default=None, init=False, repr=False)
    _import_error: str = field(default="", init=False, repr=False)

    @property
    def session(self) -> Any:
        """The live engine session, or None.

        Exposed so the speech backend can share it. The engine keeps a
        process-wide runtime singleton and raises if you init a second session
        with different arguments, so starting one per consumer does not work.
        """
        return self._session

    @property
    def started(self) -> bool:
        return self._session is not None

    # -- lifecycle ---------------------------------------------------------

    def prefetch(self, model_type: str) -> list[str]:
        """Download this model's weights before anything else needs them.

        Called from the worker rather than from the container entrypoint, and
        the reason is specific: the entrypoint ran the download before the
        process that serves /health existed, so for the three minutes the
        download took, nothing answered port 8000. The liveness probe counted
        that as failure and killed the container, and `restart_policy: always`
        turned it into a loop. The group showed `running` and then `creating`,
        forever, writing nothing.

        So the download has to happen with the health server already up, and it
        reports progress through the readiness endpoint rather than through the
        absence of a process.

        The engine is imported as `shared.api`, which is the module the
        renderer actually uses. The entrypoint's heredoc imported `wgp`, a
        module that is not what anything calls, so its download step could not
        have worked even if it had reached it.
        """
        import sys

        root = str(self.paths.root)
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from shared.utils.download import download_file  # type: ignore[import-not-found]
            from shared.utils.files_locator import (  # type: ignore[import-not-found]
                get_smart_download_location,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(
                f"cannot import the engine's downloader: {exc}"
            ) from exc

        definition_file = Path(self.paths.root) / "defaults" / f"{model_type}.json"
        if not definition_file.exists():
            raise EngineUnavailable(
                f"no definition for {model_type!r} at {definition_file}"
            )
        definition = json.loads(definition_file.read_text("utf-8"))

        urls = definition.get("URLs", [])
        quantised = [u for u in urls if "quanto" in u or "int8" in u]
        need = quantised[:1] or urls[:1]
        need += definition.get("VAE_URLs", [])[:1]
        need += definition.get("text_encoder_URLs", [])[:1]
        for key in ("URLs2", "loras", "preload_URLs"):
            extra = definition.get(key, [])
            if isinstance(extra, list) and extra and isinstance(extra[0], str):
                need += extra[:1]

        fetched: list[str] = []
        for url in need:
            name = url.rsplit("/", 1)[-1]
            target = get_smart_download_location(name)
            if Path(target).exists() and Path(target).stat().st_size > 0:
                fetched.append(name)
                continue
            download_file(url, target)
            fetched.append(name)
        return fetched

    def start(self) -> "WanGPAdapter":
        """Load the engine. Idempotent."""
        if self._session is not None:
            return self
        self.paths.ensure()
        try:
            from shared.api import init  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            self._import_error = f"{type(exc).__name__}: {exc}"
            raise EngineUnavailable(
                f"cannot import the engine's API ({self._import_error}). "
                f"Is the engine installed at {self.paths.root}?"
            ) from exc

        try:
            self._session = init(
                root=self.paths.root,
                config_path=str(Path(self.paths.config) / "wgp_config.json"),
                output_dir=self.paths.output,
                cli_args=[
                    "--attention", self.attention,
                    "--profile", str(self.profile),
                    "--preload", str(self.preload_mb),
                ],
                console_output=self.verbose,
                console_isatty=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._import_error = f"{type(exc).__name__}: {exc}"
            raise EngineUnavailable(f"engine refused to start: {exc}") from exc
        return self

    def stop(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
            self._session = None

    def __enter__(self) -> "WanGPAdapter":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- validation --------------------------------------------------------

    def validate(self, settings: dict[str, Any]) -> None:
        """Ask the engine to check a manifest before spending GPU on it.

        Cheap, and it turns "1,300 jobs, 400 of them silently wrong" into
        "one clear error in a second".
        """
        if self._session is None:
            self.start()
        try:
            defaults = self._session.get_default_settings(settings["model_type"])
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"engine does not recognise model_type "
                f"{settings.get('model_type')!r}: {exc}"
            ) from exc
        if not defaults:
            raise EngineError(
                f"engine returned no defaults for {settings.get('model_type')!r}; "
                f"the model is probably not installed"
            )

    def capabilities(self, model_type: str) -> dict[str, Any]:
        """What this model can actually do, straight from the engine."""
        if self._session is None:
            self.start()
        return self._session.get_model_schema(model_type) or {}

    # -- generation --------------------------------------------------------

    def render(self, settings: dict[str, Any]) -> str:
        """Render one shot. Returns the output path. Raises on failure."""
        if self._session is None:
            self.start()

        model_type = settings["model_type"]
        prefix = Path(settings.get("output_filename") or self._stem(settings))

        try:
            result = self._session.run_task(
                {**settings, "output_filename": str(prefix)}
            )
        except Exception as exc:  # noqa: BLE001 - re-raised for the retry ladder
            raise RuntimeError(str(exc)) from exc

        if not getattr(result, "success", False):
            errors = getattr(result, "errors", None) or []
            detail = "; ".join(
                getattr(e, "message", str(e)) for e in errors
            ) or "the engine reported failure with no message"
            raise RuntimeError(f"{model_type}: {detail}")

        files = list(getattr(result, "generated_files", None) or [])
        if not files:
            raise RuntimeError(
                f"{model_type}: the engine reported success but produced no file. "
                f"Treating as a failure rather than a silent hole in the film."
            )
        return files[0]

    def _stem(self, settings: dict[str, Any]) -> str:
        # the manifest already carries a unique seed per shot; use it so two
        # shots can never collide on an output filename
        return f"shot_{settings.get('seed', 'x')}"

    # -- post-production ---------------------------------------------------

    def mux_audio(self, video: str, audio_path: str, out: str) -> str:
        """Attach narration to a clip."""
        if self._session is None:
            self.start()
        try:
            self._session.run_task(
                {
                    "model_type": "edit_remux",
                    "video_source": video,
                    "audio_source": audio_path,
                    "output_filename": out,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"audio mux failed: {exc}") from exc
        return out


def attach_audio(
    settings: dict[str, Any], audio: dict[str, ShotAudio]
) -> dict[str, Any]:
    """Add the shot's narration to its engine settings, when there is one."""
    rec = audio.get(settings.get("_shot_id", ""))
    if rec is not None:
        settings["audio_guide"] = rec.path
        settings["audio_prompt_type"] = "A"
    return settings
