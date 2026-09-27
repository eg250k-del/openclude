"""Assembly: turn finished clips into a film.

The audit found three different concat implementations in one repo, two of
them contradicting each other, one of which was a copy-paste stub whose
"crossfade" branch was byte-identical to its "no transition" branch. This is
the fourth and only one.

Rules that came out of that mess:
  * one implementation, not three
  * audio is muxed per shot, then the clips are concatenated with `-c copy`,
    so a 2-hour concat is a file copy rather than a 2-hour re-encode
  * the final cut is probed afterwards and checked against the expected length
  * a missing clip is a loud error, never a film that quietly ends early
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

FFMPEG = os.environ.get("FFMPEG_BINARY", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BINARY", "ffprobe")

#: One shot longer than this cannot be produced in a single generation on the
#: 24 GB target, so assembly treats it as a hard error rather than a gap.
MAX_SHOT_SECONDS = 6.0


class AssemblyError(RuntimeError):
    pass


class MissingClip(AssemblyError):
    pass


class DurationDrift(AssemblyError):
    pass


def require_tools() -> None:
    missing = [t for t in (FFMPEG, FFPROBE) if shutil.which(t) is None]
    if missing:
        raise AssemblyError(
            f"missing {', '.join(missing)} on PATH. Assembly cannot run."
        )


def _run(cmd: list[str], timeout: int) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AssemblyError(f"{cmd[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise AssemblyError(
            f"{Path(cmd[0]).name} timed out after {timeout}s. A 2-hour concat is a "
            f"file copy, so a timeout here means something is wrong."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise AssemblyError(
            f"{Path(cmd[0]).name} failed (exit {proc.returncode}): "
            + " | ".join(tail)
        )
    return proc.stdout


def probe_duration(path: str | os.PathLike[str]) -> float:
    out = _run(
        [
            FFPROBE, "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        timeout=120,
    )
    try:
        return float(json.loads(out)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AssemblyError(f"cannot read the length of {path}") from exc


def extract_last_frame(
    video: str | os.PathLike[str], out_png: str | os.PathLike[str]
) -> str:
    """Grab the FINAL frame of a clip. This is the continuity handoff.

    The bug this replaces: one of the audited repos used `select='eq(n,0)'`,
    which is the FIRST frame, saved it as `shotN_last.png`, and fed the opening
    image of the previous shot into the next one. Every cut in that film was
    wrong and nothing said so.
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            FFMPEG, "-y", "-v", "error", "-sseof", "-1",
            "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out),
        ],
        timeout=300,
    )
    if not out.exists() or out.stat().st_size == 0:
        raise MissingClip(f"could not extract a last frame from {video}")
    return str(out)


def mux(
    video: str | os.PathLike[str],
    audio: str | os.PathLike[str],
    out: str | os.PathLike[str],
    trim_to_audio: bool = True,
) -> str:
    """Put narration on a clip, optionally cutting the clip to the audio length.

    Per-shot trimming is what keeps lip movement and dialogue together. The
    alternative - mux once at the end - is the version the audit flagged as
    producing stuttering audio over a two-hour film.
    """
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
    ]
    if trim_to_audio:
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", str(out_p)]
    _run(cmd, timeout=1800)
    if not out_p.exists() or out_p.stat().st_size == 0:
        raise AssemblyError(f"mux produced nothing for {video}")
    return str(out_p)


@dataclass(frozen=True)
class Clip:
    shot_id: str
    path: str
    seconds: float = 0.0
    audio: str = ""


def concat(clips: Sequence[Clip], out: str | os.PathLike[str], work_dir: str | os.PathLike[str]) -> str:
    """Join clips in order with a stream copy. No re-encode, no drift.

    `-c copy` is only safe because every clip came out of the same encoder
    with the same settings, and the per-shot mux normalised them.
    """
    if not clips:
        raise AssemblyError("nothing to assemble")
    if len(clips) < 2:
        raise AssemblyError(
            f"only one clip ({clips[0].shot_id}). A film needs at least two."
        )

    for c in clips:
        if not Path(c.path).exists():
            raise MissingClip(
                f"{c.shot_id} has no rendered file at {c.path}. "
                f"Refusing to assemble a film with a hole in it."
            )
        d = probe_duration(c.path)
        if d > MAX_SHOT_SECONDS + 0.5:
            raise AssemblyError(
                f"{c.shot_id} is {d:.2f}s, over the {MAX_SHOT_SECONDS}s single-shot "
                f"limit. It should have been split before rendering."
            )

    listing = Path(work_dir) / "concat.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    # absolute paths: a relative list breaks the moment the cwd changes, which
    # is exactly what the engine's own init() does
    listing.write_text(
        "\n".join(f"file '{Path(c.path).resolve().as_posix()}'" for c in clips) + "\n",
        encoding="utf-8",
    )

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out_p),
        ],
        timeout=3600,
    )
    if not out_p.exists() or out_p.stat().st_size == 0:
        raise AssemblyError("concat produced an empty file")
    return str(out_p)


def verify(film_seconds: float, out: str | os.PathLike[str], tolerance: float = 1.0) -> str:
    """Check the cut is as long as the script said it would be."""
    actual = probe_duration(out)
    drift = actual - film_seconds
    if abs(drift) > tolerance:
        raise DurationDrift(
            f"assembled film is {actual:.2f}s but the script totals "
            f"{film_seconds:.2f}s (drift {drift:+.2f}s). Something was dropped."
        )
    return (
        f"OK assembled={actual:.2f}s expected={film_seconds:.2f}s "
        f"drift={drift:+.2f}s"
    )


@dataclass
class AssemblyReport:
    clips: int = field(default=0)
    muxed: int = 0
    last_frames: int = 0
    output: str = ""
    note: str = ""

    def line(self) -> str:
        return (
            f"clips={self.clips} muxed={self.muxed} "
            f"last_frames={self.last_frames} output={self.output}"
        )
