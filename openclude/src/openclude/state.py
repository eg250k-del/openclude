"""Durable per-shot state — the thing none of the audited repos had.

The engine has no checkpoint of its own: if a render dies at minute 9 of 10,
that shot is gone and the film has to restart.  The brief demanded the
opposite — "one failed scene is retried, not the whole film".  This module is
that mechanism.

Design constraints that come from where this actually runs:
  * SaladCloud containers are stateless and get killed at random.  The ledger
    is a single file so it can be synced to object storage as one object.
  * A crash during a write must not corrupt the ledger.  Every save is
    atomic (temp file + os.replace).
  * Re-running must be safe.  A shot already marked done is never re-rendered.
  * The seed is pinned in the ledger, so a shot resumed tomorrow renders the
    exact same frames it would have rendered today.
  * A shot that says RUNNING when we load it means the process died mid-shot.
    That is not corruption, it is the normal case, so we repair it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import Film, SchemaError, Shot

LEDGER_VERSION = 1
DEFAULT_MAX_ATTEMPTS = 4
"""4 attempts, matching the Salad job queue's own retry policy."""


class ShotStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Attempt:
    """One try at one shot. Append-only — history is never rewritten."""

    n: int
    started_at: float
    finished_at: float
    ok: bool
    error: str = ""
    kind: str = ""
    """Coarse failure class, e.g. 'vram', 'engine', 'download', 'validation'.
    The retry ladder keys off this, not off the message text."""

    @property
    def seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    status: ShotStatus = ShotStatus.PENDING
    seed: int = 0
    """Pinned on first claim. Survives restarts, so reruns are identical."""
    params_fingerprint: str = ""
    """Hash of the engine settings. A change here means the shot must re-run."""
    output: str = ""
    seconds: float = 0.0
    frames: int = 0
    error: str = ""
    current_started_at: float = 0.0
    """When begin() was called for the attempt in flight, so timing is real."""
    attempts: tuple[Attempt, ...] = ()

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def is_done(self) -> bool:
        return self.status is ShotStatus.DONE

    @property
    def is_running(self) -> bool:
        return self.status is ShotStatus.RUNNING

    def last_kind(self) -> str:
        return self.attempts[-1].kind if self.attempts else ""

    def exhausted(self, max_attempts: int) -> bool:
        return (
            self.status is ShotStatus.FAILED
            and self.attempt_count >= max_attempts
        )


def fingerprint(settings: dict[str, Any]) -> str:
    """Stable hash of engine settings, so we notice when a shot must re-run."""
    import hashlib

    material = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass
class Ledger:
    """The whole film's progress, in one file."""

    film_id: str
    path: Path | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    records: dict[str, ShotRecord] = field(default_factory=dict)
    run_started_at: float = field(default_factory=time.time)
    version: int = LEDGER_VERSION

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        film: Film,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> "Ledger":
        """Load an existing ledger, or start one bound to this film.

        Opening twice is safe and is the normal case on a retry.
        """
        p = Path(path)
        ledger = cls(film_id=film.id, path=p, max_attempts=max_attempts)
        if p.exists():
            ledger._load()
            if ledger.film_id != film.id:
                raise LedgerError(
                    f"ledger {p} belongs to film {ledger.film_id!r}, "
                    f"refusing to reuse it for {film.id!r}"
                )
        ledger.reconcile(film)
        return ledger

    @classmethod
    def in_memory(cls, film: Film, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> "Ledger":
        return cls(film_id=film.id, path=None, max_attempts=max_attempts).reconcile(film)

    def _load(self) -> None:
        assert self.path is not None
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise LedgerError(
                f"ledger {self.path} is corrupt ({exc}). "
                f"Refusing to guess — restore it or start a new film."
            ) from exc
        if raw.get("version") != LEDGER_VERSION:
            raise LedgerError(
                f"ledger {self.path} is version {raw.get('version')}, "
                f"this build speaks version {LEDGER_VERSION}"
            )
        self.film_id = raw.get("film_id", self.film_id)
        self.run_started_at = raw.get("run_started_at", self.run_started_at)
        self.records = {
            k: ShotRecord(
                shot_id=v["shot_id"],
                status=ShotStatus(v["status"]),
                seed=v.get("seed", 0),
                params_fingerprint=v.get("params_fingerprint", ""),
                output=v.get("output", ""),
                seconds=v.get("seconds", 0.0),
                frames=v.get("frames", 0),
                error=v.get("error", ""),
                attempts=tuple(
                    Attempt(
                        n=a["n"],
                        started_at=a["started_at"],
                        finished_at=a["finished_at"],
                        ok=a["ok"],
                        error=a.get("error", ""),
                        kind=a.get("kind", ""),
                    )
                    for a in v.get("attempts", [])
                ),
            )
            for k, v in raw.get("records", {}).items()
        }

    def save(self) -> None:
        """Atomic write. Either the whole new state lands, or none of it."""
        if self.path is None:
            return
        payload = {
            "version": LEDGER_VERSION,
            "film_id": self.film_id,
            "run_started_at": self.run_started_at,
            "max_attempts": self.max_attempts,
            "updated_at": time.time(),
            "records": {
                k: {
                    "shot_id": r.shot_id,
                    "status": r.status.value,
                    "seed": r.seed,
                    "params_fingerprint": r.params_fingerprint,
                    "output": r.output,
                    "seconds": r.seconds,
                    "frames": r.frames,
                    "error": r.error,
                    "attempts": [asdict(a) for a in r.attempts],
                }
                for k, r in self.records.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(payload, indent=2, sort_keys=True)
        # temp file in the same directory so os.replace stays on one filesystem
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # -- reconciliation ----------------------------------------------------

    def reconcile(self, film: Film) -> "Ledger":
        """Make the ledger agree with the film, and repair crash damage.

        Three jobs:
          1. add records for shots the ledger has never seen
          2. drop records for shots the film no longer has
          3. a shot still marked RUNNING was interrupted by a dead process,
             so it goes back to PENDING
        """
        live = {s.id for s in film.shots}
        for shot in film.shots:
            rec = self.records.get(shot.id)
            if rec is None:
                self.records[shot.id] = ShotRecord(shot_id=shot.id)
            elif rec.is_running:
                self.records[shot.id] = replace(rec, status=ShotStatus.PENDING)
        for dead in set(self.records) - live:
            del self.records[dead]
        return self

    # -- queries -----------------------------------------------------------

    def record(self, shot_id: str) -> ShotRecord:
        try:
            return self.records[shot_id]
        except KeyError:
            raise LedgerError(f"shot {shot_id!r} is not in this ledger") from None

    def status(self, shot_id: str) -> ShotStatus:
        return self.record(shot_id).status

    def is_done(self, shot_id: str) -> bool:
        return self.record(shot_id).is_done

    def workable(self, film: Film) -> tuple[Shot, ...]:
        """Shots this run should actually attempt, in film order.

        Done shots are skipped.  Exhausted shots are skipped, because retrying
        them forever is how an unattended job burns money without finishing.
        """
        return tuple(
            s
            for s in film.shots
            if not self.record(s.id).is_done
            and not self.record(s.id).exhausted(self.max_attempts)
        )

    def blocked(self, film: Film) -> tuple[Shot, ...]:
        """Shots we gave up on, with the reason attached."""
        return tuple(
            s
            for s in film.shots
            if self.record(s.id).exhausted(self.max_attempts)
        )

    def done_count(self) -> int:
        return sum(1 for r in self.records.values() if r.is_done)

    def progress(self, film: Film) -> float:
        total = len(film.shots)
        return 0.0 if total == 0 else self.done_count() / total

    # -- transitions -------------------------------------------------------

    def begin(self, shot: Shot, settings: dict[str, Any]) -> ShotRecord:
        """Claim a shot. Pins the seed so a later resume renders the same thing."""
        rec = self.record(shot.id)
        if rec.is_done:
            raise LedgerError(
                f"shot {shot.id} is already done; refusing to re-render it"
            )
        if rec.exhausted(self.max_attempts):
            raise LedgerError(
                f"shot {shot.id} used all {self.max_attempts} attempts; "
                f"last error: {rec.error}"
            )
        pinned_seed = rec.seed or shot.resolved_seed()
        updated = replace(
            rec,
            status=ShotStatus.RUNNING,
            seed=pinned_seed,
            params_fingerprint=fingerprint(settings),
            current_started_at=time.time(),
            error="",
        )
        self.records[shot.id] = updated
        self.save()
        return updated

    def succeed(
        self, shot_id: str, output: str, seconds: float = 0.0, frames: int = 0
    ) -> ShotRecord:
        rec = self.record(shot_id)
        if not rec.is_running:
            raise LedgerError(
                f"shot {shot_id}: succeed() without begin() - the ledger would "
                f"lose the attempt record (status is {rec.status.value})"
            )
        now = time.time()
        attempt = Attempt(
            n=rec.attempt_count + 1,
            started_at=rec.current_started_at or now,
            finished_at=now,
            ok=True,
        )
        updated = replace(
            rec,
            status=ShotStatus.DONE,
            output=output,
            seconds=round(rec.seconds + seconds, 3),
            frames=frames,
            error="",
            attempts=rec.attempts + (attempt,),
        )
        self.records[shot_id] = updated
        self.save()
        return updated

    def fail(self, shot_id: str, error: str, kind: str = "unknown") -> ShotRecord:
        rec = self.record(shot_id)
        if not rec.is_running:
            raise LedgerError(
                f"shot {shot_id}: fail() without begin() - the ledger would "
                f"lose the attempt record (status is {rec.status.value})"
            )
        now = time.time()
        attempt = Attempt(
            n=rec.attempt_count + 1,
            started_at=rec.current_started_at or now,
            finished_at=now,
            ok=False,
            error=error[:2000],
            kind=kind,
        )
        updated = replace(
            rec,
            status=ShotStatus.FAILED,
            error=error[:2000],
            attempts=rec.attempts + (attempt,),
        )
        self.records[shot_id] = updated
        self.save()
        return updated

    def skip(self, shot_id: str, reason: str) -> ShotRecord:
        """Mark a shot intentionally not rendered, e.g. it was removed."""
        updated = replace(
            self.record(shot_id), status=ShotStatus.SKIPPED, error=reason[:2000]
        )
        self.records[shot_id] = updated
        self.save()
        return updated

    def reset(self, shot_id: str) -> ShotRecord:
        """Clear a shot back to pending, keeping its pinned seed."""
        rec = self.record(shot_id)
        self.records[shot_id] = replace(
            rec,
            status=ShotStatus.PENDING,
            output="",
            error="",
            attempts=(),
        )
        self.save()
        return self.records[shot_id]

    # -- observability -----------------------------------------------------

    def summary(self, film: Film) -> str:
        """One line a human can act on, the way a batch job should report."""
        total = len(film.shots)
        done = self.done_count()
        skipped = sum(1 for r in self.records.values() if r.status is ShotStatus.SKIPPED)
        failed = sum(
            1
            for s in film.shots
            if self.record(s.id).exhausted(self.max_attempts)
        )
        inflight = sum(1 for r in self.records.values() if r.is_running)
        gpu_seconds = sum(r.seconds for r in self.records.values())
        elapsed = max(0.0, time.time() - self.run_started_at)
        return (
            f"film={self.film_id} shots={total} done={done} "
            f"failed={failed} skipped={skipped} running={inflight} "
            f"progress={self.progress(film) * 100:.1f}% "
            f"gpu={gpu_seconds / 60:.1f}min elapsed={elapsed / 60:.1f}min"
        )

    def failures(self) -> Iterator[tuple[str, ShotRecord]]:
        for shot_id, rec in sorted(self.records.items()):
            if rec.status is ShotStatus.FAILED:
                yield shot_id, rec
