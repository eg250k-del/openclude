"""Blob storage, because the container dies and its disk with it.

The SaladCloud docs are explicit: container filesystems are ephemeral, volume
mounting is not supported because containers are unprivileged, and node
failover is normal. So the state file, the narration and every finished clip
have to be somewhere else the moment they exist.

The rule this module enforces: nothing is ever only local. Every write is
followed by a read-back check, because an upload that silently failed is
indistinguishable from a successful one until the film is already lost.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

CHUNK = 8 * 1024 * 1024


class StorageError(RuntimeError):
    pass


class UploadUnconfirmed(StorageError):
    """The write appeared to succeed but the object is not there. Fatal."""


class Store(Protocol):
    def put(self, key: str, path: str | os.PathLike[str]) -> str: ...
    def get(self, key: str, path: str | os.PathLike[str]) -> str: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> Iterator[str]: ...


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """One film, two coordinate systems, kept apart on purpose.

    `key_*` returns an object-store KEY: relative, slash-separated, rooted at
    the film. `path_*` returns a LOCAL FILESYSTEM PATH: absolute, so it
    survives the engine calling `os.chdir()` on init.

    Conflating these is how the audited repos ended up writing a film's output
    somewhere nobody looked, so the naming makes the distinction explicit.
    """

    film_id: str
    base: "str | os.PathLike[str]" = "."

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", Path(self.base).resolve())

    # -- object store keys (relative) --------------------------------------

    def key_state(self) -> str:
        return f"{self.film_id}/state/ledger.json"

    def key_script(self) -> str:
        return f"{self.film_id}/script/script.json"

    def key_audio(self, shot_id: str) -> str:
        return f"{self.film_id}/audio/{shot_id}.wav"

    def key_frame(self, shot_id: str) -> str:
        return f"{self.film_id}/frames/{shot_id}_last.png"

    def key_clip(self, shot_id: str) -> str:
        return f"{self.film_id}/clips/{shot_id}.mp4"

    def key_film(self) -> str:
        return f"{self.film_id}/film/final.mp4"

    def key_prefix(self) -> str:
        return f"{self.film_id}/"

    # -- local filesystem paths (absolute) ---------------------------------

    def path(self, relative: str) -> str:
        return str(self.base / relative)

    def path_state(self) -> str:
        return self.path(self.key_state())

    def path_script(self) -> str:
        return self.path(self.key_script())

    def path_audio(self, shot_id: str) -> str:
        return self.path(self.key_audio(shot_id))

    def path_frame(self, shot_id: str) -> str:
        return self.path(self.key_frame(shot_id))

    def path_clip(self, shot_id: str) -> str:
        return self.path(self.key_clip(shot_id))

    def path_voiced(self, shot_id: str) -> str:
        return str(self.base / self.film_id / "clips" / f"{shot_id}_voiced.mp4")

    def path_film(self) -> str:
        return self.path(self.key_film())


# --------------------------------------------------------------------------
# local, for tests and for a single-machine run
# --------------------------------------------------------------------------


class LocalStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = self.root / key
        # refuse to escape the root via a crafted key
        try:
            p.resolve().relative_to(self.root.resolve())
        except ValueError:
            raise StorageError(f"key {key!r} escapes the store root") from None
        return p

    def put(self, key: str, path: str | os.PathLike[str]) -> str:
        src = Path(path)
        if not src.exists():
            raise StorageError(f"cannot upload {path}: no such file")
        if src.stat().st_size == 0:
            raise StorageError(f"refusing to upload a 0-byte file: {path}")
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return key

    def get(self, key: str, path: str | os.PathLike[str]) -> str:
        src = self._p(key)
        if not src.exists():
            raise StorageError(f"{key} is not in the store")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        return str(out)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def size(self, key: str) -> int:
        p = self._p(key)
        return p.stat().st_size if p.exists() else 0

    def total_used(self) -> int:
        total = 0
        for f in self.root.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)

    def list(self, prefix: str) -> Iterator[str]:
        base = self._p(prefix.rstrip("/"))
        if not base.exists():
            return iter(())
        return (
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in sorted(base.rglob("*"))
            if p.is_file()
        )


# --------------------------------------------------------------------------
# S3-compatible, which is what the container actually uses
# --------------------------------------------------------------------------


class S3Store:
    """Cloudflare R2 or anything else with an S3 API.

    R2 specifically because Salad's own docs recommend it: no egress fees, and
    the nodes are scattered globally so egress is the thing that would
    otherwise quietly cost more than the GPU.
    """

    def __init__(
        self,
        bucket: str,
        client: object | None = None,
        prefix: str = "",
    ) -> None:
        if client is None:
            client = self._build_client()
        self.bucket = bucket
        self.client = client
        self.prefix = prefix.strip("/")

    @staticmethod
    def _build_client():  # pragma: no cover - needs credentials
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise StorageError(
                "boto3 is required for S3 storage. Install it in the image."
            ) from exc
        endpoint = os.environ.get("S3_ENDPOINT", "").strip()
        return boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            region_name=os.environ.get("S3_REGION", "auto"),
        )

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, path: str | os.PathLike[str]) -> str:
        src = Path(path)
        if not src.exists():
            raise StorageError(f"cannot upload {path}: no such file")
        if src.stat().st_size == 0:
            raise StorageError(f"refusing to upload a 0-byte file: {path}")
        self.client.upload_file(str(src), self.bucket, self._k(key))
        if not self.exists(key):  # pragma: no cover - needs a live bucket
            raise UploadUnconfirmed(
                f"uploaded {key} but it is not readable back. "
                f"Treating the upload as failed."
            )
        return key

    def get(self, key: str, path: str | os.PathLike[str]) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._k(key), str(out))
        if not out.exists() or out.stat().st_size == 0:
            raise StorageError(f"downloaded {key} but got nothing usable")
        return str(out)

    def exists(self, key: str) -> bool:  # pragma: no cover - needs a live bucket
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except Exception:  # noqa: BLE001 - head_object raises a client error
            return False

    def size(self, key: str) -> int:  # pragma: no cover - needs a live bucket
        """Bytes behind one key. Used by the storage doctor."""
        head = self.client.head_object(Bucket=self.bucket, Key=self._k(key))
        return int(head.get("ContentLength", 0))

    def total_used(self) -> int:  # pragma: no cover - needs a live bucket
        """Total bytes in the bucket. One pass, so the doctor is cheap."""
        total = 0
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self._k("")}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            total += sum(int(o.get("Size", 0)) for o in page.get("Contents", []) or [])
            if not page.get("IsTruncated"):
                return total
            token = page.get("NextContinuationToken")

    def delete(self, key: str) -> None:  # pragma: no cover
        self.client.delete_object(Bucket=self.bucket, Key=self._k(key))

    def list(self, prefix: str) -> Iterator[str]:  # pragma: no cover
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self._k(prefix)}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            for obj in page.get("Contents", []) or []:
                k = obj["Key"]
                yield k[len(self.prefix) + 1:] if self.prefix else k
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")


# --------------------------------------------------------------------------
# the sync points the pipeline uses
# --------------------------------------------------------------------------


def put_checked(store: Store, key: str, path: str | os.PathLike[str]) -> str:
    """Upload, then confirm. Every call site uses this, never raw `put`."""
    store.put(key, path)
    if not store.exists(key):
        raise UploadUnconfirmed(f"{key} did not survive the write")
    return key


def restore_if_present(
    store: Store, key: str, path: str | os.PathLike[str]
) -> str | None:
    """Fetch a file if it exists upstream. The container's first move on boot."""
    if not store.exists(key):
        return None
    return store.get(key, path)


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def prunable(
    store: Store, layout: Layout
) -> dict[str, int]:
    """What can be deleted, and how much it frees.

    A film is only prunable when its final mp4 exists, because the whole point
    of pruning is to reclaim space from things that are already redundant. A
    half-finished film has clips that are the ONLY copy of the work, and
    deleting those is how you lose a film you paid to render.

    The ledger and the final film are never touched: the ledger is kilobytes
    and is what makes a re-run possible.
    """
    freed = {"clips": 0, "audio": 0, "frames": 0, "objects": 0}
    if not store.exists(layout.key_film()):
        return freed
    for key in store.list(layout.key_prefix()):
        size = store.size(key) if hasattr(store, "size") else 0
        for kind in ("clips", "audio", "frames"):
            if f"/{kind}/" in key:
                freed[kind] += size
                freed["objects"] += 1
    return freed


def prune(store: Store, layout: Layout) -> dict[str, int]:
    """Delete the clips, narration and continuity frames of finished films.

    Never touches the final mp4 or the ledger. Idempotent: a second call finds
    nothing left to do.
    """
    freed = prunable(store, layout)
    if not freed["objects"]:
        return freed
    for key in list(store.list(layout.key_prefix())):
        if any(f"/{k}/" in key for k in ("clips", "audio", "frames")):
            store.delete(key)
    return freed


def film_progress(store: Store, layout: Layout) -> dict[str, int]:
    """Count what is already banked, so a restart knows where it stands."""
    counts = {"audio": 0, "frames": 0, "clips": 0, "other": 0}
    for key in store.list(layout.key_prefix()):
        if "/audio/" in key:
            counts["audio"] += 1
        elif "/frames/" in key:
            counts["frames"] += 1
        elif "/clips/" in key:
            counts["clips"] += 1
        else:
            counts["other"] += 1
    return counts
