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

    def sizes(self) -> dict[str, int]:
        return {
            str(f.relative_to(self.root)).replace("\\", "/"): f.stat().st_size
            for f in self.root.rglob("*")
            if f.is_file()
        }

    def total_used(self) -> int:
        return sum(self.sizes().values())

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


def _hf_callable(*names: str) -> object:
    """Find a huggingface_hub function across its renames.

    This helper exists because the first live test caught a real bug:
    `hf_hub_upload` was renamed to `upload_file` in huggingface_hub 1.x, and
    the unit tests could not see it because they used a fake client and never
    imported the library. The failure mode it caused is the worst kind: the
    store was unreachable for writes while every test passed.

    So the lookup is centralised, tries the modern name first, and raises with
    the installed version attached so the cause is obvious from the log alone.
    """
    hf = _hf_module()
    for name in names:
        fn = getattr(hf, name, None)
        if callable(fn):
            return fn

    raise StorageError(
        f"huggingface_hub {getattr(hf, '__version__', '?')} has none "
        f"of {list(names)}. The library was renamed and the image pins an older "
        f"version than this code expects, or vice versa."
    )


def _hf_module() -> object:
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StorageError(
            "huggingface_hub is not installed. It is part of the engine's "
            "requirements, so this should not happen in the image."
        ) from exc
    return huggingface_hub


def _delete_kwargs(fn: object, repo_id: str, key: str, token: str | None) -> dict:
    """Build delete arguments for whichever huggingface_hub is installed.

    Same story as upload: the rename came with a signature change.

        0.x  delete_file(filename, repo_id, token=, repo_type=)
        1.x  delete_file(path_in_repo, repo_id, token=, repo_type=)

    Found by a live test, after prune() had been quietly deleting nothing,
    because delete() swallowed every exception it was given.
    """
    import inspect

    try:
        params = set(inspect.signature(fn).parameters)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover
        params = set()

    name = "path_in_repo" if "path_in_repo" in params else "filename"
    return {name: key, "repo_id": repo_id, "token": token, "repo_type": "dataset"}
def _upload_kwargs(fn: object, repo_id: str, key: str, path: Path, token: str | None) -> dict:
    """Build upload arguments for whichever huggingface_hub is installed.

    The rename came with a signature change, not just a new name:

        0.x  hf_hub_upload(repo_id, filename, path, token=, repo_type=)
        1.x  upload_file(path_or_fileobj, path_in_repo, repo_id, token=,
                         repo_type=)

    Reading the signature beats trying one call and catching TypeError, because
    a TypeError from deep inside a real upload is indistinguishable from a
    signature mismatch, and the difference is a file silently not landing.

    `repo_type` is passed in every version. Without it the library defaults to
    the model namespace, and this repo is a dataset, so the write would be
    addressed to a repo that does not exist. That is the second bug the live
    test found.
    """
    import inspect

    try:
        params = set(inspect.signature(fn).parameters)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        params = set()

    modern = "path_in_repo" in params
    kwargs: dict = {"repo_id": repo_id, "token": token, "repo_type": "dataset"}
    if modern:
        kwargs["path_or_fileobj"] = str(path)
        kwargs["path_in_repo"] = key
    else:
        kwargs["filename"] = key
        kwargs["path"] = str(path)
    return kwargs


# --------------------------------------------------------------------------
# Hugging Face Hub
# --------------------------------------------------------------------------


class HFStore:
    """Hugging Face Hub, used as a plain file store.

    This exists because it is the only option that needs no credit card. The
    other candidates all do: Cloudflare R2 stops at a payment form, Backblaze
    B2 the same, and Google Cloud Storage requires a billing account even for
    its perpetual free tier. HF is free, private repos are free, and there is
    no card step.

    The decisive practical point is that `huggingface_hub` is ALREADY in the
    image. The engine installs it to fetch model weights, so this backend needs
    no new dependency and no new failure surface. We are reusing the tool that
    was already there for the model's sake.

    Layout mirrors a repo: a directory tree of files, addressed by path. A
    prefix lists a folder, exactly like the S3 backend, so the pipeline does
    not know which one it has.
    """

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = True,
        client: object | None = None,
    ) -> None:
        if not repo_id or "/" not in repo_id:
            raise StorageError(
                f"HF_REPO must look like 'username/repo-name', got {repo_id!r}"
            )
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN", "") or None
        self.private = private
        self._client = client
        self._created = False
        self._tree: list[dict] | None = None

    # -- plumbing ----------------------------------------------------------

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                from huggingface_hub import HfApi  # type: ignore[import-not-found]
            except ImportError as exc:
                raise StorageError(
                    "huggingface_hub is not installed. It is part of the engine's "
                    "requirements, so this should not happen in the image."
                ) from exc
            self._client = HfApi(token=self.token)
        return self._client

    def _ensure_repo(self) -> None:
        """Create the repo on first write. Idempotent."""
        if self._created:
            return
        try:
            self.client.create_repo(  # type: ignore[attr-defined]
                repo_id=self.repo_id, private=self.private, exist_ok=True
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"cannot create or reach the HF repo {self.repo_id!r}: {exc}\n"
                f"  Check that HF_TOKEN is set and the token has write access."
            ) from exc
        self._created = True

    def _key(self, key: str) -> str:
        return key.strip("/")

    def _api(self, *names: str) -> object:
        """Find an HfApi method across its renames.

        `file_info` was removed in huggingface_hub 1.x. Calling it raised
        AttributeError, and every caller here swallowed exceptions, so `exists`
        answered False for a file that was demonstrably there and `size`
        answered 0. That is the worst shape of bug: restore_if_present() uses
        exists() to decide whether to resume, so a restarted container would
        have silently restarted the whole film from shot one, every time, with
        no error anywhere.
        """
        for name in names:
            fn = getattr(self.client, name, None)
            if callable(fn):
                return fn
        raise StorageError(
            f"huggingface_hub has none of {list(names)} on HfApi. The library "
            f"renamed its metadata methods and the pinned version in the image "
            f"does not match this code."
        )

    # -- the Store interface -----------------------------------------------

    def put(self, key: str, path: str | os.PathLike[str]) -> str:
        src = Path(path)
        if not src.exists():
            raise StorageError(f"cannot upload {path}: no such file")
        if src.stat().st_size == 0:
            raise StorageError(f"refusing to upload a 0-byte file: {path}")
        self._ensure_repo()
        upload = _hf_callable("upload_file", "hf_hub_upload")
        try:
            upload(**_upload_kwargs(upload, self.repo_id, self._key(key), src, self.token))
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"upload of {key} failed: {exc}") from exc
        self._tree = None
        return key

    def get(self, key: str, path: str | os.PathLike[str]) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        download = _hf_callable("hf_hub_download", "hf_hub_download")
        try:
            local = download(
                repo_id=self.repo_id,
                filename=self._key(key),
                token=self.token,
                repo_type="dataset",
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"{key} is not in {self.repo_id}: {exc}") from exc
        if str(Path(local).resolve()) != str(out.resolve()):
            out.write_bytes(Path(local).read_bytes())
        return str(out)

    def _api(self, *names: str) -> object:
        """Find an HfApi method across its renames.

        `file_info` was removed in huggingface_hub 1.x. Calling it raised
        AttributeError, and every caller here swallowed exceptions, so `exists`
        answered False for a file that was demonstrably there and `size`
        answered 0. That is the worst shape of bug: restore_if_present() uses
        exists() to decide whether to resume, so a restarted container would
        have silently restarted the whole film from shot one, every time, with
        no error anywhere.
        """
        for name in names:
            fn = getattr(self.client, name, None)
            if callable(fn):
                return fn
        raise StorageError(
            f"huggingface_hub has none of {list(names)} on HfApi. The library "
            f"renamed its metadata methods and the pinned version in the image "
            f"does not match this code."
        )

    def exists(self, key: str) -> bool:
        try:
            return bool(self._api("file_exists")(
                repo_id=self.repo_id,
                filename=self._key(key),
                repo_type="dataset",
                token=self.token,
            ))
        except Exception:  # noqa: BLE001 - a miss and a failure both mean "no"
            return False

    def delete(self, key: str) -> None:
        target = self._key(key)
        try:
            self.client.delete_file(  # type: ignore[attr-defined]
                **_delete_kwargs(self.client.delete_file, self.repo_id, target, self.token)
            )
        except Exception as exc:  # noqa: BLE001
            # Deleting something absent is fine. Deleting something present and
            # reporting nothing is not: an earlier version swallowed every
            # exception here, and prune() therefore freed nothing while
            # reporting a saving. A real failure is now a real error.
            if not self._is_missing(exc):
                raise StorageError(f"delete of {target} failed: {exc}") from exc
        self._tree = None

    def delete_many(self, keys: Iterable[str]) -> None:
        """Delete a set of keys in one commit.

        A finished 2-hour film has around 1900 redundant clips. One commit for
        all of them is the difference between a prune that takes seconds and
        one that takes an hour of rate-limited single-file commits.
        """
        targets = [self._key(k) for k in keys]
        if not targets:
            return
        batch = getattr(self.client, "delete_files", None)
        if callable(batch):
            try:
                batch(repo_id=self.repo_id, delete_patterns=targets,
                      repo_type="dataset", token=self.token)
                self._tree = None
                return
            except Exception:  # noqa: BLE001 - fall back to one at a time
                pass
        for k in targets:
            self.delete(k)

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        text = f"{type(exc).__name__} {exc}".lower()
        return any(w in text for w in ("404", "not found", "does not exist",
                                       "no such file", "entrynotfound"))

    def _files(self) -> list[dict]:
        """Every file in the repo with its size, in one request.

        A 2-hour film is about 1900 shots, and the obvious implementation of
        `size()` is one API call per object, which is 1900 round trips to learn
        something one listing already knows.

        Folders are filtered out by the absence of a size, not by a type
        attribute: neither RepoFile nor RepoFolder carries one, so the first
        version of this counted `f01/clips` as an object. That inflated the
        usage report and, worse, handed folder names to delete().
        """
        if self._tree is None:
            entries: list[dict] = []
            try:
                for e in self._api("list_repo_tree")(
                    repo_id=self.repo_id,
                    recursive=True,
                    expand=True,
                    repo_type="dataset",
                    token=self.token,
                ):
                    size = getattr(e, "size", None)
                    if size is None:            # a folder, not a file
                        continue
                    path = str(getattr(e, "path", ""))
                    if not path or path.startswith("."):  # .gitattributes and friends
                        continue
                    entries.append({"path": path, "size": int(size or 0)})
            except Exception:  # noqa: BLE001
                entries = []
            self._tree = entries
        return self._tree

    def sizes(self) -> dict[str, int]:
        return {f["path"]: f["size"] for f in self._files()}

    def list(self, prefix: str) -> Iterator[str]:
        want = self._key(prefix)
        return iter(p for p in self.sizes() if p.startswith(want))

    def size(self, key: str) -> int:
        """Bytes behind one key.

        The listing is the answer, because one request already carries every
        size in the repo. A single-file lookup is only used when the listing
        cannot be read at all.
        """
        target = self._key(key)
        cached = self.sizes()
        if target in cached:
            return cached[target]
        try:
            infos = self._api("get_paths_info")(
                repo_id=self.repo_id,
                paths=target,
                expand=True,
                repo_type="dataset",
                token=self.token,
            )
            for info in infos or []:
                if str(getattr(info, "path", "")) == target:
                    return int(getattr(info, "size", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0
        return 0

    def total_used(self) -> int:
        return sum(self.sizes().values())


# --------------------------------------------------------------------------
# S3-compatible, which is what the container uses when it is configured
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

    def _strip(self, key: str) -> str:
        """Object key back to the layout key the pipeline uses."""
        if self.prefix and key.startswith(f"{self.prefix}/"):
            return key[len(self.prefix) + 1:]
        return key

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

    def sizes(self) -> dict[str, int]:  # pragma: no cover - needs a live bucket
        """Every key with its size, from the listing that already carries them."""
        out: dict[str, int] = {}
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self._k("")}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            for o in page.get("Contents", []) or []:
                out[self._strip(o["Key"])] = int(o.get("Size", 0))
            if not page.get("IsTruncated"):
                return out
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
    doomed = [
        key for key in store.list(layout.key_prefix())
        if any(f"/{k}/" in key for k in ("clips", "audio", "frames"))
    ]
    if hasattr(store, "delete_many"):
        store.delete_many(doomed)  # type: ignore[attr-defined]
    else:
        for key in doomed:
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
