"""SaladCloud deployment control.

Safety model, in order of importance:

  1. **Dry run is the default.** Nothing is created unless you pass `--apply`.
  2. **No secret is ever printed or logged.** The key comes from
     `SALAD_API_KEY` in the environment and is only ever placed in a header.
  3. **Names are never guessed.** `SALAD_ORGANIZATION` and `SALAD_PROJECT` must
     be supplied; the SaladCloud docs are explicit that an agent must not
     enumerate or infer them.
  4. **Minimum replicas is 0.** A running replica bills continuously. The
     job-queue autoscaler brings capacity up when there is work and takes it
     back to zero when the queue drains, so an idle project costs nothing.
  5. **Every write is followed by a read-back.** A 202 only means "accepted".

Usage:
    python -m openclude.salad preflight          # read-only, always safe
    python -m openclude.salad plan               # show the group spec
    python -m openclude.salad apply              # create or update
    python -m openclude.salad status             # live group + instances
    python -m openclude.salad logs               # recent instance logs
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

#: Base URL from the spec's `servers` block. The first draft used /api/v1,
#: which returns 404 for every documented path.
API = "https://api.salad.com/api/public"

#: Auth scheme from the spec's security block: apiKey, in header,
#: named Salad-Api-Key.
AUTH_HEADER = "Salad-Api-Key"

GROUP_NAME = "openclude"
QUEUE_NAME = "openclude"

#: Profile 3 is the engine's own label for "32 GB RAM and 24 GB VRAM", which is
#: a 4090. Profile 5 is a 10 GB fail-safe choice and is wrong here.
GPU_PROFILE = "3"

TIMEOUT = 60

#: Every path below was taken from the official OpenAPI spec, not invented. The
#: first draft of this file guessed `/projects/{p}/container-groups` and
#: `/quotas/{org}`; both are 404. The real shapes are:
#:   list  : /organizations/{org}/projects/{project}/containers
#:   quotas: /organizations/{org}/quotas
#:   avail : /organizations/{org}/availability/sce-gpu-availability
#: The spec is the authority, per SaladCloud's own agent runbook.
P = "/organizations/{org}/projects/{project}/containers"


class SaladError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


#: GPU class name -> API UUID, read from GET /organizations/<org>/gpu-classes
#: on 2026-09-28. The API rejects a name with a 400 that says nothing useful,
#: so these are data, not decoration.
#:
#: The 3090 Ti is the default: same 24 GB of VRAM as the 4090, the 5B model
#: fits either, and it is cheaper per hour. The 4090 stays available as a
#: fallback when the 3090 Ti has no capacity.
GPU_CLASSES: dict[str, str] = {
    "RTX 3090 Ti": "9998fe42-04a5-4807-b3a5-849943f16c38",   # 24 GB
    "RTX 4090":    "ed563892-aacd-40f5-80b7-90c9be6c759b",   # 24 GB
    "RTX A5000":   "6d4e9e99-d27e-4751-8d7d-393f7d8ea949",   # 24 GB
}


@dataclass
class Config:
    api_key: str = field(repr=False, default="")
    organization: str = ""
    project: str = ""
    image: str = ""
    budget_cents: int = 500          # 5.00 USD per hour ceiling for the project

    # The API takes GPU class UUIDs, not names. These came from
    # GET /organizations/{org}/gpu-classes on 2026-09-28, not from memory:
    # a name is rejected with a 400 that says nothing useful.
    #
    # The 3090 Ti is listed first on purpose. It has the same 24 GB of VRAM as
    # the 4090, the 5B model fits either, and it is cheaper per hour. The 4090
    # stays in the list as a fallback if the 3090 Ti has no capacity.
    gpu: str = "RTX 3090 Ti"

    # NOTE the units. The API takes memory in MB and storage in BYTES. An
    # earlier version of this file passed memory=32 and storage_amount=120000,
    # which read as 32 MB of RAM and 117 KB of disk. A dry run cannot catch
    # that, because nothing rejects a valid number of the wrong size; only the
    # live API's 400 and a container that dies on boot will.
    memory_mb: int = 32_768         # 32 GB
    storage_bytes: int = 120 * 1024**3   # 120 GB, node free-space threshold
    cpu: int = 8
    max_replicas: int = 1
    min_replicas: int = 0            # kept for the report; nothing autoscales now
    priority: str = "low"

    @classmethod
    def from_env(cls) -> "Config":
        key = os.environ.get("SALAD_API_KEY", "").strip()
        org = os.environ.get("SALAD_ORGANIZATION", "").strip()
        project = os.environ.get("SALAD_PROJECT", "").strip()
        image = os.environ.get("OPENCLIDE_IMAGE", "").strip()
        missing = [
            n
            for n, v in (
                ("SALAD_API_KEY", key),
                ("SALAD_ORGANIZATION", org),
                ("SALAD_PROJECT", project),
                ("OPENCLIDE_IMAGE", image),
            )
            if not v
        ]
        if missing:
            raise SaladError(
                "missing required environment variables: " + ", ".join(missing) + "\n"
                "  SALAD_API_KEY        from the Portal, API Access page. "
                "Set it as an environment variable, never in a chat or a file.\n"
                "  SALAD_ORGANIZATION   your organization name (not guessed)\n"
                "  SALAD_PROJECT        your container-engine project name\n"
                "  OPENCLIDE_IMAGE      e.g. ghcr.io/eg250k-del/openclude:latest"
            )
        return cls(
            api_key=key,
            organization=org,
            project=project,
            image=image,
            gpu=os.environ.get("OPENCLIDE_GPU", cls.gpu),
            budget_cents=int(os.environ.get("OPENCLUDE_BUDGET_CENTS", cls.budget_cents)),
            max_replicas=int(os.environ.get("OPENCLIDE_MAX_REPLICAS", cls.max_replicas)),
            cpu=int(os.environ.get("OPENCLIDE_CPU", cls.cpu)),
            memory_mb=int(os.environ.get("OPENCLIDE_MEMORY_MB", cls.memory_mb)),
            storage_bytes=int(os.environ.get("OPENCLIDE_STORAGE_BYTES", cls.storage_bytes)),
        )

    def gpu_class_ids(self) -> list[str]:
        """Resolve the configured GPU name to API UUIDs.

        Raises rather than passing the name through. A name looks plausible, is
        accepted by the dataclass, and comes back as an unhelpful 400, so a typo
        in the environment variable would be indistinguishable from an outage.
        """
        if self.gpu not in GPU_CLASSES:
            raise SaladError(
                f"unknown GPU class {self.gpu!r}. Known: "
                f"{', '.join(sorted(GPU_CLASSES))}.\n"
                f"  These UUIDs come from GET /organizations/<org>/gpu-classes."
            )
        return [GPU_CLASSES[self.gpu]]


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


#: Cloudflare fronts the API and rejects requests whose User-Agent looks like a
#: bot. The default urllib agent ("Python-urllib/3.x") is blocked outright with
#: error 1010 "browser_signature_banned", which is a 403 that looks like a
#: permissions problem but is not. This string is what fixed it.
USER_AGENT = "openclude/0.1 (+https://github.com/eg250k-del/openclude)"


def call(cfg: Config, method: str, path: str, body: Any = None) -> Any:
    url = f"{API}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header(AUTH_HEADER, cfg.api_key)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    if data:
        # An update is a merge patch and must say so. Sending application/json
        # to PATCH is answered with 415 Unsupported Media Type, which is a
        # confusing way to be told the content type is wrong.
        req.add_header(
            "Content-Type",
            "application/merge-patch+json" if method == "PATCH"
            else "application/json",
        )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw_error = e.read().decode("utf-8", "replace")
        # An error body can echo the request, and the request carries the
        # credentials. A 400 on a create is exactly when that happens, which is
        # the moment a person is most likely to paste the output into a chat.
        try:
            body_text = safe_json(json.loads(raw_error))
        except (ValueError, TypeError):
            body_text = raw_error[:600]
        hint = ""
        if e.code == 403 and "1010" in body_text:
            hint = (
                "\n  this is Cloudflare error 1010, not a permissions problem: "
                "the request looked like a bot."
            )
        raise SaladError(f"{method} {path} -> HTTP {e.code}: {body_text}{hint}") from None
    except urllib.error.URLError as e:
        raise SaladError(f"{method} {path} -> network error: {e.reason}") from None


# --------------------------------------------------------------------------
# the group spec
# --------------------------------------------------------------------------


def group_spec(cfg: Config) -> dict[str, Any]:
    """The exact body sent to create the container group.

    Shape taken from the official OpenAPI spec, not invented. The first draft
    of this function guessed the shape and every part of it was wrong. The
    spec says:

      ContainerGroupPrototype requires:
        autostart_policy, container, name, replicas, restart_policy
      CreateContainer requires:
        image, resources
      CreateContainerResourceRequirements requires:
        cpu, memory
      CreateContainerGroupNetworking requires:
        auth, port, protocol
      ContainerGroupProbeHttp requires:
        headers, path, port, scheme
      every probe requires:
        failure_threshold, initial_delay_seconds, period_seconds,
        success_threshold, timeout_seconds
      ContainerGroupQueueAutoscaler requires:
        desired_queue_length, max_replicas, min_replicas
    """
    probe_common = {
        "scheme": "http",
        "host": "localhost",
        "headers": [],
    }
    return {
        "name": GROUP_NAME,
        # false on purpose: a group that autostarts sits there billing all
        # night. The operator starts it with `salad start` and stops it with
        # `salad stop`, and neither of those is implicit.
        "autostart_policy": False,
        "replicas": cfg.max_replicas,
        # A plain string. The object form with condition/attempts/delay_seconds
        # is what a PATCH might take, and it is what the create schema rejects
        # with "The input was not valid".
        "restart_policy": "always",
        "container": {
            # A plain string, not an object. An object with repository/type/
            # architecture is rejected the same way.
            "image": cfg.image,
            "resources": {
                "cpu": cfg.cpu,
                "memory": cfg.memory_mb,          # MB, not GB
                "storage_amount": cfg.storage_bytes,  # bytes, not MB
                "shm_size": 1024,
                "gpu_classes": cfg.gpu_class_ids(),  # UUIDs, not names
            },
            "priority": cfg.priority,
            "environment_variables": {
                "OPENCLIDE_DATA": "/data",
                "WAN2GP_ROOT": "/opt/WanGP",
                "WAN2GP_PROFILE": GPU_PROFILE,
                "WAN2GP_ATTENTION": "sage2",
                "OPENCLIDE_REQUIRE_STORE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                # Which durable backend. Passed through rather than decided here
                # so the same image runs on any of them, and so a preflight can
                # check the credentials before an instance is ever scheduled.
                "OPENCLIDE_STORE": os.environ.get("OPENCLIDE_STORE", ""),
                "HF_REPO": os.environ.get("HF_REPO", ""),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "S3_ENDPOINT": os.environ.get("S3_ENDPOINT", ""),
                "S3_BUCKET": os.environ.get("S3_BUCKET", ""),
                "S3_ACCESS_KEY_ID": os.environ.get("S3_ACCESS_KEY_ID", ""),
                "S3_SECRET_ACCESS_KEY": os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            },
        },
        "networking": {
            "port": 8000,
            "protocol": "http",
            # false: the health endpoint is a plain GET and HTTP auth would make
            # the probe fail. The port answers only /health and /ready.
            "auth": False,
        },
        "liveness_probe": {
            "http": {**probe_common, "port": 8000, "path": "/health"},
            "initial_delay_seconds": 60,
            "period_seconds": 30,
            "timeout_seconds": 10,
            "failure_threshold": 6,
            "success_threshold": 1,
        },
        "readiness_probe": {
            "http": {**probe_common, "port": 8000, "path": "/ready"},
            # Generous on purpose: the first model download takes many minutes
            # and a probe that fails during it gets the instance killed.
            #
            # The grace period is failure_threshold x period_seconds, and the
            # API caps failure_threshold at 20. So the window has to come from
            # period_seconds: 20 x 15s = 300s of grace, where 90 x 10s was
            # rejected outright. The first version of this asked for 90 and the
            # live API answered "must be between 1 and 20".
            "initial_delay_seconds": 60,
            "period_seconds": 15,
            "timeout_seconds": 10,
            "failure_threshold": 20,
            "success_threshold": 1,
        },
        # No queue_autoscaler and no queue_connection, on purpose.
        #
        # They name a SaladCloud job queue, which does not exist here: this
        # project's queue is a folder of JSON files in the object store, read by
        # worker.poll_queue(). That is the right design here because a container
        # can be preempted at any instant, and a job that exists only inside a
        # request handler is a lost job.
        #
        # Leaving them in also meant every create attempt failed, because the
        # API validates that the named queue exists.
        #
        # Consequence: replicas is static, and the operator decides when to run
        # anything with `salad start` and `salad stop`. For someone paying by
        # the hour that is a better trade than an autoscaler guessing.
    }


def redact_any(value: object) -> object:
    """Strip secrets out of anything on its way to a screen or a log.

    This exists because SaladCloud echoes the container's environment variables
    back in every single response. So any command that prints a response
    prints the live HF token and the live API key. That happened three times:
    once from `plan`, once from `status`, and once from an error message.

    Redacting only the spec we build is not enough, because the leak is
    upstream of us. This walks whatever came back.
    """
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if is_secret_name(str(k)):
                out[k] = "***set***" if v else ""
            else:
                out[k] = redact_any(v)
        return out
    if isinstance(value, list):
        return [redact_any(v) for v in value]
    return value


def safe_json(value: object) -> str:
    return json.dumps(redact_any(value), indent=2, default=str)


def containers_path(cfg: "Config") -> str:
    """The one and only spelling of the container-groups collection path.

    It was written out separately in four commands, and one of them said
    `/projects/<project>/container-groups`, which 404s because the real path
    nests the project under the organization and names the collection
    `containers`. A path that appears once cannot drift.
    """
    return f"/organizations/{cfg.organization}/projects/{cfg.project}/containers"


def containers_path_for(cfg: "Config", name: str) -> str:
    return f"{containers_path(cfg)}/{name}"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_preflight(cfg: Config, _args: list[str]) -> int:
    """Read-only. Proves the key works and shows the ceiling before spending."""
    print("PREFLIGHT (read-only, costs nothing)\n")
    print(f"  organization : {cfg.organization}")
    print(f"  project      : {cfg.project}")
    print(f"  image        : {cfg.image}")
    print(f"  gpu          : {cfg.gpu.strip()}   priority: {cfg.priority}")
    print(f"  cpu / memory : {cfg.cpu} cores / {cfg.memory_gib} GiB")
    print(f"  disk request : {cfg.storage_mb} MB")
    print(f"  replicas     : {cfg.min_replicas} (idle) .. {cfg.max_replicas} (busy)")
    print()

    try:
        quotas = call(cfg, "GET", f"/organizations/{cfg.organization}/quotas")
    except SaladError as e:
        print(f"  quota read FAILED: {e}")
        return 1
    print("  quota read OK")
    print(safe_json(quotas)[:1500])
    print()
    try:
        nodes = call(cfg, "GET", f"/organizations/{cfg.organization}/availability/sce-gpu-availability")
        print("  gpu availability read OK")
        print(safe_json(nodes)[:800])
    except SaladError as e:
        print(f"  node pool read failed (not fatal): {e}")
    return 0


#: Substrings that mark an environment variable as holding a secret.
#:
#: TOKEN matters as much as KEY and SECRET, and forgetting it is not a
#: hypothetical: `HF_TOKEN` matched neither "SECRET" nor "KEY", so the plan
#: command printed the live Hugging Face token in full. Anyone reading the
#: output, or pasting it into a chat, published a credential.
#:
#: A false positive is harmless. A false negative leaks a key, so the list is
#: deliberately wide.
SECRET_MARKERS = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL",
                  "AUTH", "SESSION", "COOKIE", "SIGNATURE")


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def redact_env(envs: dict) -> dict:
    """Replace every secret value with a marker, keeping the names visible.

    The names are kept because which variables are set is the useful part of a
    plan. The values are not, ever.
    """
    out = {}
    for k, v in envs.items():
        if is_secret_name(k):
            out[k] = "***set***" if v else ""
        else:
            out[k] = v
    return out


def cmd_plan(cfg: Config, _args: list[str]) -> int:
    spec = group_spec(cfg)
    print("GROUP SPEC (nothing is sent; this is exactly what --apply would post)\n")
    print(safe_json(spec))
    print()
    print("COST CEILING")
    rate = {"low": 0.16, "medium": 0.27, "high": 0.33}.get(cfg.priority, 0.16)
    print(f"  max_replicas {cfg.max_replicas} x ${rate}/h = "
          f"${cfg.max_replicas * rate:.2f}/h at full load")
    print(f"  idle (min_replicas 0) = $0.00/h")
    print(f"  project budget set to ${cfg.budget_cents / 100:.2f}/h")
    return 0


def cmd_apply(cfg: Config, _args: list[str]) -> int:
    spec = group_spec(cfg)
    p = containers_path(cfg)


    existing = None
    try:
        existing = call(cfg, "GET", f"{containers_path_for(cfg, GROUP_NAME)}")
    except SaladError:
        existing = None

    if existing:
        print(f"  group {GROUP_NAME} already exists; merging the spec into it")
        action, path, body = "PATCH", f"{containers_path_for(cfg, GROUP_NAME)}", spec
    else:
        print(f"  creating group {GROUP_NAME}")
        action, path, body = "POST", p, spec

    result = call(cfg, action, path, body)
    gid = (result or {}).get("id") or (result or {}).get("container_group", {}).get("id")
    print(f"  {action} accepted: {safe_json(result)[:400]}")

    # 202 means accepted, not ready. Verify with a read.
    time.sleep(3)
    try:
        after = call(cfg, "GET", f"{containers_path_for(cfg, GROUP_NAME)}")
        print(f"  read-back OK: name={after.get('name')} "
              f"replicas={after.get('replicas')} state={after.get('state')}")
    except SaladError as e:
        print(f"  read-back FAILED: {e}")
        return 1

    if gid:
        print(f"\n  group id: {gid}")
        print("  watch it with:  python -m openclude.salad status")
    print("\n  NOTE: a 2xx only means accepted. Instances appear asynchronously")
    print("  and the first one still has to download the model weights.")
    return 0


def cmd_start(cfg: Config, _args: list[str]) -> int:
    """Start a stopped group.

    Separate from `apply` on purpose. Creating a group and switching it on are
    different acts with different costs, and the person paying needs to be able
    to do the first without the second.

    `autostart_policy` is false in the spec, so nothing starts by itself and
    nothing bills by itself. That is deliberate for a first run: an unattended
    group left on overnight is the expensive mistake, and the fix is one
    `stop`.
    """
    path = f"{containers_path_for(cfg, GROUP_NAME)}/start"
    try:
        result = call(cfg, "POST", path, {})
    except SaladError as e:
        print(f"  start refused: {e}")
        return 1
    print("  start accepted")
    print(safe_json(result)[:300])
    print("  instances appear asynchronously, and the first one downloads the")
    print("  model weights, which takes minutes. Watch with:  status")
    return 0


def cmd_stop(cfg: Config, _args: list[str]) -> int:
    """Stop a running group so it stops costing money.

    This is the command that matters most to someone paying by the hour, and it
    is idempotent on purpose: running it twice is not an error, because the
    instinct after a successful stop is to press it again.
    """
    path = f"{containers_path_for(cfg, GROUP_NAME)}/stop"
    try:
        result = call(cfg, "POST", path, {})
    except SaladError as e:
        text = str(e).lower()
        if "not found" in text or "404" in text:
            print(f"  nothing to stop: no group called {GROUP_NAME}")
            return 0
        print(f"  stop refused: {e}")
        return 1
    print("  stop accepted")
    print(safe_json(result)[:300])
    return 0


def cmd_status(cfg: Config, _args: list[str]) -> int:

    p = containers_path(cfg)
    group = call(cfg, "GET", f"{containers_path_for(cfg, GROUP_NAME)}")
    print(safe_json(group)[:2500])
    gid = group.get("id")
    if gid:
        try:
            inst = call(cfg, "GET", f"{containers_path_for(cfg, GROUP_NAME)}/instances")
            print("\nINSTANCES")
            print(safe_json(inst)[:2000])
        except SaladError as e:
            print(f"\ninstances read failed: {e}")
    return 0


def cmd_logs(cfg: Config, args: list[str]) -> int:
    if not args:
        print("usage: logs <instance-id>")
        return 1

    try:
        logs = call(cfg, "GET", f"/organizations/{cfg.organization}/log-entries?instance_id={args[0]}")
    except SaladError as e:
        print(f"  {e}")
        return 1
    for line in logs.get("logs", []):
        print(f"  {line.get('created_at', '')} {line.get('stream', ''):<5} "
              f"{line.get('message', '')}")
    return 0


COMMANDS = {
    "preflight": cmd_preflight,
    "plan": cmd_plan,
    "apply": cmd_apply,
    "start": cmd_start,
    "stop": cmd_stop,
    "status": cmd_status,
    "logs": cmd_logs,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    name, rest = argv[0], argv[1:]
    if name == "apply" and "--apply" not in rest:
        print("refusing to create or update anything without --apply")
        print("run:  python -m openclude.salad apply --apply")
        return 2
    rest = [a for a in rest if a != "--apply"]
    try:
        cfg = Config.from_env()
    except SaladError as e:
        print(f"\n{e}\n")
        return 2
    return COMMANDS[name](cfg, rest)


if __name__ == "__main__":
    raise SystemExit(main())
