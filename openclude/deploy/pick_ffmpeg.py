"""Pick the newest static ffmpeg build from the BtbN release feed.

A standalone script rather than an inline heredoc, because a heredoc inside a
single RUN instruction is not valid Dockerfile: the parser only joins lines that
end with a backslash, so the heredoc body is parsed as new instructions and the
build dies with "unknown instruction: );".

Called as:
    python3 pick_ffmpeg.py <releases-api-url>
prints the download URL on stdout and exits non-zero if nothing matches.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

# ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz
PATTERN = re.compile(r"^ffmpeg-n(?P<v>\d+\.\d+)-latest-linux64-gpl-(?P=v)\.tar\.xz$")


def pick(spec: dict) -> str:
    best: tuple[tuple[int, ...], dict] | None = None
    for asset in spec.get("assets", []) or []:
        name = asset.get("name", "")
        m = PATTERN.match(name)
        if not m:
            continue
        version = tuple(int(x) for x in m.group("v").split("."))
        if best is None or version > best[0]:
            best = (version, asset)
    if best is None:
        raise SystemExit(
            "no ffmpeg asset matched "
            f"{PATTERN.pattern!r}; BtbN may have changed its naming"
        )
    url = best[1].get("browser_download_url")
    if not url:
        raise SystemExit("matched asset has no browser_download_url")
    return url


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: pick_ffmpeg.py <releases-api-url>", file=sys.stderr)
        return 2
    req = urllib.request.Request(
        sys.argv[1], headers={"User-Agent": "openclude-docker-build"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        spec = json.load(r)
    print(pick(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
