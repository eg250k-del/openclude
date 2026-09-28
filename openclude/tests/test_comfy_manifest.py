"""Every model URL in the manifest must resolve.

The first manifest was written from the SaladCloud documentation's example
paths. Three of the four were wrong: two 404'd, and the repository they named,
`Comfy-Org/Wan_2.2_TI2V-5B_ComfyUI_repackaged`, does not exist. Nothing catches
that until a rented GPU sits on a residential connection downloading 16 GB and
then fails to find the model, an hour later, for a dollar.

So the URLs are checked here, in CI, with a HEAD request. A 404 fails the build
instead of the run.

The test skips when the network is unavailable, because a test that fails
because someone is offline trains people to ignore it.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "comfy" / "manifest.yaml"
DOCKERFILE = Path(__file__).resolve().parents[1] / "comfy" / "Dockerfile"


def _urls() -> list[str]:
    text = MANIFEST.read_text("utf-8")
    return re.findall(r"url:\s*(\S+)", text)


def _reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_the_manifest_exists() -> None:
    assert MANIFEST.exists(), "the manifest is how the container knows what to fetch"


def test_there_are_models() -> None:
    assert _urls(), "no model URLs at all"


def test_every_url_is_https() -> None:
    for url in _urls():
        assert url.startswith("https://"), url


def test_every_url_is_free_to_download() -> None:
    """No Civitai, because that needs a key the user does not have and a card
    they do not have either. Everything here must be a plain public fetch."""
    for url in _urls():
        assert "civitai.com" not in url, (
            f"{url} needs Civitai credentials, which do not exist here"
        )


def test_every_model_has_a_destination() -> None:
    text = MANIFEST.read_text("utf-8")
    urls = len(_urls())
    paths = len(re.findall(r"local_path:\s*(\S+)", text))
    assert paths == urls, f"{urls} urls but {paths} destinations"


def test_every_destination_is_inside_the_comfy_models_directory() -> None:
    """A typo like /opt/ComcludeUI downloads 16 GB into a directory nothing reads."""
    for path in re.findall(r"local_path:\s*(\S+)", MANIFEST.read_text("utf-8")):
        assert path.startswith("/opt/ComfyUI/models/"), path


def test_no_two_models_share_a_destination() -> None:
    paths = re.findall(r"local_path:\s*(\S+)", MANIFEST.read_text("utf-8"))
    assert len(paths) == len(set(paths)), "two models would overwrite each other"


def test_no_two_models_share_a_url() -> None:
    urls = _urls()
    assert len(urls) == len(set(urls))


def test_the_image_and_video_families_are_both_present() -> None:
    """The user wants both, and a manifest with only images would look like
    success right up until they tried to make a video."""
    text = MANIFEST.read_text("utf-8")
    assert "flux" in text.lower(), "no image model"
    assert "wan" in text.lower(), "no video model"


def test_the_video_model_is_quantised() -> None:
    """fp16 Wan 2.2 is about 10 GB and the text encoder another 11, which does
    not fit alongside anything on a 24 GB card."""
    text = MANIFEST.read_text("utf-8").lower()
    assert ".gguf" in text, "the video model is not quantised, so it will not fit"


# --------------------------------------------------------------------------
# the base image
# --------------------------------------------------------------------------


def test_the_base_image_is_the_official_salad_one() -> None:
    text = DOCKERFILE.read_text("utf-8")
    assert "FROM ghcr.io/saladtechnologies/comfyui-api:" in text


def test_the_base_image_is_not_pinned_to_a_tag_that_does_not_exist() -> None:
    """The SaladCloud documentation quotes comfy0.7.0-api1.16.1-*, which 404s.

    That was found by listing the live tags rather than trusting the docs, and
    it is asserted here so a future session does not copy the documented tag
    back in.
    """
    text = DOCKERFILE.read_text("utf-8")
    assert "comfy0.7.0" not in text, (
        "that tag is from the documentation and does not exist in the registry"
    )
    assert "api1." in text, "pick a tag that includes the API component"


def test_the_container_exposes_the_web_ui_port() -> None:
    """8188 is the ComfyUI interface, the part the user actually wants to see."""
    assert "8188" in DOCKERFILE.read_text("utf-8")


def test_the_manifest_is_copied_in_and_named() -> None:
    text = DOCKERFILE.read_text("utf-8")
    assert "COPY manifest.yaml" in text
    assert "MANIFEST" in text


# --------------------------------------------------------------------------
# the network check
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _reachable(), reason="no network")
@pytest.mark.parametrize("url", _urls())
def test_the_model_url_resolves(url: str) -> None:
    """A 404 here is a run that costs money and produces nothing."""
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "openclude-manifest-check/0.1")
    try:
        r = urllib.request.urlopen(req, timeout=45)
    except urllib.error.HTTPError as e:
        pytest.fail(f"{url} returned HTTP {e.code}; a rented node will fail the same way")
    assert r.status == 200, url
