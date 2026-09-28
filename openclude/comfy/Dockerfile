# ComfyUI on SaladCloud, for the user.

# One base image, no build steps. The official Salad image already has ComfyUI,
# the API wrapper, the web UI on 8188 and the model downloader. Everything the
# user needs is in it, and this file only adds the manifest that tells it which
# models to fetch at boot.
#
# Chosen after listing the live registry rather than trusting the documentation:
# the docs quote a newer-looking ComfyUI and API version pair that does not exist
# in the registry at all. The newest tag that actually exists and carries the API
# component is this one. tests/test_comfy_manifest.py asserts the documented
# version is not what is pinned here, so a future session does not copy it back
# in from the docs.
#
# runtime, not devel: devel carries the full CUDA toolkit and is about twice the
# size, which matters because a residential node downloads the whole image over
# its own internet before anything runs. Switch to -devel only if a video
# workflow needs to compile a custom node.

FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.3.43-api1.9.1-torch2.7.1-cuda12.8-runtime

# The model list. Read at startup, before the server accepts requests, so the
# readiness probe does not pass until the models are on disk and loadable.
COPY manifest.yaml /manifest.yaml
ENV MANIFEST=/manifest.yaml

# The web UI the user already knows, on the port ComfyUI listens on.
EXPOSE 8188

CMD ["./comfyui-api"]
