# ComfyUI on SaladCloud, for the user.
#
# NO MANIFEST. NO BAKED-IN MODELS. The container starts empty and fast, the
# user opens the interface and picks whatever they want, and only then does
# anything get downloaded.
#
# The first version of this file shipped a manifest.yaml with FLUX and Wan 2.2
# in it, about 30 GB. That was the wrong call and the user caught it: the
# container took minutes to become usable before anyone had chosen anything, and
# on a residential node that download is slow enough to look broken. It also
# pre-decided the models, which is the user's decision, not this file's.
#
# So: the image is the official Salad ComfyUI image and nothing else. The
# storage side of the project is `tools/model_store.py`, which mirrors whatever
# the user installs once the interface tells them what a template needs.
#
# Base tag read from the live registry, because the tag quoted in the
# SaladCloud documentation does not exist. See tests/test_comfy_manifest.py.

FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.3.43-api1.9.1-torch2.7.1-cuda12.8-runtime

# The web interface, on the port the access domain name forwards to.
EXPOSE 8188

CMD ["./comfyui-api"]
