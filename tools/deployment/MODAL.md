# YOPO on Modal

This repository now includes a Modal deployment scaffold at `tools/deployment/modal_yopo.py` for monocular pose inference.

## Why this shape

The released YOPO pose configs use evaluation pipelines that include `Load9DPoseAnnotations`. Arbitrary-image serving therefore needs a separate inference path. The new `yopo.apis.pose_inference` helper removes annotation loading, injects camera intrinsics, and serializes pose outputs for HTTP responses.

The Modal app keeps one model loaded per container:

- `@app.cls(...)` holds the GPU worker
- `@modal.enter()` loads the checkpoint once
- a Modal `Volume` caches checkpoints under `/models`
- `predict_web` exposes a `POST` endpoint
- `predict` supports Python-to-Python calls

## Prerequisites

```bash
pip install modal
modal setup
```

If you want local rendering from the JSON response, also install:

```bash
pip install numpy opencv-python
```

Optional environment variables:

```bash
export YOPO_MODEL_KEY=nocs-r50
export YOPO_PROXY_AUTH=1
```

Available `YOPO_MODEL_KEY` values:

- `nocs-r50`
- `nocs-swinl`
- `nocs-swinl-finetune-real`
- `housecat6d-swinl`

## Local smoke test

```bash
modal run tools/deployment/modal_yopo.py \
  --image-path path/to/image.png \
  --fx 577.5 --fy 577.5 --cx 319.5 --cy 239.5
```

This sends the image to the remote Modal class and prints the serialized pose predictions as JSON.

## Serve and deploy

Temporary dev endpoint:

```bash
modal serve tools/deployment/modal_yopo.py
```

Persistent deployment:

```bash
modal deploy tools/deployment/modal_yopo.py
```

After `modal serve` or `modal deploy`, you can send a request with:

```bash
python tools/deployment/modal_client.py \
  https://<your-endpoint>.modal.run \
  path/to/image.png \
  --fx 577.5 --fy 577.5 --cx 319.5 --cy 239.5 \
  --json-out result.json \
  --render-out result.png
```

The lightweight client renders with only `opencv-python` and `numpy`. It draws:

- 2D bounding boxes and class labels
- Pose axes
- Size-based 3D cuboids

## HTTP request format

`predict_web` expects JSON:

```json
{
  "image_base64": "<base64-encoded image bytes>",
  "intrinsic": [577.5, 577.5, 319.5, 239.5],
  "score_thr": 0.2
}
```

`intrinsic` is required because YOPO reconstructs translation from the predicted 2D center and depth. If you later want file upload instead of base64 JSON, move from `fastapi_endpoint` to a small `FastAPI` app with `@modal.asgi_app`.
