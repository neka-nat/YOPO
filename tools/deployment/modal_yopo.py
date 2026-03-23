"""Modal deployment scaffold for YOPO monocular pose inference.

This module assumes one deployed endpoint serves one fixed checkpoint/config
pair. If you need multiple models, deploy separate Modal apps or duplicate the
class with a different ``MODEL_KEY``.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import modal

APP_NAME = "yopo-pose"


def _resolve_repo_local_root() -> Path:
    module_path = Path(__file__).resolve()
    if (
        len(module_path.parents) >= 3
        and module_path.parent.name == "deployment"
        and module_path.parent.parent.name == "tools"
    ):
        return module_path.parents[2]
    return Path.cwd()


REPO_LOCAL_ROOT = _resolve_repo_local_root()
REPO_ROOT = Path("/root/YOPO")
MODEL_ROOT = Path("/models")
MODEL_VOLUME = modal.Volume.from_name("yopo-models", create_if_missing=True)

MODEL_SPECS = {
    "nocs-r50": {
        "config": "configs/yopo/nocs_yopo_real_camera_r50.py",
        "checkpoint": (
            "https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/"
            "nocs_yopo_real_camera_r50.pth"
        ),
    },
    "nocs-swinl": {
        "config": "configs/yopo/nocs_yopo_real_camera_swinl.py",
        "checkpoint": (
            "https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/"
            "nocs_yopo_real_camera_swinl.pth"
        ),
    },
    "nocs-swinl-finetune-real": {
        "config": "configs/yopo/nocs_yopo_real_camera_swinl_finetune_real.py",
        "checkpoint": (
            "https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/"
            "nocs_yopo_real_camera_swinl_finetune_real.pth"
        ),
    },
    "housecat6d-swinl": {
        "config": "configs/yopo/housecat6d_yopo_swinl.py",
        "checkpoint": (
            "https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/"
            "housecat6d_yopo_swinl.pth"
        ),
    },
}

# Keep one deployment bound to one model/checkpoint pair.
MODEL_KEY = os.environ.get("YOPO_MODEL_KEY", "nocs-r50")
REQUIRES_PROXY_AUTH = os.environ.get("YOPO_PROXY_AUTH", "0").lower() in {
    "1",
    "true",
    "yes",
}

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel")
    .apt_install(
        "ffmpeg",
        "git",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxcb1",
        "libxrender1",
    )
    .add_local_dir(
        local_path=str(REPO_LOCAL_ROOT),
        remote_path=str(REPO_ROOT),
        copy=True,
        ignore=[".git", "data", "work_dirs", "__pycache__", "*.pth", "*.pt"],
    )
    .workdir(str(REPO_ROOT))
    .run_commands("pip install -U pip setuptools wheel openmim mmengine")
    .run_commands("mim install mmcv==2.2.0")
    .run_commands("pip install -r requirements/runtime.txt")
    .run_commands("pip install plyfile fastapi[standard] scikit-learn")
)

app = modal.App(APP_NAME)


def _normalize_intrinsic(value: Any) -> Any:
    if value is None:
        raise ValueError(
            "intrinsic is required. Pass either [fx, fy, cx, cy] or a 3x3 matrix."
        )

    if isinstance(value, dict):
        keys = ("fx", "fy", "cx", "cy")
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"intrinsic dict is missing keys: {missing}")
        return [float(value["fx"]), float(value["fy"]),
                float(value["cx"]), float(value["cy"])]

    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            return [float(v) for v in value]
        if len(value) == 3 and all(isinstance(row, (list, tuple)) for row in value):
            return [[float(v) for v in row] for row in value]
        if len(value) == 9:
            return [float(v) for v in value]

    raise ValueError(
        "intrinsic must be [fx, fy, cx, cy], a flat 9-value matrix, "
        "or a nested 3x3 list."
    )


@app.cls(
    image=image,
    env={"YOPO_MODEL_KEY": MODEL_KEY},
    gpu="A10G",
    volumes={str(MODEL_ROOT): MODEL_VOLUME},
    scaledown_window=300,
)
class YOPOPoseService:

    @modal.enter()
    def load(self) -> None:
        import mmcv

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from yopo.apis import (build_pose_test_pipeline, init_pose_detector,
                               pose_data_sample_to_dict)

        if MODEL_KEY not in MODEL_SPECS:
            raise KeyError(
                f"Unknown MODEL_KEY={MODEL_KEY!r}. Available keys: {sorted(MODEL_SPECS)}"
            )

        self._decode_image = mmcv.imfrombytes
        self._serialize = pose_data_sample_to_dict

        spec = MODEL_SPECS[MODEL_KEY]
        checkpoint_url = spec["checkpoint"]
        checkpoint_name = checkpoint_url.rsplit("/", 1)[-1]
        checkpoint_path = MODEL_ROOT / checkpoint_name
        print(f"YOPO model_key={MODEL_KEY} checkpoint={checkpoint_name}")

        MODEL_VOLUME.reload()
        if not checkpoint_path.exists():
            print(f"Downloading checkpoint to {checkpoint_path}")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            MODEL_VOLUME.commit()
            print(f"Committed checkpoint {checkpoint_name} to volume")

        self.model = init_pose_detector(
            config=str(REPO_ROOT / spec["config"]),
            checkpoint=str(checkpoint_path),
            device="cuda:0",
        )
        self.test_pipeline = build_pose_test_pipeline(
            self.model.cfg, from_ndarray=True)
        self.model_key = MODEL_KEY
        self.classes = tuple(self.model.dataset_meta.get("classes", ()))

    def _predict_impl(self, payload: dict) -> dict:
        from yopo.apis import inference_pose_detector

        image_b64 = payload.get("image_base64")
        if not image_b64:
            raise ValueError("image_base64 is required")

        image_bytes = base64.b64decode(image_b64)
        image = self._decode_image(image_bytes)
        if image is None:
            raise ValueError("Failed to decode image_base64 into an image")

        intrinsic = _normalize_intrinsic(payload.get("intrinsic"))
        score_thr = float(payload.get("score_thr", 0.2))

        data_sample = inference_pose_detector(
            model=self.model,
            imgs=image,
            intrinsic=intrinsic,
            test_pipeline=self.test_pipeline,
        )
        result = self._serialize(
            data_sample=data_sample,
            classes=self.classes,
            score_thr=score_thr,
        )
        result["model_key"] = self.model_key
        return result

    @modal.method()
    def predict(self, payload: dict) -> dict:
        return self._predict_impl(payload)

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
        requires_proxy_auth=REQUIRES_PROXY_AUTH,
    )
    def predict_web(self, payload: dict) -> dict:
        return self._predict_impl(payload)


@app.local_entrypoint()
def main(
    image_path: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    score_thr: float = 0.2,
) -> None:
    image_bytes = Path(image_path).read_bytes()
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "intrinsic": [fx, fy, cx, cy],
        "score_thr": score_thr,
    }
    result = YOPOPoseService().predict.remote(payload)
    print(json.dumps(result, indent=2))
