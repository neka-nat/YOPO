import copy
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose

from yopo.structures import DetDataSample
from yopo.utils import get_test_pipeline_cfg

from .inference import init_detector

IntrinsicType = Union[Sequence[float], np.ndarray, torch.Tensor]
ImageType = Union[str, np.ndarray]

_POSE_ANNOTATION_TRANSFORMS = {
    'LoadPoseAnnotations',
    'Load9DPoseAnnotations',
    'yopo.LoadPoseAnnotations',
    'yopo.Load9DPoseAnnotations',
}


def build_pose_test_pipeline(
    cfg,
    from_ndarray: bool = True,
) -> Compose:
    """Build an inference-only test pipeline for monocular pose models.

    YOPO's released pose configs use evaluation pipelines that include
    annotation loading. This helper strips those transforms so the same config
    can be reused for arbitrary-image inference.
    """
    pipeline_cfg = copy.deepcopy(get_test_pipeline_cfg(cfg))
    pipeline_cfg = [
        step for step in pipeline_cfg
        if step.get('type') not in _POSE_ANNOTATION_TRANSFORMS
    ]

    if not pipeline_cfg:
        raise RuntimeError('The pose test pipeline is empty after filtering.')

    if from_ndarray:
        pipeline_cfg[0] = copy.deepcopy(pipeline_cfg[0])
        pipeline_cfg[0]['type'] = 'yopo.LoadImageFromNDArray'

    return Compose(pipeline_cfg)


def init_pose_detector(
    config: Union[str, Path],
    checkpoint: str,
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a YOPO pose detector and attach an inference pipeline."""
    model = init_detector(
        config=config,
        checkpoint=checkpoint,
        device=device,
        cfg_options=cfg_options,
    )
    model.pose_test_pipeline = build_pose_test_pipeline(
        model.cfg, from_ndarray=True)
    return model


def _normalize_intrinsics(
    imgs: Sequence[ImageType],
    intrinsic: Union[IntrinsicType, Sequence[IntrinsicType]],
) -> Sequence[IntrinsicType]:
    if not isinstance(imgs, (list, tuple)):
        raise TypeError('imgs must be a sequence here')

    if len(imgs) == 0:
        return []

    if isinstance(intrinsic, np.ndarray) and intrinsic.ndim in (1, 2):
        return [intrinsic] * len(imgs)
    if isinstance(intrinsic, torch.Tensor) and intrinsic.ndim in (1, 2):
        return [intrinsic] * len(imgs)
    if isinstance(intrinsic, (list, tuple)):
        if len(intrinsic) == 0:
            raise ValueError('intrinsic must not be empty')
        first = intrinsic[0]
        if isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
            if len(intrinsic) != len(imgs):
                raise ValueError(
                    'When passing per-image intrinsics, the number of '
                    'intrinsic entries must match the number of images.')
            return intrinsic
        return [intrinsic] * len(imgs)

    raise TypeError(
        'intrinsic must be a 4-value sequence, a 3x3 matrix, or a list of '
        'per-image intrinsics.')


def inference_pose_detector(
    model: nn.Module,
    imgs: Union[ImageType, Sequence[ImageType]],
    intrinsic: Union[IntrinsicType, Sequence[IntrinsicType]],
    test_pipeline: Optional[Compose] = None,
) -> Union[DetDataSample, Sequence[DetDataSample]]:
    """Run single-image or batched pose inference with explicit intrinsics."""
    if isinstance(imgs, (list, tuple)):
        is_batch = True
        img_list = list(imgs)
    else:
        is_batch = False
        img_list = [imgs]

    intrinsic_list = _normalize_intrinsics(img_list, intrinsic)

    if test_pipeline is None:
        use_ndarray = isinstance(img_list[0], np.ndarray)
        test_pipeline = build_pose_test_pipeline(
            model.cfg, from_ndarray=use_ndarray)

    result_list = []
    for img, single_intrinsic in zip(img_list, intrinsic_list):
        if isinstance(img, np.ndarray):
            data = dict(img=img, img_id=0, intrinsic=single_intrinsic)
        else:
            data = dict(img_path=img, img_id=0, intrinsic=single_intrinsic)

        data = test_pipeline(data)
        data['inputs'] = [data['inputs']]
        data['data_samples'] = [data['data_samples']]

        with torch.no_grad():
            result = model.test_step(data)[0]

        result_list.append(result)

    if is_batch:
        return result_list
    return result_list[0]


def pose_data_sample_to_dict(
    data_sample: DetDataSample,
    classes: Optional[Sequence[str]] = None,
    score_thr: float = 0.0,
) -> dict:
    """Convert YOPO pose outputs into a JSON-serializable structure."""
    result = {'detections': []}

    if hasattr(data_sample, 'img_path'):
        result['img_path'] = data_sample.img_path
    if hasattr(data_sample, 'img_shape'):
        result['img_shape'] = list(data_sample.img_shape)
    if hasattr(data_sample, 'ori_shape'):
        result['ori_shape'] = list(data_sample.ori_shape)
    if hasattr(data_sample, 'intrinsic'):
        intrinsic = data_sample.intrinsic
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.detach().cpu().numpy()
        if isinstance(intrinsic, np.ndarray):
            intrinsic = intrinsic.tolist()
        result['intrinsic'] = intrinsic

    if 'pred_instances' not in data_sample:
        return result

    pred_instances = data_sample.pred_instances
    if score_thr > 0 and 'scores' in pred_instances:
        pred_instances = pred_instances[pred_instances.scores >= score_thr]

    pred_instances = pred_instances.cpu().numpy()
    num_instances = len(pred_instances)

    for idx in range(num_instances):
        label = int(pred_instances.labels[idx])
        detection = {
            'label': label,
            'score': float(pred_instances.scores[idx]),
        }

        if classes is not None and 0 <= label < len(classes):
            detection['class_name'] = classes[label]
        if 'bboxes' in pred_instances:
            detection['bbox'] = pred_instances.bboxes[idx].tolist()
        if 'translations' in pred_instances:
            detection['translation'] = pred_instances.translations[idx].tolist()
        if 'sizes' in pred_instances:
            detection['size'] = pred_instances.sizes[idx].tolist()
        if 'centers_2d' in pred_instances:
            detection['center_2d'] = pred_instances.centers_2d[idx].tolist()
        if 'z' in pred_instances:
            z_value = np.asarray(pred_instances.z[idx]).reshape(-1)
            detection['z'] = float(z_value[0]) if z_value.size else None
        if 'rotations' in pred_instances:
            detection['rotation_raw'] = pred_instances.rotations[idx].tolist()
        if 'T' in pred_instances:
            detection['transform'] = pred_instances.T[idx].tolist()
            detection['rotation_matrix'] = pred_instances.T[idx][:3, :3].tolist()

        result['detections'].append(detection)

    return result
