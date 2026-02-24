# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import copy
import os
from typing import List, Optional, Sequence, Union
import json
import warnings

import cv2
from sklearn.neighbors import KDTree

import numpy as np
from mmcv.ops import batched_nms
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from plyfile import PlyData
import torch

from yopo.registry import METRICS
from ..functional import compute_add, compute_add_symmetric
from ..functional import eval_map

def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    # NOTE: Modified from https://github.com/ylabbe/cosypose/blob/c90a04f434b1e89f02341cc03899eb63ea8facba/cosypose/evaluation/meters/utils.py#L132
    errors = errors.copy()
    # convert errors from mm to m as in the original dataset
    errors = errors / 1000
    d = np.sort(errors)
    d[d > 0.1] = np.inf # 10cm
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap

@METRICS.register_module()
class YCBVideoMetric(BaseMetric):
    """YCB-Video evaluation metric.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'mAP', 'recall'. If is list, the first setting in the list will
             be used to evaluate metric.
        models_path (str): Path to the models used for evaluation.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'ycb_video'
    symmetric_objects = {
        0 : "002_master_chef_can" , 12 : "024_bowl" , 13 : "025_mug", 15 : "036_wood_block",
        17 : "040_large_marker", 18 : "051_large_clamp", 19 : "052_extra_large_clamp", 20 : "061_foam_brick" 
    }

    def __init__(self,
                 metric: Union[str, List[str]] = ['ADD(-S)', 'mAP', 'rotations', 'centers_2d', 'z'],
                 models_path: str = 'data/ycbv/models_eval',
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 nms_cfg: dict = dict(
                        type='nms', iou_threshold=0.5),
                 score_thr: float = 0.5,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['ADD(-S)', 'mssd', 'mspd', 'mAP', 'rotations',
                          'centers_2d', 'z'] 
    

        invalid_metrics = set(metric) - set(allowed_metrics)
        if invalid_metrics:
            raise KeyError(
                f"metric should be one of 'ADD(-S)', 'mssd', 'mspd', 'mAP', "
                f"'rotations', 'centers_2d', 'z', but got {invalid_metrics}.")
        self.metric = metric
        self.nms_cfg = nms_cfg
        self.score_thr = score_thr
        self.percentage = 0.1
        self.format_only = format_only
        self.load_models(models_path)
    
    def load_models(self, models_path: str) -> None:
        """Load models from the specified path.

        Args:
            models_path (str): Path to the models used for evaluation.
        """
        self.models_path = models_path
        self.models = {}
        for model in os.listdir(models_path):
            if model.endswith('.ply'):
                ply_data = PlyData.read(os.path.join(models_path, model))
                # strip the 'model_' prefix and '.ply' suffix
                model_id = int(model.split('.')[0][4:]) -1
                vertices = ply_data['vertex']
                points = np.stack([vertices[:]['x'],
                                   vertices[:]['y'],
                                   vertices[:]['z']], axis=-1).astype(np.float64)
                self.models[model_id] = points

        models_info_path = os.path.join(models_path, 'models_info.json')
        with open(models_info_path, 'r') as f:
            models_info = json.load(f)
        self.models_info = models_info

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']

            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                T=gt_instances['T'].cpu().numpy(),
                rotations=gt_instances['rotations'].cpu().numpy(),
                z=gt_instances['z'].cpu().numpy(),
                centers_2d=gt_instances['centers_2d'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())

            result = dict()
            pred = data_sample['pred_instances']

            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['T'] = pred['T'].cpu().numpy()
            result['rotations'] = pred['rotations'].cpu().numpy()
            result['z'] = pred['z'].cpu().numpy()
            result['centers_2d'] = pred['centers_2d'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['img_id'] = data_sample['img_id']

            self.results.append((ann, result))
    
    def results2csv(self, results: list) -> None:
        """Convert the results to csv format.

        Args:
            results (list): The processed results of each batch.
        """
        import csv

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Converting results to CSV format...')
        # row format:
        # scene_id, image_id, object_id, score, R, T, time
        csv_file_path = os.path.join(self.work_dir, 'ycbv_results.csv')
        with open(csv_file_path, 'w') as f:
            writer = csv.writer(f)
            for result in results:
                ann, pred = result
                image_id = ann['img_id']
                scene_id, image_id = image_id.split('_')
        

                pred_bboxes = pred['bboxes']
                if self.nms_cfg is not None:
                    pred_bboxes = torch.tensor(pred_bboxes).cuda()
                    pred_scores = torch.tensor(pred['scores']).cuda()
                    pred_labels = torch.tensor(pred['labels']).cuda()

                    _, keep_nms = batched_nms(
                        pred_bboxes, pred_scores, pred_labels,
                        self.nms_cfg)
                    keep_nms = keep_nms.cpu().numpy()
                if self.score_thr > 0:
                    keep_score = np.where(pred['scores'] > self.score_thr)[0]
                keep_idxs = np.intersect1d(keep_nms, keep_score)
                pred_labels = pred['labels'][keep_idxs]
                pred_T = pred['T'][keep_idxs]
                scores = pred['scores'][keep_idxs]

                for obj_id, score, T in zip(pred_labels, scores, pred_T):
                    time = -1
                    R = T[:3, :3]
                    t = T[:3, 3]
                    writer.writerow([scene_id, image_id, obj_id, score, R, t, time])

        logger.info(f'Results converted to {csv_file_path}')

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()
        for metric in self.metric:
            if metric == 'mAP':
                for iou_thr in [0.5]:
                    logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    # Follow the official implementation,
                    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                    # we should use the legacy coordinate system in yopo 1.x,
                    # which means w, h should be computed as 'x2 - x1 + 1` and
                    # `y2 - y1 + 1`
                    class_preds = []
                    for pred in preds:
                        tmp_dets = []
                        for label in range(len(self.dataset_meta['classes'])):
                            index = np.where(pred['labels'] == label)[0]
                            pred_bbox_scores = np.hstack(
                                [pred['bboxes'][index], pred['scores'][index].reshape((-1, 1))])
                            tmp_dets.append(pred_bbox_scores)
                        class_preds.append(tmp_dets)
                    mean_ap, _ = eval_map(
                        class_preds,
                        gts,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset=self.dataset_meta['classes'],
                        logger=logger,
                        eval_mode='area',
                        use_legacy_coordinate=True)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            if metric == 'ADD(-S)':
                errors_add = []
                errors_adds = []
                class_ids = []
                for gt, pred in zip(gts, preds):
                    gt_labels = gt['labels']
                    gt_T = gt['T']
                    gt_R = gt_T[:, :3, :3]
                    gt_t = gt_T[:, :3, 3]

                    pred_bboxes = pred['bboxes']
                    if self.nms_cfg is not None:
                        pred_bboxes = torch.tensor(pred_bboxes).cuda()
                        pred_scores = torch.tensor(pred['scores']).cuda()
                        pred_labels = torch.tensor(pred['labels']).cuda()

                        _, keep_nms = batched_nms(
                            pred_bboxes, pred_scores, pred_labels,
                            self.nms_cfg)
                        keep_nms = keep_nms.cpu().numpy()
                    if self.score_thr > 0:
                        keep_score = np.where(pred['scores'] > self.score_thr)[0]
                    keep_idxs = np.intersect1d(keep_nms, keep_score)
                    pred_labels = pred['labels'][keep_idxs]
                    pred_T = pred['T'][keep_idxs]
                    for i, gt_label in enumerate(gt_labels):
                        if sum(pred_labels==gt_label) == 1:
                            pred_T_i = pred_T[pred_labels == gt_label]
                            pred_R_i = pred_T_i[0, :3, :3]
                            pred_t_i = pred_T_i[0, :3, 3]

                            gt_R_i = gt_R[i]
                            gt_t_i = gt_t[i]
                            gt_points = self.models[gt_label]

                            pts_3d_gt = np.matmul(gt_R_i, gt_points.T).T + gt_t_i
                            pts_3d_pred = np.matmul(pred_R_i, gt_points.T).T + pred_t_i

                            # ADD-S
                            kdt = KDTree(pts_3d_gt, metric='euclidean')
                            distance, _ = kdt.query(pts_3d_pred, k=1)
                            error_adds = np.mean(distance)

                            # ADD(-S)
                            if gt_label not in self.symmetric_objects:
                                # ADD part
                                error = np.linalg.norm(pts_3d_gt - pts_3d_pred, axis=-1)
                                error_add = np.mean(error)
                            else:
                                # ADD-S part
                                error_add = error_adds
                        else:
                            error_add = 1e6
                            error_adds = 1e6

                        errors_add.append(error_add)
                        errors_adds.append(error_adds)
                        class_ids.append(gt_label)

                errors_add = np.array(errors_add)
                errors_adds = np.array(errors_adds)
                eval_results['AUC ADD(-S)'] = compute_auc_posecnn(errors_add)
                eval_results['AUC ADD-S'] = compute_auc_posecnn(errors_adds)

                class_ids = np.array(class_ids)
                class_ADD = dict()
                for class_id in range(len(self.models)):
                    num_instances = np.sum(np.array(class_ids) == class_id)
                    diameter_threshold = self.models_info[str(class_id+1)]['diameter']
                    if num_instances > 0:
                        # ADD
                        score = np.sum(errors_add[class_ids == class_id] < \
                                       self.percentage * diameter_threshold) / num_instances
                        class_ADD[class_id] = score
                    else:
                        class_ADD[class_id] = np.inf
                # ADD
                ADD_scores = np.array(list(class_ADD.values()))
                ADD_scores = ADD_scores[np.isfinite(ADD_scores)]
                eval_results['ADD(-S)-0.1d'] = np.mean(ADD_scores)

                for class_id in range(len(self.models)):
                    cls_name = self.dataset_meta['classes'][class_id]
                    if np.isfinite(class_ADD[class_id]):
                        eval_results[f'ADD(-S)-0.1d_{cls_name}'] = class_ADD[class_id]
            if metric == 'rotations':
                errors_r = []
                errors_r_angle = []
                errors_centers_2d = []
                errors_z = []

                matching_count = 0
                matching_total = 0
                # matching predictions with ground truth
                for i, gt in enumerate(gts):
                    gt_labels = gt['labels']
                    gt_T = gt['T']
                    gt_R = gt_T[:, :3, :3]
                    gt_t = gt_T[:, :3, 3]

                    pred_bboxes = preds[i]['bboxes']
                    if self.nms_cfg is not None:
                        pred_bboxes = torch.tensor(pred_bboxes).cuda()
                        pred_scores = torch.tensor(preds[i]['scores']).cuda()
                        pred_labels = torch.tensor(preds[i]['labels']).cuda()

                        _, keep_nms = batched_nms(
                            pred_bboxes, pred_scores, pred_labels,
                            self.nms_cfg)
                        keep_nms = keep_nms.cpu().numpy()
                    if self.score_thr > 0:
                        keep_score = np.where(preds[i]['scores'] > self.score_thr)[0]
                    keep_idxs = np.intersect1d(keep_nms, keep_score)
                    pred_labels = preds[i]['labels'][keep_idxs]
                    pred_T = preds[i]['T'][keep_idxs]

                    pred_centers_2d = preds[i]['centers_2d'][keep_idxs]
                    pred_z = preds[i]['z'][keep_idxs]
                    gt_centers_2d = gt['centers_2d']
                    gt_z = gt['z']


                    for j, gt_label in enumerate(gt_labels):
                        matching_total += 1
                        if sum(pred_labels==gt_label) == 1:
                            pred_T_j = pred_T[pred_labels == gt_label]
                            pred_R_j = pred_T_j[0, :3, :3]
                            pred_t_j = pred_T_j[0, :3, 3]

                            gt_R_j = gt_R[j]
                            gt_t_j = gt_t[j]

                            error = gt_R_j - pred_R_j
                            error_r = np.linalg.norm(error, ord='fro')

                            cos_theta = 1 - (error_r**2) / 4
                            cos_theta = np.clip(cos_theta, -1, 1)
                            theta = np.arccos(cos_theta)
                            errors_r_angle.append(np.rad2deg(theta))

                            # compute centers_2d error
                            gt_center_2d = gt_centers_2d[j]
                            pred_center_2d = pred_centers_2d[pred_labels == gt_label]
                            error_center_2d = np.abs(gt_center_2d - pred_center_2d)
                            error_center_2d = np.mean(error_center_2d)
                            # compute z error
                            gt_z_j = np.exp(gt_z[j])
                            pred_z_j = np.exp(pred_z[pred_labels == gt_label])
                            # gt_z_j = gt_z[j]
                            # pred_z_j = pred_z[pred_labels == gt_label]
                            error_z = np.abs(gt_z_j - pred_z_j)
                            error_z = np.mean(error_z)

                            matching_count += 1

                            errors_r.append(error_r)
                            errors_centers_2d.append(error_center_2d)
                            errors_z.append(error_z)
                        else:
                            pass
                            # print(f'pred_labels: {sorted(pred_labels)}')
                            # print(f'gt_labels: {sorted(gt_labels)}')
                eval_results['matching_rate'] = matching_count/matching_total 
                eval_results['rotations'] = np.mean(errors_r)
                eval_results['angle_errors'] = np.mean(errors_r_angle)
                eval_results['centers_2d'] = np.mean(errors_centers_2d)
                eval_results['z'] = np.mean(errors_z)
        return eval_results

@METRICS.register_module()
class LMOMetric(BaseMetric):
    """LINEMOD-Occulded evaluation metric.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'mAP', 'recall'. If is list, the first setting in the list will
             be used to evaluate metric.
        models_path (str): Path to the models used for evaluation.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'linemod_occluded'
    symmetric_objects = {
        0 : "002_master_chef_can" , 12 : "024_bowl" , 13 : "025_mug", 15 : "036_wood_block",
        17 : "040_large_marker", 18 : "051_large_clamp", 19 : "052_extra_large_clamp", 20 : "061_foam_brick" 
    }
    class_to_name_orig = {
                     0: "ape",  4: "can", 5: "cat", 7: "driller",
                     8: "duck", 9: "eggbox", 10: "glue", 11: "holepuncher",
                    }
    class_to_name = {                               #class map used for training
                    0: "ape",  1: "can", 2: "cat", 3: "driller",
                    4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher",
                    }
    class_map = {                            #class map used for training
                 0: 0, 4: 1, 5: 2,
                 7: 3, 8: 4, 9: 5,
                 10: 6, 11: 7,
                 }

    def __init__(self,
                 metric: Union[str, List[str]] = ['ADD', 'mAP', 'rotations', 'centers_2d', 'z'],
                 models_path: str = 'data/lmo/models_eval',
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 nms_cfg: dict = dict(
                        type='nms', iou_threshold=0.5),
                 score_thr: float = 0.5,
                 p: float = 0.1,
                 prefix: Optional[str] = None) -> None:
        super().__init__(metric=metric,
                        models_path=models_path,
                        format_only=format_only,
                        nms_cfg=nms_cfg,
                        score_thr=score_thr,
                        p=p,
                        collect_device=collect_device, prefix=prefix)
