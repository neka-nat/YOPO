# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import copy
import os
from typing import List, Optional, Sequence, Union
import json
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

import numpy as np
from mmcv.ops import batched_nms
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from plyfile import PlyData
import torch

from yopo.registry import METRICS
from ..functional import eval_map


@METRICS.register_module()
class NOCSMetric(BaseMetric):
    """NOCS evaluation metric.

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

    def __init__(
        self,
        format_only: bool = False,
        collect_device: str = "cpu",
        nms_cfg: dict = dict(type="nms", iou_threshold=0.5),
        score_thr: float = 0.2,
        dump_results_path: Optional[str] = None,
        dump_format: str = "pickle",  # "pickle" or "json"
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.nms_cfg = nms_cfg
        self.score_thr = score_thr
        self.format_only = format_only
        self.dump_results_path = dump_results_path
        assert dump_format in ["pickle", "json"]
        self.dump_format = dump_format
        self.sequence_data = {}  # For collecting JSON data per sequence
        if self.dump_results_path:
            os.makedirs(self.dump_results_path, exist_ok=True)

    def dump_results(self, result: dict, data_sample: dict):
        """Dump predictions to a pickle or JSON file."""
        img_path = data_sample["img_path"]
        parts = img_path.split("/")
        scene_id = parts[-2]
        frame_id = os.path.splitext(parts[-1])[0]

        # Apply score threshold filtering for dumping
        if self.score_thr > 0:
            valid_indices = np.where(result['scores'] > self.score_thr)[0]
            if len(valid_indices) > 0:
                filtered_result = {
                    "bboxes": result['bboxes'][valid_indices, :],
                    "scores": result['scores'][valid_indices],
                    "labels": result['labels'][valid_indices],
                    "translations": result['translations'][valid_indices, :],
                    "rotations": result['rotations'][valid_indices, :],
                    "sizes": result['sizes'][valid_indices, :],
                    "T": result['T'][valid_indices, :]
                }
            else:
                # No predictions above threshold, create empty arrays
                filtered_result = {
                    "bboxes": np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "labels": np.zeros(0, dtype=np.int32),
                    "translations": np.zeros((0, 3)),
                    "rotations": np.zeros((0, 3, 3)),
                    "sizes": np.zeros((0, 3)),
                    "T": np.zeros((0, 4, 4))
                }
        else:
            filtered_result = result

        if self.dump_format == "pickle":
            # Original pickle format
            dump_dir = os.path.join(self.dump_results_path, scene_id)
            os.makedirs(dump_dir, exist_ok=True)

            file_path = os.path.join(dump_dir, f"{frame_id}.pkl")

            dump_data = {
                "pred_class_ids": filtered_result["labels"],
                "pred_bboxes": filtered_result["bboxes"],
                "pred_scores": filtered_result["scores"],
                "pred_RTs": filtered_result["T"],
                "pred_scales": filtered_result["sizes"],
            }

            with open(file_path, "wb") as f:
                pickle.dump(dump_data, f)
                
        elif self.dump_format == "json":
            # JSON format - collect data for the sequence
            if scene_id not in self.sequence_data:
                self.sequence_data[scene_id] = {
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {"id": i+1, "name": name} 
                        for i, name in enumerate(self.dataset_meta["classes"])
                    ]
                }
            
            # Get image info
            img_height, img_width = data_sample.get("img_shape", (480, 640))  # Default values
            
            # Extract camera intrinsics if available
            intrinsic = data_sample['intrinsic']
            if len(intrinsic) == 9:
                fx = intrinsic[0]
                fy = intrinsic[4]
                cx = intrinsic[2]
                cy = intrinsic[5]
            elif len(intrinsic) == 4:
                fx = intrinsic[0]
                fy = intrinsic[1]
                cx = intrinsic[2]
                cy = intrinsic[3]
       
            # Create image entry
            image_id = len(self.sequence_data[scene_id]["images"]) + 1
            image_entry = {
                "id": image_id,
                "file_name": f"{frame_id}.png",
                "width": img_width,
                "height": img_height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
            self.sequence_data[scene_id]["images"].append(image_entry)
            
            # Create annotation entries (only for filtered predictions)
            num_predictions = len(filtered_result["labels"])
            for i in range(num_predictions):
                annotation_id = len(self.sequence_data[scene_id]["annotations"]) + 1
                
                # Extract rotation matrix from transformation matrix
                T_matrix = filtered_result["T"][i]  # 4x4 transformation matrix
                translation = T_matrix[:3, 3].tolist()
                rotation_matrix = T_matrix[:3, :3]
                rot_scale = np.cbrt(np.linalg.det(rotation_matrix))
                rotation_matrix = (rotation_matrix / rot_scale).flatten().tolist()
                sizes = (filtered_result["sizes"][i] * rot_scale).tolist()
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(filtered_result["labels"][i]) + 1,  # Convert to 1-based indexing
                    "translation": translation,
                    "rotation_matrix": rotation_matrix,
                    "size": sizes,
                    "bbox": filtered_result["bboxes"][i].tolist(),
                    "score": float(filtered_result["scores"][i])  # Add prediction score
                }
                self.sequence_data[scene_id]["annotations"].append(annotation_entry)
                
    def finalize_json_dumps(self):
        """Write collected JSON data for all sequences."""
        if self.dump_format == "json" and self.sequence_data:
            for scene_id, data in self.sequence_data.items():
                json_path = os.path.join(self.dump_results_path, f"{scene_id}.json")
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
            print(f"JSON files saved for {len(self.sequence_data)} sequences in {self.dump_results_path}")

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
            gt_instances = gt["gt_instances"]
            gt_ignore_instances = gt["ignored_instances"]

            ann = dict(
                labels=gt_instances["labels"].cpu().numpy(),
                bboxes=gt_instances["bboxes"].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances["bboxes"].cpu().numpy(),
                labels_ignore=gt_ignore_instances["labels"].cpu().numpy(),
                translations=gt_instances["translations"].cpu().numpy(),
                rotations=gt_instances["rotations"].cpu().numpy(),
                sizes=gt_instances["sizes"].cpu().numpy(),
                T=gt_instances["T"].cpu().numpy(),
            )

            if "gt_handle_visibility" in gt_instances:
                ann["gt_handle_visibility"] = (
                    gt_instances["gt_handle_visibility"].cpu().numpy()
                )

            result = dict()
            pred = data_sample["pred_instances"]

            # if self.nms_cfg is not None:
            #     pred_bboxes = pred['bboxes']
            #     pred_scores = pred['scores']
            #     pred_labels = pred['labels']

            #     _, keep_nms = batched_nms(
            #         pred_bboxes, pred_scores, pred_labels,
            #         self.nms_cfg)
            #     keep_nms = keep_nms.cpu().numpy()
            # if self.score_thr > 0:
            #     keep_score = torch.where(pred['scores'] > self.score_thr)[0].cpu().numpy()
            # else:
            #     keep_score = np.arange(len(pred['scores']))
            # keep_idxs = np.intersect1d(keep_nms, keep_score)

            result["bboxes"] = pred["bboxes"].cpu().numpy()
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()

            result["translations"] = pred["translations"].cpu().numpy()
            result["rotations"] = pred["rotations"].cpu().numpy()
            result["sizes"] = pred["sizes"].cpu().numpy()
            result["T"] = pred["T"].cpu().numpy()

            # if keep_idxs.size > 0:
            #     result['bboxes'] = result['bboxes'][keep_idxs]
            #     result['scores'] = result['scores'][keep_idxs]
            #     result['labels'] = result['labels'][keep_idxs]
            #     result['translations'] = result['translations'][keep_idxs]
            #     result['rotations'] = result['rotations'][keep_idxs]
            #     result['sizes'] = result['sizes'][keep_idxs]
            #     result['T'] = result['T'][keep_idxs]

            # valid_indices = np.where(result['scores'] > self.score_thr)[0]
            # if len(valid_indices):
            #     result['bboxes'] = result['bboxes'][valid_indices, :]
            #     result['scores'] = result['scores'][valid_indices]
            #     result['labels'] = result['labels'][valid_indices]
            #     result['translations'] = result['translations'][valid_indices, :]
            #     result['rotations'] = result['rotations'][valid_indices, :]
            #     result['sizes'] = result['sizes'][valid_indices, :]
            #     result['T'] = result['T'][valid_indices, :]

            # gt_scale = np.linalg.norm(ann['sizes'], axis=1)
            pred_scale = np.linalg.norm(result["sizes"], axis=1)
            result["sizes"] = result["sizes"] / pred_scale[:, None]
            # ann['sizes'] = ann['sizes'] / gt_scale[:, None]
            # # gt_R = ann['T'][:, :3, :3] * gt_scale[:, None, None]
            pred_R = result["T"][:, :3, :3] * pred_scale[:, None, None]
            # # ann['T'][:, :3, :3] = gt_R
            result["T"][:, :3, :3] = pred_R

            if self.dump_results_path:
                self.dump_results(result, data_sample)

            self.results.append((ann, result))

    def compute_independent_mAP(
        self,
        preds,
        gts,
        degree_thresholds=[5, 10],
        shift_thresholds=[2, 5, 10],
        iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75],
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
        logger=None,
        cat_id=-1,
        classes=None,
        num_workers=None,
    ):
        if num_workers is None:
            num_workers = min(
                cpu_count(), 8
            )  # Limit to 8 processes to avoid memory issues

        total_images = len(preds)
        num_classes = len(classes)
        degree_thres_list = list(degree_thresholds) + [360]
        num_degree_thres = len(degree_thres_list)

        shift_thres_list = list(shift_thresholds) + [100]
        num_shift_thres = len(shift_thres_list)

        iou_thres_list = list(iou_3d_thresholds)
        num_iou_thres = len(iou_thres_list)

        if use_matches_for_pose:
            assert iou_pose_thres in iou_thres_list

        iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
        iou_pred_matches_all = [
            np.zeros((num_iou_thres, 0)) for _ in range(num_classes)
        ]
        iou_pred_scores_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
        iou_gt_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]

        pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
        pose_pred_matches_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]
        pose_gt_matches_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]
        pose_pred_scores_all = [
            np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
        ]

        # Create smaller batches for more frequent progress updates
        # Use smaller batch size to get more granular progress feedback
        images_per_batch = max(
            1, min(10, total_images // (num_workers * 4))
        )  # 4x more batches than workers
        total_batches = (
            total_images + images_per_batch - 1
        ) // images_per_batch  # Ceiling division

        batches = [
            (preds[i : i + images_per_batch], gts[i : i + images_per_batch])
            for i in range(0, len(preds), images_per_batch)
        ]

        # Prepare worker function arguments
        worker_args = [
            (
                batch_preds,
                batch_gts,
                num_classes,
                classes,
                iou_thres_list,
                degree_thres_list,
                shift_thres_list,
                use_matches_for_pose,
                iou_pose_thres,
            )
            for batch_preds, batch_gts in batches
        ]

        print(
            f"Processing {total_images} images across {len(batches)} batches ({images_per_batch} images per batch) with {num_workers} workers..."
        )

        # Process batches with image-level progress tracking
        if num_workers > 1 and len(batches) > 1:
            with Pool(num_workers) as pool:
                # Create a progress bar that tracks the total number of images
                with tqdm(
                    total=total_images, desc="Processing images", unit="img"
                ) as pbar:
                    batch_results = []
                    for i, batch_result in enumerate(
                        pool.imap(_process_batch_worker, worker_args)
                    ):
                        batch_results.append(batch_result)
                        # Update progress by the actual size of the current batch
                        current_batch_size = len(batches[i][0])  # batch_preds length
                        pbar.update(current_batch_size)
        else:
            # Fallback to sequential processing with image-level progress
            batch_results = []
            with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
                for i, args in enumerate(worker_args):
                    batch_result = _process_batch_worker(args)
                    batch_results.append(batch_result)
                    # Update progress by the actual size of the current batch
                    current_batch_size = len(batches[i][0])  # batch_preds length
                    pbar.update(current_batch_size)

        # Aggregate results from all batches
        print("Aggregating results from all batches...")
        for batch_result in tqdm(
            batch_results, desc="Aggregating batches", unit="batch"
        ):
            (
                batch_iou_pred_matches,
                batch_iou_pred_scores,
                batch_iou_gt_matches,
                batch_pose_pred_matches,
                batch_pose_pred_scores,
                batch_pose_gt_matches,
            ) = batch_result

            for cls_id in range(num_classes):
                iou_pred_matches_all[cls_id] = np.concatenate(
                    (iou_pred_matches_all[cls_id], batch_iou_pred_matches[cls_id]),
                    axis=-1,
                )
                iou_pred_scores_all[cls_id] = np.concatenate(
                    (iou_pred_scores_all[cls_id], batch_iou_pred_scores[cls_id]),
                    axis=-1,
                )
                iou_gt_matches_all[cls_id] = np.concatenate(
                    (iou_gt_matches_all[cls_id], batch_iou_gt_matches[cls_id]), axis=-1
                )

                pose_pred_matches_all[cls_id] = np.concatenate(
                    (pose_pred_matches_all[cls_id], batch_pose_pred_matches[cls_id]),
                    axis=-1,
                )
                pose_pred_scores_all[cls_id] = np.concatenate(
                    (pose_pred_scores_all[cls_id], batch_pose_pred_scores[cls_id]),
                    axis=-1,
                )
                pose_gt_matches_all[cls_id] = np.concatenate(
                    (pose_gt_matches_all[cls_id], batch_pose_gt_matches[cls_id]),
                    axis=-1,
                )

        # Compute AP scores (this part remains sequential as it's already fast)
        print("Computing IoU AP scores...")
        iou_dict = {}
        iou_dict["thres_list"] = iou_thres_list
        for cls_id in tqdm(range(num_classes), desc="Computing IoU APs", unit="class"):
            class_name = classes[cls_id]
            for s, iou_thres in enumerate(iou_thres_list):
                iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(
                    iou_pred_matches_all[cls_id][s, :],
                    iou_pred_scores_all[cls_id][s, :],
                    iou_gt_matches_all[cls_id][s, :],
                )

        iou_3d_aps[-1, :] = np.mean(iou_3d_aps[:-1, :], axis=0)

        print("Computing pose AP scores...")
        total_pose_combinations = len(degree_thres_list) * len(shift_thres_list)
        with tqdm(
            total=total_pose_combinations, desc="Computing pose APs", unit="combination"
        ) as pbar:
            for i, degree_thres in enumerate(degree_thres_list):
                for j, shift_thres in enumerate(shift_thres_list):
                    for cls_id in range(num_classes):
                        cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][
                            i, j, :
                        ]
                        cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                        cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                        pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(
                            cls_pose_pred_matches_all,
                            cls_pose_pred_scores_all,
                            cls_pose_gt_matches_all,
                        )

                    pose_aps[-1, i, j] = np.mean(pose_aps[:-1, i, j])
                    pbar.update(1)

        if logger is not None:
            logger.warning(
                "3D IoU at 25: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.25)] * 100
                )
            )
            logger.warning(
                "3D IoU at 50: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.5)] * 100
                )
            )
            logger.warning(
                "3D IoU at 75: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.75)] * 100
                )
            )

            logger.warning(
                "5 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            logger.warning(
                "5 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            logger.warning(
                "10 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            logger.warning(
                "10 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            logger.warning(
                f"3D IoU at 25 per category:{iou_3d_aps[:, iou_thres_list.index(0.25)] * 100}"
            )
            logger.warning(
                f"3D IoU at 50 per category:{iou_3d_aps[:, iou_thres_list.index(0.5)] * 100}"
            )
            logger.warning(
                f"3D IoU at 75 per category:{iou_3d_aps[:, iou_thres_list.index(0.75)] * 100}"
            )

            logger.warning(
                f"5 degree, 2cm per category:{pose_aps[:, degree_thres_list.index(5), shift_thres_list.index(2)] * 100}"
            )
            logger.warning(
                f"5 degree, 5cm per category:{pose_aps[:, degree_thres_list.index(5), shift_thres_list.index(5)] * 100}"
            )
            logger.warning(
                f"10 degree, 2cm per category:{pose_aps[:, degree_thres_list.index(10), shift_thres_list.index(2)] * 100}"
            )
            logger.warning(
                f"10 degree, 5cm per category:{pose_aps[:, degree_thres_list.index(10), shift_thres_list.index(5)] * 100}"
            )

        else:
            print(
                "3D IoU at 25: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.25)] * 100
                )
            )
            print(
                "3D IoU at 50: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.5)] * 100
                )
            )
            print(
                "3D IoU at 75: {:.1f}".format(
                    iou_3d_aps[cat_id, iou_thres_list.index(0.75)] * 100
                )
            )

            print(
                "5 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            print(
                "5 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(5), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

            print(
                "10 degree, 2cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(2)
                    ]
                    * 100
                )
            )
            print(
                "10 degree, 5cm: {:.1f}".format(
                    pose_aps[
                        cat_id, degree_thres_list.index(10), shift_thres_list.index(5)
                    ]
                    * 100
                )
            )

        result = OrderedDict()
        for i, iou_thres in enumerate(iou_thres_list):
            result[f"3d_iou_{iou_thres:.2f}"] = iou_3d_aps[cat_id, i]

        for i, degree_thres in enumerate(degree_thres_list):
            for j, shift_thres in enumerate(shift_thres_list):
                result[f"pose {int(degree_thres)} degree, {int(shift_thres)}cm"] = (
                    pose_aps[cat_id, i, j]
                )

        return result

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
        for iou_thr in [0.5]:
            logger.info(f"\n{'-' * 15}iou_thr: {iou_thr}{'-' * 15}")
            # Follow the official implementation,
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
            # we should use the legacy coordinate system in yopo 1.x,
            # which means w, h should be computed as 'x2 - x1 + 1` and
            # `y2 - y1 + 1`
            class_preds = []
            for pred in preds:
                tmp_dets = []
                for label in range(len(self.dataset_meta["classes"])):
                    index = np.where(pred["labels"] == label)[0]
                    pred_bbox_scores = np.hstack(
                        [pred["bboxes"][index], pred["scores"][index].reshape((-1, 1))]
                    )
                    tmp_dets.append(pred_bbox_scores)
                class_preds.append(tmp_dets)
            mean_ap, _ = eval_map(
                class_preds,
                gts,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=self.dataset_meta["classes"],
                logger=logger,
                eval_mode="area",
                use_legacy_coordinate=True,
            )
            eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
        pose_results = self.compute_independent_mAP(
            preds, gts, logger=logger, cat_id=-1, classes=self.dataset_meta["classes"]
        )
        eval_results.update(pose_results)

        # Finalize JSON dumps if using JSON format
        if self.dump_format == "json":
            self.finalize_json_dumps()

        return eval_results


def compute_3d_matches(
    gt_class_ids,
    gt_RTs,
    gt_scales,
    gt_handle_visibility,
    class_names,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_RTs,
    pred_scales,
    iou_3d_thresholds,
    score_threshold=0,
):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """

    def trim_zeros(x):
        """It's common to have tensors larger than the available data and
        pad with zeros. This function removes rows that are all zeros.
        x: [rows, columns].
        """

        pre_shape = x.shape
        assert len(x.shape) == 2, x.shape
        new_x = x[~np.all(x == 0, axis=1)]
        post_shape = new_x.shape
        assert pre_shape[0] == post_shape[0]
        assert pre_shape[1] == post_shape[1]

        return new_x

    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[: pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_iou(
                pred_RTs[i],
                gt_RTs[j],
                pred_scales[i],
                gt_scales[j],
                gt_handle_visibility[j],
                class_names[pred_class_ids[i]],
                class_names[gt_class_ids[j]],
            )

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[: low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                # print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices


def compute_RT_overlaps(
    gt_class_ids, gt_RTs, gt_handle_visibility, pred_class_ids, pred_RTs, class_names
):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(
                pred_RTs[i],
                gt_RTs[j],
                gt_class_ids[j],
                gt_handle_visibility[j],
                class_names,
            )
    return overlaps


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, class_names):
    """
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    class_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    class_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    """

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if class_names[class_id] in ["bottle", "can", "bowl"]:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif class_names[class_id] == "mug" and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_names[class_id] in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(
            np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2)
        )
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])
    return result


def compute_match_from_degree_cm(
    overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list
):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    # print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if (
                        overlaps[i, j, 0] > degree_thres
                        or overlaps[i, j, 1] > shift_thres
                    ):
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_3d_iou(
    RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2
):
    """Computes IoU overlaps between two 3d bboxes.
    bbox_3d_1, bbox_3d_1: [3, 8]
    """

    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        # new
        # bbox_1_max = np.amax(bbox_3d_1, axis=1)
        # bbox_1_min = np.amin(bbox_3d_1, axis=1)
        # bbox_2_max = np.amax(bbox_3d_2, axis=1)
        # bbox_2_min = np.amin(bbox_3d_2, axis=1)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = (
            np.prod(bbox_1_max - bbox_1_min)
            + np.prod(bbox_2_max - bbox_2_min)
            - intersections
        )
        overlaps = intersections / union
        return overlaps

    if RT_1 is None or RT_2 is None:
        return -1

    if (class_name_1 in ["bottle", "bowl", "can"] and class_name_1 == class_name_2) or (
        class_name_1 == "mug"
        and class_name_1 == class_name_2
        and handle_visibility == 0
    ):
        # For symmetric objects, find the optimal rotation around the y-axis
        # to maximize the IoU instead of brute-forcing.
        def y_rotation_matrix(theta):
            return np.array(
                [
                    [np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1],
                ]
            )

        # The rotation part of RT_1 and RT_2 is scaled.
        # We need to extract the pure rotation to find the optimal angle.
        R1_scaled_mat = RT_1[:3, :3]
        R2_scaled_mat = RT_2[:3, :3]

        # Extract pure rotation matrices
        R1 = R1_scaled_mat / np.cbrt(np.linalg.det(R1_scaled_mat))
        R2 = R2_scaled_mat / np.cbrt(np.linalg.det(R2_scaled_mat))

        # Compute the relative rotation.
        R_12 = R1.T @ R2
        # Find the angle that aligns R1's y-rotation with R2's.
        # This is derived by maximizing trace(R_y(theta)^T R_1^T R_2).
        theta = np.arctan2(R_12[0, 2] - R_12[2, 0], R_12[0, 0] + R_12[2, 2])

        # Create a new RT_1 by applying the optimal rotation.
        # The new rotation matrix must be rescaled with the original scaling.
        rotated_R1 = R1 @ y_rotation_matrix(theta)[:3, :3]

        # Re-apply original scaling to the rotated matrix
        rotated_R1_scaled = rotated_R1 * np.cbrt(np.linalg.det(R1_scaled_mat))

        rotated_RT_1 = RT_1.copy()
        rotated_RT_1[:3, :3] = rotated_R1_scaled

        max_iou = asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2)
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou


def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = (
            np.array(
                [
                    [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                ]
            )
            + shift
        )
    else:
        bbox_3d = (
            np.array(
                [
                    [scale / 2, +scale / 2, scale / 2],
                    [scale / 2, +scale / 2, -scale / 2],
                    [-scale / 2, +scale / 2, scale / 2],
                    [-scale / 2, +scale / 2, -scale / 2],
                    [+scale / 2, -scale / 2, scale / 2],
                    [+scale / 2, -scale / 2, -scale / 2],
                    [-scale / 2, -scale / 2, scale / 2],
                    [-scale / 2, -scale / 2, -scale / 2],
                ]
            )
            + shift
        )

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def _process_batch_worker(args):
    """Worker function for parallel processing of prediction batches."""
    (
        batch_preds,
        batch_gts,
        num_classes,
        classes,
        iou_thres_list,
        degree_thres_list,
        shift_thres_list,
        use_matches_for_pose,
        iou_pose_thres,
    ) = args

    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_iou_thres = len(iou_thres_list)

    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]

    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pose_pred_scores_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]

    for pred, gt in zip(batch_preds, batch_gts):
        gt_class_ids = gt["labels"].astype(np.int32)
        gt_RTs = gt["T"]
        gt_sizes = gt["sizes"]

        if "gt_handle_visibility" in gt:
            gt_handle_visibility = gt["gt_handle_visibility"]
        else:
            gt_handle_visibility = np.ones_like(gt_class_ids)

        pred_bboxes = np.array(pred["bboxes"])
        pred_class_ids = pred["labels"]
        pred_scores = pred["scores"]
        pred_RTs = pred["T"]
        pred_sizes = pred["sizes"]

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = (
                gt_class_ids[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros(0)
            )
            cls_gt_scales = (
                gt_sizes[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 3))
            )
            cls_gt_RTs = (
                gt_RTs[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 4, 4))
            )

            cls_pred_class_ids = (
                pred_class_ids[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros(0)
            )
            cls_pred_bboxes = (
                pred_bboxes[pred_class_ids == cls_id, :]
                if len(pred_class_ids)
                else np.zeros((0, 4))
            )
            cls_pred_scores = (
                pred_scores[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros(0)
            )
            cls_pred_RTs = (
                pred_RTs[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 4, 4))
            )
            cls_pred_scales = (
                pred_sizes[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 3))
            )

            # calculate the overlap between each gt instance and pred instance
            if classes[cls_id] != "mug":
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = (
                    gt_handle_visibility[gt_class_ids == cls_id]
                    if len(gt_class_ids)
                    else np.ones(0)
                )

            (iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices) = (
                compute_3d_matches(
                    cls_gt_class_ids,
                    cls_gt_RTs,
                    cls_gt_scales,
                    cls_gt_handle_visibility,
                    classes,
                    cls_pred_bboxes,
                    cls_pred_class_ids,
                    cls_pred_scores,
                    cls_pred_RTs,
                    cls_pred_scales,
                    iou_thres_list,
                )
            )
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]

            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1
            )
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate(
                (iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1
            )
            assert (
                iou_pred_matches_all[cls_id].shape[1]
                == iou_pred_scores_all[cls_id].shape[1]
            )
            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1
            )

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                cls_pred_class_ids = (
                    cls_pred_class_ids[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros(0)
                )
                cls_pred_RTs = (
                    cls_pred_RTs[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4, 4))
                )
                cls_pred_scores = (
                    cls_pred_scores[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros(0)
                )
                cls_pred_bboxes = (
                    cls_pred_bboxes[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4))
                )

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = (
                    cls_gt_class_ids[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros(0)
                )
                cls_gt_RTs = (
                    cls_gt_RTs[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros((0, 4, 4))
                )
                cls_gt_handle_visibility = (
                    cls_gt_handle_visibility[iou_thres_gt_match > -1]
                    if len(iou_thres_gt_match) > 0
                    else np.zeros(0)
                )

            RT_overlaps = compute_RT_overlaps(
                cls_gt_class_ids,
                cls_gt_RTs,
                cls_gt_handle_visibility,
                cls_pred_class_ids,
                cls_pred_RTs,
                classes,
            )

            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(
                RT_overlaps,
                cls_pred_class_ids,
                cls_gt_class_ids,
                degree_thres_list,
                shift_thres_list,
            )

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1
            )

            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, 1)
            )
            pose_pred_scores_all[cls_id] = np.concatenate(
                (pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1
            )
            assert (
                pose_pred_scores_all[cls_id].shape[2]
                == pose_pred_matches_all[cls_id].shape[2]
            ), "{} vs. {}".format(
                pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape
            )
            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1
            )

    return (
        iou_pred_matches_all,
        iou_pred_scores_all,
        iou_gt_matches_all,
        pose_pred_matches_all,
        pose_pred_scores_all,
        pose_gt_matches_all,
    )
