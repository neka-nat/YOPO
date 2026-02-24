# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import os
import cv2
import mmcv
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color

from yopo.utils.transform3d_utils import transform_3d_coordinates, project_3d_to_2d


def load_models(models_path: str) -> dict:
    """
    Loads the 3D model for a given object ID using plyfile.
    Returns a dictionary with 'vertices' and 'faces'.
    """
    models = dict()
    for model in os.listdir(models_path):
        if model.endswith(".ply"):
            ply_data = PlyData.read(os.path.join(models_path, model))
            # strip the 'model_' prefix and '.ply' suffix
            model_id = int(model.split(".")[0][4:]) - 1
            vertices = ply_data["vertex"]
            points = np.stack(
                [vertices[:]["x"], vertices[:]["y"], vertices[:]["z"]], axis=-1
            ).astype(np.float64)
            models[model_id] = points

    return models


def get_exact_cuboid_corners(vertices, pose):
    """
    Computes the eight corners of the object's cuboid based on the model's
    dimensions (from its axis-aligned bounding box) and the ground-truth 6D pose.

    Steps:
      1. Compute the axis-aligned bounding box (AABB) of the model to get the extents and center.
      2. Define eight corner offsets in a standard order:
           Corners are defined (relative to the AABB center) as:
             0: (-dx/2, -dy/2, -dz/2)
             1: ( dx/2, -dy/2, -dz/2)
             2: ( dx/2,  dy/2, -dz/2)
             3: (-dx/2,  dy/2, -dz/2)
             4: (-dx/2, -dy/2,  dz/2)
             5: ( dx/2, -dy/2,  dz/2)
             6: ( dx/2,  dy/2,  dz/2)
             7: (-dx/2,  dy/2,  dz/2)
      3. Add the offsets to the center to form the eight corners (in the model coordinate system).
      4. Apply the ground-truth 6D transform (pose) to these corners.

    Parameters:
      vertices (np.ndarray): Nx3 array of 3D points representing the model.
      pose (np.ndarray): 3x4 ground-truth pose matrix.

    Returns:
      np.ndarray: An (8, 3) array of the transformed cuboid corners in camera coordinates.
    """
    # Get the axis-aligned bounding box from the model.
    # Step 1: Compute AABB from raw vertices
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    dims = max_bound - min_bound
    center = (min_bound + max_bound) / 2

    # Define the 8 offset vectors (standard order).
    offsets = np.array(
        [
            [-dims[0] / 2, -dims[1] / 2, -dims[2] / 2],
            [dims[0] / 2, -dims[1] / 2, -dims[2] / 2],
            [dims[0] / 2, dims[1] / 2, -dims[2] / 2],
            [-dims[0] / 2, dims[1] / 2, -dims[2] / 2],
            [-dims[0] / 2, -dims[1] / 2, dims[2] / 2],
            [dims[0] / 2, -dims[1] / 2, dims[2] / 2],
            [dims[0] / 2, dims[1] / 2, dims[2] / 2],
            [-dims[0] / 2, dims[1] / 2, dims[2] / 2],
        ]
    )

    # Compute the corners in the model coordinate system.
    corners_model = center + offsets

    # Form a 4x4 transformation matrix from the 3x4 pose.
    T = np.vstack((pose, [0, 0, 0, 1]))

    # Convert corners to homogeneous coordinates and transform.
    corners_hom = np.hstack((corners_model, np.ones((8, 1))))
    corners_transformed = (T @ corners_hom.T).T[:, :3]
    return corners_transformed


@VISUALIZERS.register_module()
class PoseLocalVisualizer(Visualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from yopo.structures import DetDataSample
        >>> from yopo.visualization import DetLocalVisualizer

        >>> det_local_visualizer = DetLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_instances.labels = torch.randint(0, 2, (1,))
        >>> gt_det_data_sample = DetDataSample()
        >>> gt_det_data_sample.gt_instances = gt_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample)
        >>> det_local_visualizer.add_datasample(
        ...                       'image', image, gt_det_data_sample,
        ...                        out_file='out_file.jpg')
        >>> det_local_visualizer.add_datasample(
        ...                        'image', image, gt_det_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_det_data_sample = DetDataSample()
        >>> pred_det_data_sample.pred_instances = pred_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample,
        ...                         pred_det_data_sample)
    """

    def __init__(
        self,
        name: str = "visualizer",
        image: Optional[np.ndarray] = None,
        vis_backends: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        axis_length: float = 0.05,
        alpha: float = 0.5,
    ) -> None:  # originally 0.8
        super().__init__(
            name=name, image=image, vis_backends=vis_backends, save_dir=save_dir
        )
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.axis_length = axis_length
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}
        self.models = None

    def _draw_instances(
        self,
        image: np.ndarray,
        instances: ["InstanceData"],
        classes: Optional[List[str]],
        palette: Optional[List[tuple]],
        intrinsic: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if "bboxes" in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width,
            )

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                if "label_names" in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = (
                        classes[label] if classes is not None else f"class {label}"
                    )
                if "scores" in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f": {score}"

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[
                        {
                            "facecolor": "black",
                            "alpha": 0.8,
                            "pad": 0.7,
                            "edgecolor": "none",
                        }
                    ],
                )

        if "masks" in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors="w", alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if len(labels) > 0 and (
                "bboxes" not in instances or instances.bboxes.sum() == 0
            ):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8
                    )
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    if "label_names" in instances:
                        label_text = instances.label_names[i]
                    else:
                        label_text = (
                            classes[label] if classes is not None else f"class {label}"
                        )
                    if "scores" in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f": {score}"

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments="center",
                        bboxes=[
                            {
                                "facecolor": "black",
                                "alpha": 0.8,
                                "pad": 0.7,
                                "edgecolor": "none",
                            }
                        ],
                    )
        if intrinsic is not None and len(intrinsic) == 4:
            # If intrinsic is a list of 4 elements, assume it is [fx, fy, cx, cy]
            intrinsic = [
                intrinsic[0],
                0,
                intrinsic[2],
                0,
                intrinsic[1],
                intrinsic[3],
                0,
                0,
                1,
            ]
        intrinsic = (
            np.array(intrinsic).reshape(3, 3) if intrinsic is not None else np.eye(3)
        )

        # 6D pose
        if "T" in instances:
            base_xyz_axis = np.array(
                [
                    [0, 0, 0],  # origin
                    [1, 0, 0],  # x-axis
                    [0, 1, 0],  # y-axis
                    [0, 0, 1],  # z-axis
                ]
            )

            # get the 3x4 transformation matrix
            Ts = instances.T
            if Ts.shape[1] == 4:
                # If T is a 4x4 matrix, we need to convert it to 3x4
                Ts = Ts[:, :3, :4]
            labels = instances.labels
            for idx, (transform, label) in enumerate(zip(Ts, labels)):
                if isinstance(transform, torch.Tensor):
                    transform = transform.cpu().numpy()

                origin = transform[:, 3]  # object center in camera coordinates
                if "sizes" in instances:
                    sizes = instances.sizes.cpu().numpy()[idx]
                    if len(sizes) == 3:
                        # If sizes are provided, we scale the axis accordingly
                        width, height, depth = sizes.tolist()
                        base_xyz_axis = np.array(
                            [
                                [0, 0, 0],  # origin
                                [width / 2, 0, 0],  # x-axis
                                [0, height / 2, 0],  # y-axis
                                [0, 0, depth / 2],
                            ]
                        )  # z-axis
                else:
                    # If sizes are not provided, we use the default axis
                    base_xyz_axis = np.array(
                        [
                            [0, 0, 0],  # origin
                            [1, 0, 0],  # x-axis
                            [0, 1, 0],  # y-axis
                            [0, 0, 1],  # z-axis
                        ]
                    )

                    # Calculate adaptive axis length based on dataset type and distance
                    distance = np.linalg.norm(origin)  # distance from camera to object
                    axis_length = self.axis_length  # default axis length
                    # If the distance is greater than 100,
                    # it means the distance is in millimeters,
                    # so we scale the axis length to meters.
                    if distance > 100:
                        axis_length *= 1000
                    base_xyz_axis = base_xyz_axis * axis_length

                transformed_axis = transform_3d_coordinates(base_xyz_axis, transform)
                projected_axis = project_3d_to_2d(transformed_axis, intrinsic)

                # Draw the axes
                for j in range(1, 4):
                    color = (
                        (255, 0, 0)
                        if j == 1
                        else (0, 255, 0)
                        if j == 2
                        else (0, 0, 255)
                    )
                    self.draw_lines(
                        projected_axis[[0, j], 0],
                        projected_axis[[0, j], 1],
                        colors=color,
                        line_widths=2,
                    )
                # Draw the cuboid
                if self.models is not None:
                    vertices = self.models[label.item()]
                    cuboid_corners = get_exact_cuboid_corners(vertices, transform)
                    projected_cuboid = (intrinsic @ cuboid_corners.T).T
                    projected_cuboid[:, :2] /= projected_cuboid[:, 2:3]
                    # Draw the cuboid edges
                    for i in range(4):
                        self.draw_lines(
                            projected_cuboid[[i, (i + 1) % 4], 0],
                            projected_cuboid[[i, (i + 1) % 4], 1],
                            colors=(85, 221, 85),
                            # colors=(255, 255, 0),
                            line_widths=2,
                        )
                        self.draw_lines(
                            projected_cuboid[[i + 4, ((i + 1) % 4) + 4], 0],
                            projected_cuboid[[i + 4, ((i + 1) % 4) + 4], 1],
                            colors=(85, 221, 85),
                            # colors=(255, 255, 0),
                            line_widths=2,
                        )
                        self.draw_lines(
                            projected_cuboid[[i, i + 4], 0],
                            projected_cuboid[[i, i + 4], 1],
                            colors=(85, 221, 85),
                            # colors=(255, 255, 0),
                            line_widths=2,
                        )
                # draw the cuboid corners
                elif "sizes" in instances:
                    sizes = instances.sizes.cpu().numpy()
                    Ts = instances.T
                    if Ts.shape[1] == 4:
                        # If T is a 4x4 matrix, we need to convert it to 3x4
                        Ts = Ts[:, :3, :4]
                    if len(sizes) > 0:
                        for i, (size, transform) in enumerate(zip(sizes, Ts)):
                            origin = transform[:, 3].cpu().numpy()
                            # size is [width, height, depth]
                            width, height, depth = (size).tolist()
                            # Create a cuboid centered at the origin
                            cuboid_corners = np.array(
                                [
                                    [-width / 2, -height / 2, -depth / 2],
                                    [width / 2, -height / 2, -depth / 2],
                                    [width / 2, height / 2, -depth / 2],
                                    [-width / 2, height / 2, -depth / 2],
                                    [-width / 2, -height / 2, depth / 2],
                                    [width / 2, -height / 2, depth / 2],
                                    [width / 2, height / 2, depth / 2],
                                    [-width / 2, height / 2, depth / 2],
                                ]
                            )

                            # Transform cuboid corners using the pose matrix
                            transform_np = transform.cpu().numpy()
                            transformed_cuboid = transform_3d_coordinates(
                                cuboid_corners, transform_np
                            )
                            # Project to 2D coordinates
                            projected_cuboid = project_3d_to_2d(
                                transformed_cuboid, intrinsic
                            )

                            # Draw the cuboid edges
                            for j in range(4):
                                self.draw_lines(
                                    projected_cuboid[[j, (j + 1) % 4], 0],
                                    projected_cuboid[[j, (j + 1) % 4], 1],
                                    colors=(85, 221, 85),
                                    line_widths=2,
                                )
                                self.draw_lines(
                                    projected_cuboid[[j + 4, ((j + 1) % 4) + 4], 0],
                                    projected_cuboid[[j + 4, ((j + 1) % 4) + 4], 1],
                                    colors=(85, 221, 85),
                                    line_widths=2,
                                )
                                self.draw_lines(
                                    projected_cuboid[[j, j + 4], 0],
                                    projected_cuboid[[j, j + 4], 1],
                                    colors=(85, 221, 85),
                                    line_widths=2,
                                )
        return self.get_image()

    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional["DetDataSample"] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        # TODO: Supported in mmengine's Viusalizer.
        out_file: Optional[str] = None,
        pred_score_thr: float = 0.3,
        step: int = 0,
    ) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get("classes", None)
        palette = self.dataset_meta.get("palette", None)

        if out_file is not None:
            extension = out_file.split(".")[-1]
            root_folder = "/".join(out_file.split("/")[:-1])
            out_file = f"{root_folder}/{data_sample.img_id}.{extension}"

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        models_info_path = data_sample.get("models_info_path", None)
        if self.models is None and models_info_path is not None:
            # Load the 3D model for a given object ID using plyfile.
            # Returns a dictionary with 'vertices' and 'faces'.
            self.models = load_models(models_info_path)

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if "gt_instances" in data_sample:
                gt_img_data = self._draw_instances(
                    image,
                    data_sample.gt_instances,
                    classes,
                    palette,
                    data_sample.intrinsic,
                )

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if "pred_instances" in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(
                    image, pred_instances, classes, palette, data_sample.intrinsic
                )

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
