# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from typing import List, Optional, Union

import numpy as np
import torch
from mmengine.fileio import get, get_local_path, list_from_file

from yopo.registry import DATASETS
from ..base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class YCBVideoBOPDataset(BaseDetDataset):
    """YCB-Video in BOP Challenge dataset for 6D Pose Estimation.

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    METAINFO = {
        'classes':
        ("master_chef_can", "cracker_box", "sugar_box",
        "tomato_soup_can", "mustard_bottle", "tuna_fish_can",
        "pudding_box", "gelatin_box",
        "potted_meat_can", "banana",
        "pitcher_base", "bleach_cleanser",
        "bowl", "mug",
        "power_drill", "wood_block",
        "scissors", "large_marker",
        "large_clamp", "extra_large_clamp",
        "foam_brick"),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0)]
    }

    IMG_SHAPE = (480, 640) # (height, width)

    def __init__(self,
                 use_model: bool = False,
                 norm_factor_z: float = (141.37, 1740.27),
                 **kwargs) -> None:
        self.use_model = use_model
        self.norm_factor_z = norm_factor_z
        super().__init__(**kwargs)

    @property
    def sub_data_root(self) -> str:
        """Return the sub data root."""
        return self.data_prefix.get('sub_data_root', '')

    def load_data_list(self) -> List[dict]:
        """Load annotation from YCB-Video in BOP Challenge dataset.

        The folder structure is as follows:

        data_root/
            sub_data_root/
                ├── 000001/
                │   ├── rgb/
                │   ├── scene_camera.json
                │   ├── scene_gt.json
                │   └── scene_gt_info.json
                ├── 000002/
                │   ├── rgb/
                │   ├── scene_camera.json
                │   ├── scene_gt.json
                │   └── scene_gt_info.json
            ...

        """
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        sequences = os.listdir(self.sub_data_root)
        for seq in sequences:
            imgs = os.listdir(osp.join(self.sub_data_root, seq, 'rgb'))
            
            camera_info_path = os.path.join(self.sub_data_root, seq, 'scene_camera.json')
            gt_info_path = os.path.join(self.sub_data_root, seq, 'scene_gt.json')
            gt_metainfo_path = os.path.join(self.sub_data_root, seq, 'scene_gt_info.json')

            with open(camera_info_path, 'r') as f:
                camera_infos = json.load(f)
            with open(gt_info_path, 'r') as f:
                gt_infos = json.load(f)
            with open(gt_metainfo_path, 'r') as f:
                gt_metainfos = json.load(f)


            for img in imgs:
                file_name = osp.join(seq, 'rgb', img)

                frame_id = img.split('.')[0]
                img_id = seq+'_'+frame_id

                raw_img_info = {}
                raw_img_info['img_id'] = img_id
                raw_img_info['file_name'] = file_name

                frame_id = str(int(frame_id))
                camera_info = camera_infos[frame_id]
                gt_info = gt_infos[frame_id]
                gt_metainfo = gt_metainfos[frame_id]

                parsed_data_info = self.parse_data_info(raw_img_info,
                                                        gt_info, gt_metainfo,
                                                        camera_info)
                data_list.append(parsed_data_info)

        return data_list


    def parse_data_info(self,
                        img_info: dict,
                        gt_info: dict,
                        gt_metainfo: dict,
                        camera_info: dict,
        ) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.sub_data_root, img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']


        data_info['height'] = self.IMG_SHAPE[0]
        data_info['width'] = self.IMG_SHAPE[1]
        data_info['intrinsic'] = camera_info['cam_K']
        data_info['norm_factor_z'] = self.norm_factor_z
        data_info['models_info_path'] = osp.join(
            self.data_root, 'models')

        data_info['instances'] = self._parse_instance_info(
            gt_info, gt_metainfo, camera_info)

        return data_info

    def _parse_instance_info(self,
                             gt_info: dict,
                             gt_metainfo: dict,
                             camera_info: dict) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (dict): Raw annotation information.
        Returns:
            List[dict]: List of instances.
        """
        instances = []
        K = torch.tensor(camera_info['cam_K']).reshape(3, 3)
        for obj, obj_info in zip(gt_info, gt_metainfo):
            if obj_info['visib_fract'] <= 0:
                continue
            instance = {}

            r = obj["cam_R_m2c"]
            r_6d_rep = [r[i] for i in [0, 3, 6, 1, 4, 7]]
            t = obj["cam_t_m2c"]

            # convert to 3x4 matrix
            r = torch.tensor(r).reshape(3, 3)

            t = torch.tensor(t).reshape(3, 1)
            T = torch.cat((r, t), dim=1) # 3x4 matrix

            center_2d = K @ torch.tensor(t) / t[2] # normalize by z
            center_2d = center_2d[:2].tolist() # only x, y

            x, y, w, h= obj_info["bbox_obj"]
            # clip bbox to image size
            x = max(0, min(self.IMG_SHAPE[1], x))
            y = max(0, min(self.IMG_SHAPE[0], y))
            w = max(0, min(self.IMG_SHAPE[1], x + w)) - x
            h = max(0, min(self.IMG_SHAPE[0], y + h)) - y
            # convert to x1, y1, x2, y2 format
            bbox = [x, y, x + w, y + h]
            class_id = obj["obj_id"] - 1

            instance['bbox'] = bbox
            instance['bbox_label'] = class_id
            instance['rotation'] = r_6d_rep
            instance['center_2d'] = center_2d
            # instance['z'] = t[2] / self.norm_factor_z
            # instance['z'] = (t statust[2] - self.norm_factor_z[0]) \
            #     / (self.norm_factor_z[1] - self.norm_factor_z[0])
            instance['z'] = np.log(t[2])
            # instance['z'] = t[2] / 1800.

            instance['T'] = T
            instance['ignore_flag'] = 0

            instances.append(instance)
        return instances

@DATASETS.register_module()
class LMOBOPDataset(BaseDetDataset):
    """YCB-Video in BOP Challenge dataset for 6D Pose Estimation.

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    METAINFO = {
        'classes':
        ("ape", "benchvise", "bowl", "cam", "can", "cat", "cup", "driller", 
         "duck", "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), 
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), 
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0), 
                    (175, 116, 175), (250, 0, 30), (165, 42, 42)]
    }

    IMG_SHAPE = (416, 416) # (height, width)

