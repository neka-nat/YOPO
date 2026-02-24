import os.path as osp
from typing import List
import pickle
import json

import numpy as np
import cv2

from yopo.registry import DATASETS
from yopo.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class Wild6DDataset(BaseDetDataset):
    """Wild6D dataset for 6D Pose Estimation.
    """
    METAINFO = {
        'classes': ('bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142),
                    (0, 0, 230), (106, 0, 228), (0, 60, 100)]
    }

    def __init__(self,
                 data_root: str = 'data/wild6d',
                 split: str = 'test',
                 **kwargs) -> None:
        self.split = split
        super().__init__(data_root=data_root, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from Wild6D dataset."""
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        if self.split == 'train':
            return self._load_train_data()
        else:
            return self._load_test_data()

    def _load_train_data(self) -> List[dict]:
        """Load training data."""
        data_list = []
        for class_name in self._metainfo['classes']:
            class_dir = osp.join(self.data_root, class_name)
            if not osp.isdir(class_dir):
                continue

            train_list_file = osp.join(class_dir, 'train_list.txt')
            if not osp.exists(train_list_file):
                continue

            with open(train_list_file, 'r') as f:
                img_indices = [line.strip() for line in f.readlines()]

            for img_index in img_indices:
                scene_dir, frame_name = img_index.split('/')
                img_path = osp.join(class_dir, scene_dir, f"{frame_name}.jpg")
                if not osp.exists(img_path):
                    continue
                
                raw_img_info = self._get_raw_img_info(img_path, class_name, scene_dir, frame_name)
                
                gt_info_path = osp.join(self.data_root, 'pkl_annotations', class_name, f"{class_name}-{scene_dir}-{frame_name}.pkl")
                if not osp.exists(gt_info_path):
                    # Fallback for test set structure
                    gt_info_path = osp.join(self.data_root, 'test_set', 'pkl_annotations', class_name, f"{class_name}-{scene_dir}-{frame_name}.pkl")

                if osp.exists(gt_info_path):
                    with open(gt_info_path, 'rb') as f:
                        gt_info = pickle.load(f)
                    
                    frame_idx = int(frame_name)
                    if frame_idx < len(gt_info['annotations']):
                        gt_anno = gt_info['annotations'][frame_idx]
                        parsed_data_info = self.parse_data_info(raw_img_info, gt_anno)
                        if parsed_data_info:
                            data_list.append(parsed_data_info)
        return data_list

    def _load_test_data(self) -> List[dict]:
        """Load test data."""
        data_list = []
        test_dir = osp.join(self.data_root, 'test_set')
        for class_name in self._metainfo['classes']:
            test_list_file = osp.join(test_dir, f'test_list_{class_name}.txt')
            if not osp.exists(test_list_file):
                continue

            with open(test_list_file, 'r') as f:
                img_paths = [line.strip() for line in f.readlines()]
            # filter root paths
            img_paths = ['/'.join(p.split('/')[3:]) for p in img_paths if p]
            # replace 'rgbd' with 'images' TODO: Adjust based on actual structure
            img_paths = [p.replace('rgbd', 'images') for p in img_paths]
            
            # temporary
            # img_paths = img_paths[:500]  # Limit to 100 images for testing

            for img_path_rel in img_paths:
                img_path = osp.join(test_dir, img_path_rel)
                if not osp.exists(img_path):
                    continue
                
                parts = img_path_rel.split('/')
                scene_dir = parts[-4]
                video_dir = parts[-3]
                frame_name = parts[-1].replace('.jpg', '')

                raw_img_info = self._get_raw_img_info(img_path, class_name, scene_dir, video_dir)

                
                gt_info_path = osp.join(test_dir, 'pkl_annotations', class_name, f"{class_name}-{scene_dir}-{video_dir}.pkl")

                if not osp.exists(gt_info_path):
                    continue
                with open(gt_info_path, 'rb') as f:
                    gt_info = pickle.load(f)
                
                frame_idx = int(frame_name)
                if frame_idx < len(gt_info['annotations']):
                    gt_anno = gt_info['annotations'][frame_idx]
                    parsed_data_info = self.parse_data_info(raw_img_info, gt_anno)
                    if parsed_data_info:
                        data_list.append(parsed_data_info)
        return data_list

    def _get_raw_img_info(self, img_path, class_name, scene_dir, video_dir) -> dict:
        """Get raw image info dictionary."""
        height, width, _ = cv2.imread(img_path).shape
        frame_name = osp.basename(img_path).replace('.jpg', '')
        
        image_folder = osp.dirname(img_path)
        metadata_path = osp.dirname(image_folder)
        # It seems the metadata file is just called 'metadata'
        metadata_file = osp.join(metadata_path, 'metadata')
        
        intrinsic = [0,0,0,0] # fx, fy, cx, cy
        if osp.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
            K = meta['K']
            intrinsic = [K[0], K[4], K[6], K[7]]

        raw_img_info = {
            'img_id': f"{class_name}/{scene_dir}_{video_dir}/frame_{frame_name}",
            'img_path': img_path,
            'height': height,
            'width': width,
            'intrinsic': intrinsic,
            'class_name': class_name,
            'scene_dir': scene_dir,
            'frame_name': frame_name
        }
        return raw_img_info

    def parse_data_info(self, img_info: dict, gt_info: dict) -> dict:
        """Parse raw annotation to target format."""
        data_info = {
            'img_path': img_info['img_path'],
            'img_id': img_info['img_id'],
            'height': img_info['height'],
            'width': img_info['width'],
            'intrinsic': img_info['intrinsic'],
        }

        mask_path = img_info['img_path'].replace('.jpg', '-mask.png')
        if osp.exists(mask_path):
            data_info['mask_path'] = mask_path
        
        depth_path = img_info['img_path'].replace('.jpg', '-depth.png')
        if osp.exists(depth_path):
            data_info['depth_path'] = depth_path

        instances = self._parse_instance_info(gt_info, img_info)
        if not instances:
            return None
        
        data_info['instances'] = instances
        return data_info

    def _parse_instance_info(self, gt_info: dict, img_info: dict) -> List[dict]:
        """Parse instance information."""
        instances = []
        
        class_id = self.cat2label.get(img_info['class_name'])
        if class_id is None:
            return instances

        # Bbox from mask
        mask_path = img_info['img_path'].replace('.jpg', '-mask.png')
        if not osp.exists(mask_path):
            return instances
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return instances
            
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return instances
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = [x1, y1, x2, y2]

        rotation = gt_info.get('rotation')
        translation = gt_info.get('translation')
        size = gt_info.get('size')

        if rotation is None or translation is None or size is None:
            return instances

        intrinsic = img_info['intrinsic']
        fx, fy, cx, cy = intrinsic
        tx, ty, tz = translation
        if tz == 0:
            center_2d = [(x1 + x2) / 2, (y1 + y2) / 2]
        else:
            center_2d = [tx / tz * fx + cx, ty / tz * fy + cy]

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.array(rotation, dtype=np.float32)
        T[:3, 3] = np.array(translation, dtype=np.float32)

        instance = {
            'bbox': bbox,
            'bbox_label': class_id,
            'rotation': np.array(rotation, dtype=np.float32),
            'translation': np.array(translation, dtype=np.float32),
            'size': np.array(size, dtype=np.float32),
            'T': T,
            'center_2d': center_2d,
            'z': translation[2],
            'ignore_flag': 0,
        }
        instances.append(instance)
        return instances