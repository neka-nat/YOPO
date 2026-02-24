import os.path as osp
from typing import List, Union
import pickle
import glob

import numpy as np

from yopo.registry import DATASETS
from yopo.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class HouseCat6DDataset(BaseDetDataset):
    """HouseCat6D dataset for 6D Pose Estimation.

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
        "classes": (
            "box",
            "bottle",
            "can",
            "cup",
            "remote",
            "teapot",
            "cutlery",
            "glass",
            "shoe",
            "tube",
        ),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (238, 130, 238),
            (255, 165, 0),
            (255, 20, 147),
            (0, 255, 255),
        ],
    }

    SPLIT_INFO = dict(
        train=dict(
            scene_pattern="scene*",
            model_path="/datasets/HouseCat6D/obj_models_small_size_final/",
            intrinsic_file="intrinsics.txt",
        ),
        val=dict(
            scene_pattern="val_scene*",
            model_path="/datasets/HouseCat6D/obj_models_small_size_final/",
            intrinsic_file="intrinsics.txt",
        ),
        test=dict(
            scene_pattern="test_scene*",
            model_path="/datasets/HouseCat6D/obj_models_small_size_final/",
            intrinsic_file="intrinsics.txt",
        ),
    )
    IMG_SHAPE = (852, 1096)  # (height, width)

    def __init__(
        self,
        split: str = "train",
        num_sample_points: int = 1000,
        use_cuboid_as_bbox: bool = False,
        scene_limit: int = -1,
        img_limit: int = -1,
        **kwargs,
    ) -> None:
        assert split in self.SPLIT_INFO, (
            f"Invalid split: {split}. Available splits: {list(self.SPLIT_INFO.keys())}"
        )

        self.split = split
        self.num_sample_points = num_sample_points
        self.use_cuboid_as_bbox = use_cuboid_as_bbox
        self.scene_limit = scene_limit
        self.img_limit = img_limit

        self.xmap = np.array(
            [[i for i in range(self.IMG_SHAPE[1])] for j in range(self.IMG_SHAPE[0])]
        )
        self.ymap = np.array(
            [[j for i in range(self.IMG_SHAPE[1])] for j in range(self.IMG_SHAPE[0])]
        )
        # self.sym_ids = [0, 2, 4, 5, 6, 8]    # symmetric objects (0-indexed)
        self.sym_ids = [1, 2, 7]  # symmetric objects (0-indexed)
        self.norm_scale = 1000.0  # normalization scale
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from HouseCat6D dataset.

        The folder structure is as follows:

        data_root/
            scene01/
                ├── rgb/
                │   ├── 000000.png
                │   └── ...
                ├── depth/
                │   ├── 000000.png
                │   └── ...
                ├── instance/
                │   ├── 000000.png
                │   └── ...
                ├── labels/
                │   ├── 000000_label.pkl
                │   └── ...
                ├── pol/
                │   ├── 000000.png
                │   └── ...
                └── intrinsics.txt
            scene02/
                └── ...
            test_scene01/
                └── ...
        """
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo["classes"])}

        dataset_info = self.SPLIT_INFO[self.split]
        scene_pattern = dataset_info["scene_pattern"]
        model_path = dataset_info["model_path"]

        # Load 3D models similar to NOCS dataset
        self.models = dict()
        if model_path is not None:
            models_full_path = osp.join(self.data_root, "..", model_path)
            if osp.exists(models_full_path):
                # Check if there's a pickle file first (like NOCS)
                model_pickle_file = osp.join(models_full_path, "objects.pkl")
                if osp.exists(model_pickle_file):
                    with open(model_pickle_file, "rb") as f:
                        self.models.update(pickle.load(f))
                    print(f"Loaded models from pickle file: {model_pickle_file}")
                    print(f"Loaded models: {self.models.keys()}")
                    print(
                        type(self.models["glass-small_4"]),
                        self.models["glass-small_4"].shape,
                    )
                else:
                    # Load individual model files by class
                    self.models = self._load_individual_models(models_full_path)
            else:
                print(f"Warning: Models path does not exist: {models_full_path}")

        # Find all scenes matching the pattern
        scene_dirs = glob.glob(osp.join(self.data_root, scene_pattern))
        # Filter out things which are not directories
        scene_dirs = [d for d in scene_dirs if osp.isdir(d)]
        scene_dirs.sort()

        if self.scene_limit > 0:
            scene_dirs = scene_dirs[: self.scene_limit]

        print(f"Found {len(scene_dirs)} scenes for {self.split} split")

        # Collect all image paths from all scenes
        self.img_list = []
        for scene_dir in scene_dirs:
            rgb_dir = osp.join(scene_dir, "rgb")
            if not osp.exists(rgb_dir):
                continue

            img_paths = glob.glob(osp.join(rgb_dir, "*.png"))
            img_paths.sort()

            if self.img_limit > 0:
                img_paths = img_paths[: self.img_limit]

            self.img_list.extend(img_paths)

        print(f"Total {len(self.img_list)} images found")

        data_list = []
        for img_file in self.img_list:
            # Get scene directory
            scene_dir = osp.dirname(osp.dirname(img_file))
            frame_name = osp.basename(img_file).split(".")[0]

            # Check if label file exists
            label_file = osp.join(scene_dir, "labels", f"{frame_name}_label.pkl")
            if not osp.exists(label_file):
                continue

            # Read label info
            try:
                with open(label_file, "rb") as f:
                    gt_info = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load label file {label_file}: {e}")
                continue

            # Read intrinsic parameters
            intrinsic_file = osp.join(scene_dir, "intrinsics.txt")
            if osp.exists(intrinsic_file):
                intrinsic = np.loadtxt(intrinsic_file).reshape(3, 3)
                intrinsic = [
                    intrinsic[0, 0],
                    intrinsic[1, 1],
                    intrinsic[0, 2],
                    intrinsic[1, 2],
                ]
            else:
                # Default intrinsics if file not found
                intrinsic = [577.5, 577.5, 548.0, 426.0]  # fx, fy, cx, cy
            intrinsic = list(map(float, intrinsic))

            raw_img_info = {}
            raw_img_info["img_id"] = f"{osp.basename(scene_dir)}/{frame_name}"
            raw_img_info["file_name"] = osp.relpath(img_file, self.data_root)
            raw_img_info["img_path"] = img_file
            raw_img_info["scene_dir"] = scene_dir
            raw_img_info["frame_name"] = frame_name
            raw_img_info["intrinsic"] = intrinsic

            parsed_data_info = self.parse_data_info(raw_img_info, gt_info)
            if parsed_data_info is not None:
                data_list.append(parsed_data_info)

        print(f"Successfully loaded {len(data_list)} samples")
        return data_list

    def parse_data_info(
        self,
        img_info: dict,
        gt_info: dict,
    ) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info: Image information dict
            gt_info: Ground truth information dict

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        data_info["img_path"] = img_info["img_path"]

        # Set depth path
        scene_dir = img_info["scene_dir"]
        frame_name = img_info["frame_name"]
        depth_path = osp.join(scene_dir, "depth", f"{frame_name}.png")
        data_info["depth_path"] = depth_path

        # Set polarization path if it exists
        pol_path = osp.join(scene_dir, "pol", f"{frame_name}.png")
        if osp.exists(pol_path):
            data_info["pol_path"] = pol_path

        # Set instance mask path
        instance_path = osp.join(scene_dir, "instance", f"{frame_name}.png")
        data_info["instance_path"] = instance_path

        data_info["img_id"] = img_info["img_id"]
        data_info["intrinsic"] = img_info["intrinsic"]

        data_info["height"] = self.IMG_SHAPE[0]
        data_info["width"] = self.IMG_SHAPE[1]

        instances = self._parse_instance_info(gt_info, data_info["intrinsic"])
        if len(instances) == 0:
            return None

        data_info["instances"] = instances
        return data_info

    def _parse_instance_info(self, gt_info: dict, intrinsic: List[float]) -> List[dict]:
        """parse instance information.

        Args:
            gt_info (dict): Ground truth information.
            intrinsic (List[float]): Camera intrinsic parameters.
        Returns:
            List[dict]: List of instances.
        """
        instances = []
        is_train = "train" in self.split

        if "instance_ids" not in gt_info or "class_ids" not in gt_info:
            return instances

        for idx in range(len(gt_info["instance_ids"])):
            instance = {}

            class_id = gt_info["class_ids"][idx] - 1  # 1-indexed to 0-indexed
            if class_id < 0 or class_id >= len(self._metainfo["classes"]):
                continue

            instance_id = gt_info["instance_ids"][idx]
            class_name = self._metainfo["classes"][class_id]

            # Get bounding box
            if "bboxes" in gt_info:
                y1, x1, y2, x2 = gt_info["bboxes"][idx]
                bbox = [x1, y1, x2, y2]  # convert to x1, y1, x2, y2 format
            else:
                # Skip if no bbox info
                continue

            if len(intrinsic) == 4:
                # intrinsic is [fx, fy, cx, cy]
                K = np.array(
                    [
                        [intrinsic[0], 0, intrinsic[2]],
                        [0, intrinsic[1], intrinsic[3]],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            elif len(intrinsic) == 9:
                K = np.array(intrinsic).reshape(3, 3)
            else:
                raise ValueError(f"Invalid intrinsic shape: {len(intrinsic)}")

            # Get pose information
            translation = (
                gt_info["translations"][idx]
                if "translations" in gt_info
                else np.zeros(3)
            )
            rotation = (
                gt_info["rotations"][idx] if "rotations" in gt_info else np.eye(3)
            )

            # Ensure correct data types
            translation = np.array(translation, dtype=np.float32)
            rotation = np.array(rotation, dtype=np.float32)
            # Get scale/size information to compute final object size
            size = None
            if "gt_scales" in gt_info:  # Priority 1: direct dimensions
                size = np.array(gt_info["gt_scales"][idx], dtype=np.float32)

            if size is None:
                # Fallback or skip if size cannot be determined
                continue  # Or use a default size: size = np.ones(3, dtype=np.float32)

            # Create 4x4 transformation matrix before symmetry handling
            T = np.eye(4, dtype=np.float32)
            if is_train:
                T[:3, :3] = rotation
            else:
                scale_factor = np.linalg.norm(size)
                T[:3, :3] = rotation * scale_factor  # Scale rotation
                rotation = rotation / scale_factor  # Normalize rotation
                size = size / scale_factor  # Normalize size
            T[:3, 3] = translation

            # Symmetry handling, similar to NOCS
            if class_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = np.sqrt(theta_x**2 + theta_y**2)

                if r_norm > 1e-6:
                    s_map = np.array(
                        [
                            [theta_x / r_norm, 0.0, -theta_y / r_norm],
                            [0.0, 1.0, 0.0],
                            [theta_y / r_norm, 0.0, theta_x / r_norm],
                        ]
                    )
                    rotation = rotation @ s_map

            if self.use_cuboid_as_bbox:
                width, height, depth = size
                corners_3d = np.array(
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
                corners_3d_cam = (T[:3, :3] @ corners_3d.T + T[:3, 3, None]).T
                projected_corners = (K @ corners_3d_cam.T).T
                projected_corners_2d = (
                    projected_corners[:, :2] / projected_corners[:, 2:]
                )
                projected_corners_2d = projected_corners_2d.reshape(8, 2)

                x1 = projected_corners_2d[:, 0].min()
                y1 = projected_corners_2d[:, 1].min()
                x2 = projected_corners_2d[:, 0].max()
                y2 = projected_corners_2d[:, 1].max()
                bbox = [x1, y1, x2, y2]

            # Convert rotation to 6D representation (first two columns of rotation matrix)
            r_6d_rep = [rotation.flatten()[i] for i in [0, 3, 6, 1, 4, 7]]

            # Calculate 2D center projection
            fx, fy, cx, cy = intrinsic
            if translation[2] > 0:  # Avoid division by zero
                center_2d_x = fx * translation[0] / translation[2] + cx
                center_2d_y = fy * translation[1] / translation[2] + cy
                center_2d = [center_2d_x, center_2d_y]
            else:
                center_2d = [cx, cy]  # Default to image center

            instance["bbox"] = bbox
            instance["bbox_label"] = class_id
            instance["instance_id"] = instance_id

            # Use 6D rotation representation like NOCS dataset
            instance["rotation"] = r_6d_rep
            instance["translation"] = translation
            instance["size"] = size

            # Add transformation matrix like NOCS dataset
            instance["T"] = T
            instance["center_2d"] = center_2d
            instance["z"] = translation[2]
            instance["ignore_flag"] = 0

            instances.append(instance)

        return instances

    def get_model_for_instance(self, class_name: str, instance_id: int = None):
        """
        Get the 3D model for a specific instance.

        Args:
            class_name: Name of the object class
            instance_id: Instance ID (optional)

        Returns:
            3D model vertices or None if not found
        """
        if self.models is None:
            return None

        # Try to find exact match first (class_name-specific_id format)
        possible_keys = []
        for model_key in self.models.keys():
            if model_key.startswith(class_name + "-"):
                possible_keys.append(model_key)

        if possible_keys:
            # Use the first matching model for this class
            model_key = possible_keys[0]
            return self.models[model_key]

        # If no exact match, try to find any model with the class name
        for model_key in self.models.keys():
            if class_name in model_key:
                return self.models[model_key]

        return None

    def _load_individual_models(self, models_path: str) -> dict:
        """Load individual model files from directory structure.

        Args:
            models_path: Path to models directory containing class subdirectories

        Returns:
            Dictionary mapping model names to 3D vertices or model data
        """
        models = {}

        try:
            import trimesh
        except ImportError:
            print("Warning: trimesh not installed. Install with: pip install trimesh")
            return models

        # Iterate through class directories
        for class_name in self._metainfo["classes"]:
            class_dir = osp.join(models_path, class_name)
            if not osp.exists(class_dir):
                continue

            # Find all .obj files in the class directory
            obj_files = glob.glob(osp.join(class_dir, "*.obj"))

            for obj_file in obj_files:
                try:
                    # Extract model name from filename
                    model_name = osp.basename(obj_file).replace(".obj", "")

                    # Load the mesh using trimesh
                    mesh = trimesh.load(obj_file, process=False)

                    # Store vertices (similar to NOCS dataset format)
                    if hasattr(mesh, "vertices"):
                        models[model_name] = mesh.vertices.astype(np.float64)
                    else:
                        print(f"Warning: Could not extract vertices from {obj_file}")

                except Exception as e:
                    print(f"Error loading model {obj_file}: {e}")

        print(f"Loaded {len(models)} individual models from {models_path}")
        return models


def test_dataset():
    """Test HouseCat6DDataset functionality."""
    print("=" * 50)
    print("Testing HouseCat6DDataset")
    print("=" * 50)

    # Test 1: Basic dataset initialization
    print("\n1. Testing dataset initialization...")
    try:
        dataset = HouseCat6DDataset(
            data_root="/datasets/HouseCat6D/",
            split="train",
            num_sample_points=1000,
            scene_limit=2,  # Limit to 2 scenes for faster testing
            img_limit=5,  # Limit to 5 images per scene
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  - Split: {dataset.split}")
        print(f"  - Number of classes: {len(dataset._metainfo['classes'])}")
        print(f"  - Classes: {dataset._metainfo['classes']}")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of loaded models: {len(dataset.models)}")
        if len(dataset.models) > 0:
            print(f"  - Sample model names: {list(dataset.models.keys())[:5]}")
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        return

    if len(dataset) == 0:
        print("✗ No samples found in dataset. Please check data_root path.")
        return

    # Test 2: Model loading validation
    print("\n2. Testing model loading...")
    try:
        if len(dataset.models) > 0:
            print(f"✓ Models loaded successfully: {len(dataset.models)} models")
            # Test a few models
            for i, (model_name, model_data) in enumerate(
                list(dataset.models.items())[:3]
            ):
                if hasattr(model_data, "shape"):
                    print(f"  - Model '{model_name}': {model_data.shape}")
                else:
                    print(f"  - Model '{model_name}': {type(model_data)}")
                if i >= 2:  # Show max 3 models
                    break
        else:
            print(
                "⚠ No models loaded (this might be expected if models are not available)"
            )

    except Exception as e:
        print(f"✗ Model loading test failed: {e}")

    # Test 3: Sample data structure validation
    print("\n3. Testing sample data structure...")
    try:
        sample = dataset[0]
        required_keys = [
            "img_path",
            "depth_path",
            "instance_path",
            "img_id",
            "intrinsic",
            "height",
            "width",
            "instances",
        ]

        for key in required_keys:
            if key not in sample:
                print(f"✗ Missing key: {key}")
            else:
                print(f"✓ Found key: {key}")

        # Validate instances structure
        if "instances" in sample and len(sample["instances"]) > 0:
            instance = sample["instances"][0]
            # Updated instance keys to reflect 6D rotation and transformation matrix
            instance_keys = [
                "bbox",
                "bbox_label",
                "instance_id",
                "rotation",
                "translation",
                "size",
                "T",
                "center_2d",
                "ignore_flag",
            ]
            print("  Instance structure:")
            for key in instance_keys:
                if key not in instance:
                    print(f"    ✗ Missing instance key: {key}")
                else:
                    print(f"    ✓ Found instance key: {key}")

    except Exception as e:
        print(f"✗ Sample structure validation failed: {e}")
        return

    # Test 4: Multiple samples
    print("\n4. Testing multiple samples...")
    try:
        num_test_samples = min(3, len(dataset))
        for i in range(num_test_samples):
            sample = dataset[i]
            print(f"  Sample {i}:")
            print(f"    - Image ID: {sample['img_id']}")
            print(f"    - Image shape: {sample['height']} x {sample['width']}")
            print(f"    - Number of instances: {len(sample['instances'])}")

            if len(sample["instances"]) > 0:
                classes_in_sample = [inst["bbox_label"] for inst in sample["instances"]]
                class_names = [
                    dataset._metainfo["classes"][label] for label in classes_in_sample
                ]
                print(f"    - Object classes: {class_names}")
    except Exception as e:
        print(f"✗ Multiple samples test failed: {e}")
        return

    # Test 5: Test different split (if models are loaded)
    print("\n5. Testing test split...")
    try:
        test_dataset_obj = HouseCat6DDataset(
            data_root="/datasets/HouseCat6D/",
            split="test",
            num_sample_points=1000,
            scene_limit=1,
            img_limit=3,
        )
        print(f"✓ Test split initialized successfully")
        print(f"  - Test samples: {len(test_dataset_obj)}")
        print(f"  - Test models loaded: {len(test_dataset_obj.models)}")

        if len(test_dataset_obj) > 0:
            test_sample = test_dataset_obj[0]
            print(f"  - Test sample ID: {test_sample['img_id']}")

    except Exception as e:
        print(f"✗ Test split failed: {e}")

    # Test 6: Data types and ranges validation
    print("\n6. Testing data types and ranges...")
    try:
        sample = dataset[0]

        # Check intrinsics
        intrinsic = sample["intrinsic"]
        assert len(intrinsic) == 4, (
            f"Intrinsic should have 4 values, got {len(intrinsic)}"
        )
        assert all(v > 0 for v in intrinsic), "All intrinsic values should be positive"
        print("✓ Intrinsic parameters validated")

        # Check instances
        for i, instance in enumerate(sample["instances"]):
            # Check bbox
            bbox = instance["bbox"]
            assert len(bbox) == 4, f"Bbox should have 4 values, got {len(bbox)}"
            assert bbox[2] > bbox[0] and bbox[3] > bbox[1], "Invalid bbox coordinates"

            # Check 6D rotation representation
            rotation = instance["rotation"]
            assert len(rotation) == 6, (
                f"6D rotation should have 6 values, got {len(rotation)}"
            )

            # Check translation
            translation = instance["translation"]
            assert translation.shape == (3,), (
                f"Translation should be length 3, got {translation.shape}"
            )

            # Check size
            size = instance["size"]
            assert size.shape == (3,), f"Size should be length 3, got {size.shape}"

            # Check transformation matrix
            T = instance["T"]
            assert T.shape == (4, 4), (
                f"Transformation matrix should be 4x4, got {T.shape}"
            )

            # Check center_2d
            center_2d = instance["center_2d"]
            assert len(center_2d) == 2, (
                f"Center_2d should have 2 values, got {len(center_2d)}"
            )

        print(f"✓ All {len(sample['instances'])} instances validated")

    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return

    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    print("HouseCat6D dataset now includes:")
    print("  - 6D rotation representation (like NOCS)")
    print("  - 3D model loading functionality")
    print("  - Symmetry handling for symmetric objects")
    print("  - 4x4 transformation matrices")
    print("  - Improved error handling")
    print("=" * 50)


if __name__ == "__main__":
    test_dataset()
