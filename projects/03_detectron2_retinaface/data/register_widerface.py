import os
import os.path as osp
import copy

from detectron2.data.datasets import register_coco_instances

"""
Register wider face dataset
"""
WIDERFACE_KEYPOINT_NAMES = (
    "left_eye", "right_eye", "nose", "left_mouth", "right_mouth"
)

WIDERFACE_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"), ("left_mouth", "right_mouth")
)

widerface_metadata = {
    "thing_classes": ["face"],
    "keypoint_names": WIDERFACE_KEYPOINT_NAMES,
    "keypoint_flip_map": WIDERFACE_KEYPOINT_FLIP_MAP,
}

# root = os.getenv("DETECTRON2_DATASETS", "datasets")
root = "/home/idealabs/data/opensource_dataset/WIDER/"
widerface_train_image_root = osp.join(root, "train/images")
widerface_train_annotation_file = osp.join(root, "train/widerface_coco.json")
register_coco_instances("widerface_train", widerface_metadata, widerface_train_annotation_file, widerface_train_image_root)
