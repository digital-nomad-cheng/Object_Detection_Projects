import os
import os.path as osp
import argparse
import json
from collections import defaultdict
from PIL import Image

"""
Convert wider face format data to COCO format.
The wider face annotation comes from here: 
https://github.com/deepinsight/insightface/issues/971
https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0
We could also generate from the original wider face format, in which case
this script has to be modified accordingly.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert widerface format annotation to coco-style annotation."
    )
    parser.add_argument(
        '--ann_file', type=str, default="./datasets/widerface/train/label.txt", 
        help="annotation file of widerface dataset."
    )
    parser.add_argument(
        '--img_path', type=str, default="./datasets/widerface/train/images", 
        help="path for widerface training images."
    )
    args = parser.parse_args()
    return args


def is_valid_image(name, IMG_EXTENSIONS=('.jpg', '.jpeg', '.png')):
    return name.lower().endswith(IMG_EXTENSIONS)

def main(args):

    # COCO 
    CATEGORIES = [
        {'id': 1, 'name': 'face'},
    ]
    coco_output = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # Process image infos
    # image path -> image id
    image_path_to_id = {}
    img_id = 1
    d_names  = [d.name for d in os.scandir(args.img_path) if d.is_dir()]
    for d_name in d_names:
        img_names = [e for e in os.listdir(osp.join(args.img_path, d_name)) if is_valid_image(e)]
        for img_name in img_names:
            # Add item
            try:
                img = Image.open(osp.join(args.img_path, d_name, img_name))
            except :
                print("Skip image: ", osp.join(args.img_path, d_name, img_name))
                continue
            width, height = img.size
            relative_filepath = '/'.join([d_name, img_name])
            item = {
                'id': img_id,
                'height': height,
                'width': width,
                'file_name': relative_filepath,
            }
            coco_output['images'].append(item)
            image_path_to_id[relative_filepath] = img_id
            img_id += 1

    print("There are {} images in training set".format(img_id - 1))

    # Process labels
    ann_id = 1
    # Read original labels
    lines = open(args.ann_file).readlines()
    current_imgpath = None
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            path = line[2:]
            current_imgpath = path
        else:
            # one line for one instance
            line = line.split(' ')
            label = [float(x) for x in line]
            # bbox: [x,y,w,h]
            bbox = label[:4]

            # keypoints: [x1,y1,v1, ...]
            keypoints = label[4:19]
            v = 2
            if label[4] < 0:
                # when the value == -1, set it to not labeled
                v = 0
            for _i, _k in enumerate(keypoints):
                if _i % 3 != 2:
                    # NOTE built-in coco loader will add shift 0.5
                    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py#L184 
                    keypoints[_i] = _k - 0.5
                else:
                    keypoints[_i] = v
            
            img_id = image_path_to_id[current_imgpath]
            area = bbox[2] * bbox[3]

            # segmentation
            segm = [[
                bbox[0], bbox[1],
                bbox[0] + bbox[2], bbox[1],
                bbox[0] + bbox[2], bbox[1] + bbox[3],
                bbox[0], bbox[1] +bbox[3]
            ]]

            # NOTE detectron2 wo data filtering strategy
            if bbox[2] * bbox[3] < 32:
                # NOTE filter small faces, min step is 8
                continue

            item = {
                'id': ann_id,
                'image_id': img_id,
                'bbox': bbox,
                'segmentation': segm,
                'keypoints': keypoints,
                'num_keypoints': 5,
                'category_id': 1, # face
                'iscrowd': 0, 
                'area': area,
            }
            coco_output['annotations'].append(item)
            ann_id += 1

    print("There are {} annotations".format(ann_id - 1))
            
    # Save translated coco annotations
    save_dir = osp.dirname(args.wface)
    save_file = osp.join(save_dir, "widerface_coco.json")
    with open(save_file, 'w') as w_obj:
        json.dump(coco_output, w_obj)

    print("COCO json annotation is saved to {}".format(save_file))

if __name__ == "__main__":
    args = parse_args()
    main(args)
