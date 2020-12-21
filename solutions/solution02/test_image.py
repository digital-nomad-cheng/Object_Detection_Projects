import os
import glob

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


def visualize(cfg, img_path:str, result_path:str = None):
    predictor = DefaultPredictor(cfg)
    images = glob.glob(img_path+"/*.jpeg")
    metadata = MetadataCatalog.get("coco_2017_train")

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for image_name in images:
        img = cv2.imread(image_name)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(result_path+'/'+image_name.split('/')[-1], out.get_image()[:,:,::-1])

        # cv2.imshow("result", out.get_image()[:, :, ::-1])
        # cv2.waitKey(2000)


# RetinaNet
retinanet_cfg = get_cfg()
retinanet_cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_50_FPN_1x.yaml'))
retinanet_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
retinanet_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
visualize(retinanet_cfg, "test_images", "retinanet_result")

# Faster RCNN
rcnn_cfg = get_cfg()
rcnn_cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'))
rcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
visualize(rcnn_cfg, "test_images", "rcnn_result")
