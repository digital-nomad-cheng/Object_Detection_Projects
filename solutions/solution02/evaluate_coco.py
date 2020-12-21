from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# RetinaNet
retina_cfg = get_cfg()
retina_cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_50_FPN_1x.yaml'))
retina_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
retina_cfg.DATALOADER.NUM_WORKERS = 8
model = build_model(retina_cfg)
DetectionCheckpointer(model).load(retina_cfg.MODEL.WEIGHTS)
val_loader = build_detection_test_loader(retina_cfg, "coco_2017_val")
evaluator = COCOEvaluator('coco_2017_val', retina_cfg, False, output_dir="./output/")
inference_on_dataset(model, val_loader, evaluator)

# Faster RCNN
rcnn_cfg = get_cfg()
rcnn_cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'))
rcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
rcnn_cfg.DATALOADER.NUM_WORKERS = 8
model = build_model(rcnn_cfg)
DetectionCheckpointer(model).load(rcnn_cfg.MODEL.WEIGHTS)
val_loader = build_detection_test_loader(rcnn_cfg, "coco_2017_val")
evaluator = COCOEvaluator('coco_2017_val', rcnn_cfg, False, output_dir="./output/")
inference_on_dataset(model, val_loader, evaluator)
