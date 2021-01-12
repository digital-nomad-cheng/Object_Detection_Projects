


## Prepare dataset

1. Generate COCO format annotation of wider face.
2. Register wider face into detectron2.

## Build RetinaFace

## Inference
```python
python demo/demo.py \
--config-file configs/retinaface/retinaface_mnv1_FPN.yaml \
--input datasets/widerface/val/images/*/*.jpg \
--output work_dirs/retinaface_mnv1_FPN/val  \
--opts MODEL.WEIGHTS work_dirs/retinaface_mnv1_FPN/model_final.pth
```

## Results
MobileNetV1
==================== Results ====================
Easy   Val AP: 0.8683708233826524
Medium Val AP: 0.8233959503487284
Hard   Val AP: 0.5829218604866397
=================================================


## Reference
1. https://github.com/yi-ming-lin/widerface-coco-convertor
