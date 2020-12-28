
Assignment:

Finish a debuging report, write down the problems you encounter during the
process and solutions you come up with. You could also write down your new 
understandings.

Arguments:
```
--config-file 
./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml 
--num-gpus 1 
SOLVER.IMS_PER_BATCH 6 
INPUT.MIN_SIZE_TRAIN (800,) 
DATASETS.TRAIN ('coco_2017_val',) 
DATALOADER.NUM_WORKERS 0
```


cfg 是整个项目的配置文件，控制 Trainer 的构建。训练逻辑主要在 Trainer 里面。主要分为5个部分。

1. build model
    + build resnet: resnet 作为 backbone 特征提取器，输出特征对应 
        ```
        bottom_up = build_resnet_backbone(cfg, input_shape)
        ```
    + build fpn: \
      fpn 使用 resnet 输出的特征, 构建 feature pyramid。 \
      cfg.MODEL.FPN.IN_FEATURES 和 cfg.MODEL.RESNET.OUT_FEATURES 对应。 \
      LastLevelMaxPool() 是在原始 resnet 输出特征之上在通过 max pool 叠加一个 stride 2 feature map。
        ```
        backbone = FPN(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.MODEL.FPN.NORM,
            top_block=LastLevelMaxPool(),
            fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        )
        ```
      

2. build optimizer

3. build dataloader

4. build scheduler

5. build checkpointer



