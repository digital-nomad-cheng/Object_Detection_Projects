
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


cfg 是整个项目的配置文件，控制 Trainer 的构建。训练逻辑主要在 Trainer 里面, 主要分为5个部分。

1. build model: 整个网络 forward 的核心逻辑在 rcnn.py 里面 
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
     + build rpn: \
       rpn 使用 fpn 输出的特征和ground truch来输出 proposal \
       rpn 使用 anchor generator 根据 feature map 的大小来生产 anchors
       ```
       anchors = self.anchor_generator(features)
       ```
       rpn 使用输入特征预测每个 anchor 的类别和偏移
       ```
       pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
       ```
       rpn 使用 anchors 和 ground truth label 进行 IoU 计算，根据 Matcher 中的规则将 anchor 和 ground truth 类别匹配。\
       从而得到 rpn 计算 loss 所需要的类别信息。 
       ```
        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
        ```
     + build roi_heads
       roi_heads 根据 anchors 原始位置加上 rpn 预测的偏移量，     
    
2. build optimizer

3. build dataloader

4. build scheduler

5. build checkpointer



