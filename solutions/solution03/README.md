
Assignment:

Finish a debugging report, write down the problems you encounter during the
process and solutions you come up with. You could also write down your new 
understandings.

Build from source:
```
conda create -n .detectron2
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
git clone https://github.com/facebookresearch/detectron2
python -m pip install -e detectron2
```


Training Arguments:
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
     + build rpn: 
       rpn architecture according to the paper. In the paper the objectness score is num_anchors*2, \
       while in detectron2 implementation is numm_anchors
       ```python
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
       ```
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
       roi_heads 根据 anchors 原始位置加上 rpn 预测的偏移量，提取 proposals.\
       然后根据 proposal 映射 feature map 的特征区域，然后通过 RoIPooling 将 featuremap 处理成固定大小。\
       通过增强后的特征地候选区域进行类别和框位置预测。
       ```
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
       ```
       
2. build optimizer
   构建优化器
   ```
   def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
    )
   ```

3. build dataloader
   ```
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )
   ```
   
4. build scheduler
   构建 learning rate 调节器
   ```
   self.scheduler = self.build_lr_scheduler(cfg, optimizer)
   ```

5. build checkpointer
   构建中间模型保存器
   ```
   self.checkpointer = DetectionCheckpointer(
        model, 
        cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=self.scheduler,
   )
   ```



