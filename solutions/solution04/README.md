# setup config
1. anchor generator sizes
    ```yaml
    ANCHOR_GENERATOR:
        SIZES: !!python/object/apply:eval ["[[x, x * 2**(1.0/3), x * 2**(2.0/3) ] for x in [32, 64, 128, 256, 512 ]]"]
    ```

2. _C.MODEL.RETINANET.NUM_CONVS = 4

3. _C.MODEL.RETINANET.PRIOR_PROB = 0.01: for stable training

# setup training
```python
model = self.build_model(cfg)
optimizer = self.build_optimizer(cfg, model)
data_loader = self.build_train_loader(cfg)
```
## build model

1. build retinaface  
   ```python
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
   ```
   LastLevelP6P7: use two stride two convolutions to further downsample the feature map.
   ```python
   class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
   ```
   
2. build resnet

3. build backbone

## build optimizer
```python
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
## build data loader
Aspect ratio grouping
Mapper data augmentation
## build lr_scheduler
```python
def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
```

# training
