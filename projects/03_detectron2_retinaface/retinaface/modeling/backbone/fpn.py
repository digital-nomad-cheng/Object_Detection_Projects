from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.layers import ShapeSpec
from .mobilenet import build_mnv1_backbone

__all__ = ['build_mnv1_fpn_wo_top_block_backbone',
           'build_resnet_fpn_wo_top_block_backbone']


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_wo_top_block_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE
    )

    return backbone


@BACKBONE_REGISTRY.register()
def build_mnv1_fpn_wo_top_block_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_mnv1_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE
    )

    return backbone
