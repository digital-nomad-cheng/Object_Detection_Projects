from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
import torchvision

from nets.modules import MobileNetV1, FPN, SSH

class ClassHead(nn.Module):
    def __init__(self, in_channels, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channels, self.num_anchors*2, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors*4, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors*10, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg.MODEL.backbone == "mobilenet0.25":
            backbone = MobileNetV1()
            if cfg.TRAIN.pretrained:
                ckpt = torch.load('./pretrained_weights/mobilenetv10.25.pth',
                    map_location=torch.device('cpu')
                )
                state_dict = OrderedDict()
                for k, v in ckpt['state_dict'].items():
                    name = k[7:] # remove module
                    state_dict[name] = v
                backbone.load_state_dict(state_dict)
        elif cfg.MODEL.backbone == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=cfg['pretrained'])
        
        self.features = IntermediateLayerGetter(backbone, cfg.MODEL.return_layers)
        in_channels_list = cfg.MODEL.in_channels_list
        out_channels = cfg.MODEL.out_channels
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.cls_heads = self._make_cls_head(3, in_channels=out_channels)
        self.bbox_heads = self._make_bbox_head(3, in_channels=out_channels)
        self.ldmk_heads= self._make_ldmk_head(3, in_channels=out_channels)
        
    def _make_cls_head(self, num_fpn=3, in_channels=64, num_anchor=2):
        cls_heads = nn.ModuleList()
        for i in range(num_fpn):
            cls_heads.append(ClassHead(in_channels, num_anchor))
        
        return cls_heads

    def _make_bbox_head(self, num_fpn=3, in_channels=64, num_anchor=2):
        bbox_heads = nn.ModuleList()
        for i in range(num_fpn):
            bbox_heads.append(BboxHead(in_channels, num_anchor))
    
        return bbox_heads

    def _make_ldmk_head(self, num_fpn=3, in_channels=64, num_anchor=2):
        ldmk_heads = nn.ModuleList()
        for i in range(num_fpn):
            ldmk_heads.append(LandmarkHead(in_channels, num_anchor))
        
        return ldmk_heads

    def forward(self, inputs):
        features = self.features(inputs)
        fpn = self.fpn(features)

        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        classification = torch.cat(
            [self.cls_heads[i](f) for i, f in enumerate(features)], dim=1
        )
        bbox_regression = torch.cat(
            [self.bbox_heads[i](f) for i, f in enumerate(features)], dim=1
        )
        ldmk_regression = torch.cat(
            [self.ldmk_heads[i](f) for i, f in enumerate(features)], dim=1
        )

        if self.phase == 'train':
            return (classification, bbox_regression, ldmk_regression)
            
        return (F.softmax(classification, dim=-1), bbox_regression, ldmk_regression)



