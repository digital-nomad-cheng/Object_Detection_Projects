import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7

__all__ = [
    "build_mnv1_backbone",
#    "build_mnv2_backbone"
]

def conv_bn_leaky(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw_leaky(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )

class MobileNetV1(Backbone):
    def __init__(self, cfg, input_channels, width_mult=1.0, out_features=None):
        super().__init__()

        base_channels = 32
        output_channels = int(base_channels * width_mult)

        name = "stem"
        self.stem = conv_bn_leaky(input_channels, output_channels, 2, leaky=0.1)
        current_stride = 2
        self._out_feature_strides = {name: current_stride}
        self._out_feature_channels = {name: output_channels}

        dw_settings = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.return_features_indices = [3, 5, 11, 13]
        self.features = nn.ModuleList([])
        
        input_channels = output_channels
        for c, n, s in dw_settings:
            output_channels = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(conv_dw_leaky(input_channels, output_channels, s))
                else:
                    self.features.append(conv_dw_leaky(input_channels, output_channels, 1))
                input_channels = output_channels

                if len(self.features) in self.return_features_indices:
                    name = "mob{}".format(
                        self.return_features_indices.index(len(self.features)) + 2
                    )
                    self._out_feature_channels.update({
                        name: output_channels
                    })
                    current_stride *= 2
                    self._out_feature_strides.update({
                        name: current_stride
                    })
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                   m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        if freeze_at > 0:
            for p in self.stem.parameters():
                p.requires_grad = False

            if freeze_at > 1:
                # freeze features
                freeze_at = freeze_at - 2
                freeze_layers = self.return_features_indices[freeze_at] \
                    if freeze_at < len(self.return_features_indices) \
                    else self.return_features_indices[-1]
                for layer_index in range(freeze_layers):
                    for p in self.features[layer_index].parameters():
                        p.requires_grad = False
        
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for i, m in enumerate(self.features, 1):
            x = m(x)
            for i in self.return_features_indices:
                name = "mob{}".format(self.return_features_indices.index(i) + 2)
                if name in self._out_features:
                    outputs[name] = x
    
        print(outputs)
        
        return outputs

@BACKBONE_REGISTRY.register()
def build_mnv1_backbone(cfg, input_shape: ShapeSpec):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.MNET.OUT_FEATURES
    width_mult = cfg.MODEL.MNET.WIDTH_MULT
    model = MobileNetV1(cfg, input_shape.channels, width_mult, out_features)
    model.freeze(freeze_at)

    return model
        

