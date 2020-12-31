import torch
import torch.nn as nn

def ConvBN(inp, oup, kernel_size=3, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup)
    )

def ConvBNReLU(inp, oup, kernel_size=3, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def DepthWiseConv(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        # depthwise
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        
        # pointwise
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class FPN(nn.Module):
    """
    Feature Pyramid Network
    """
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.lateral1 = ConvBNReLU(in_channels_list[0], out_channels, 1, 1, leaky)
        self.lateral2 = ConvBNReLU(in_channels_list[1], out_channels, 1, 1, leaky)
        self.lateral3 = ConvBNReLU(in_channels_list[2], out_channels, 1, 1, leaky)

        self.out_conv1 = ConvBNReLU(out_channels, out_channels, 3, leaky=leaky)
        self.out_conv2 = ConvBNReLU(out_channels, out_channels, 3, leaky=leaky)

    def forward(self, features):
        """
        features: dict, dict of features from backbone network
        """
        
        # Todo: compare with official implementation
        inputs = features.values()
        
        output1 = self.lateral1(inputs[0])
        output2 = self.lateral2(inputs[1])
        output3 = self.lateral3(inputs[2])
        
        up3 = F.interpolate(output3, size=(output2.size(2), output2.size(3)))
        output2 = output2 + up3
        output2 = self.out_conv2(output2)

        up2 = F.interpolate(output2, size=(output1.size(2), output1.size(3)))
        output1 = output1 + up2
        output1 = self.out_conv1(output1)

        return [output1, output2, output3]

class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        
        self.conv3x3 = ConvBN(in_channels, out_channels//2, stride=1)
        
        self.conv5x5 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels//4, stride=1, leaky=leaky),
            ConvBN(out_channels//4, out_channels//4, stride=1)
        )

        self.conv7x7 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels//4, stride=1, leaky=leaky),
            ConvBNReLU(out_channels//4, out_channels//4, stride=1, leaky=leaky),
            ConvBN(out_channels//4, out_channels//4, stride=1)
        )

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        conv7x7 = self.conv7x7(x)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = F.relu(out)

        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 8, 3, 2, leaky=0.1),
            DepthWiseConv(8, 16, 1),
            DepthWiseConv(16, 32, 2),
            DepthWiseConv(32, 32, 1),
            DepthWiseConv(32, 64, 2),
            DepthWiseConv(64, 64, 1),
        )
        
        self.stage2 = nn.Sequential(
            DepthWiseConv(64, 128, 2),
            DepthWiseConv(128, 128, 1),
            DepthWiseConv(128, 128, 1),
            DepthWiseConv(128, 128, 1),
            DepthWiseConv(128, 128, 1),
            DepthWiseConv(128, 128, 1),
        )

        self.stage3 = nn.Sequential(
            DepthWiseConv(128, 256, 2),
            DepthWiseConv(256, 256, 1)
        )
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)

        return x



