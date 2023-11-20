
import torch
from torch import nn
from torchvision import models
from torch.nn.modules.pooling import AdaptiveMaxPool2d


class ResNet(nn.Module):
    def __init__(self, layers=18, num_class=2, weights=None):
        super(ResNet, self).__init__()
        if layers == 18:
            self.resnet = models.resnet18(weights=weights)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif layers == 34:
            self.resnet = models.resnet34(weights=weights)
        elif layers == 50:
            self.resnet = models.resnet50(weights=weights)
        elif layers == 101:
            self.resnet = models.resnet101(weights=weights)
        elif layers == 152:
            self.resnet = models.resnet152(weights=weights)
        else:
            raise ValueError('layers should be 18, 34, 50, 101.')
        self.num_class = num_class
        if layers in [18, 34]:
            self.fc = nn.Linear(512, num_class)
        if layers in [50, 101, 152]:
            self.fc = nn.Linear(512 * 4, num_class)

    def conv_base(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.conv_base(x)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SPPNet(nn.Module):
    def __init__(self, backbone=18, num_classes=2, pool_size=(1, 2, 6), weights=None):
        # Only resnet is supported in this version
        super(SPPNet, self).__init__()
        self.backbone=backbone
        self.n_classes=num_classes
        self.pool_size=pool_size
        self.weights=weights
        if backbone in [18, 34, 50, 101, 152]:
            self.resnet = ResNet(backbone, num_classes, weights)
        else:
            raise ValueError('Resnet{} is not supported yet.'.format(backbone))

        if backbone in [18, 34]:
            self.c = 512
        if backbone in [50, 101, 152]:
            self.c = 2048

        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        num_features = self.c * (pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2)
        self.classifier = nn.Linear(2*num_features, num_classes)

    def forward(self, x, y):
        _, _, _, x = self.resnet.conv_base(x)
        x = self.spp(x)
        _, _, _, y = self.resnet.conv_base(y)
        y = self.spp(y)
        output = torch.cat((x, y), 1)
        output = self.classifier(output)
        return output


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            max_pool = AdaptiveMaxPool2d(output_size=(n, n))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out