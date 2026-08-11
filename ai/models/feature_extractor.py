import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class ResNet18FeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        backbone = resnet18(
            weights=weights
        )

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1

        self.layer2 = backbone.layer2

        self.layer3 = backbone.layer3

        self.layer4 = backbone.layer4

        # Freeze pretrained network
        for parameter in self.parameters():
            parameter.requires_grad = False


    def forward(self, x):

        x = self.layer0(x)

        feature1 = self.layer1(x)

        feature2 = self.layer2(feature1)

        feature3 = self.layer3(feature2)

        feature4 = self.layer4(feature3)

        return {
            "layer1": feature1,
            "layer2": feature2,
            "layer3": feature3,
            "layer4": feature4,
        }