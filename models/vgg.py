import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class VGG(nn.Module):

    class Variant(Enum):
        VGG11 = '11'
        VGG13 = '13'
        VGG16 = '16'
        VGG19 = '19'

    _architectures = {
        Variant.VGG11: [
            64, 'M',
            128, 'M',
            256, 256, 'M',
            512, 512, 'M',
            512, 512, 'M'
        ],
        Variant.VGG13: [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 'M',
            512, 512, 'M',
            512, 512, 'M'
        ],
        Variant.VGG16: [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M'
        ],
        Variant.VGG19: [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'
        ]
    }

    def __init__(self, in_channels, num_classes, variant='16'):
        super(VGG, self).__init__()
        if not isinstance(variant, self.Variant):
            variant = self.Variant(variant)
        self.conv_layers = self._make_conv_layers(
            in_channels,
            self._architectures[variant]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self._init_weights()

    def _make_conv_layers(self, in_channels, arch):
        layers = []
        channels = in_channels
        for x in arch:
            if type(x) == int:
                layers.append(nn.Conv2d(channels, x, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                channels = x
            elif x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"VGG unknown layer: {x}")
        layers.append(nn.AdaptiveAvgPool2d((7, 7)))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    # Create a dummy input image (batch size 1, 3 channels, 224x224 pixels)
    dummy_input = torch.randn(1, 3, 224, 224)
    # Instantiate the Vision Transformer
    model_args = {
        "in_channels": 3,
        "num_classes": 1000,
    }
    model = VGG(**model_args)
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [1, num_classes]
