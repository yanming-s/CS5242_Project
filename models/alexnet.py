# Build an AlexNet model using Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(AlexNet, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

    self.fc_layers = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),

        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),

        nn.Linear(4096, num_classes)
    )

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
  model = AlexNet(**model_args)
  # Forward pass
  output = model(dummy_input)
  print("Output shape:", output.shape)  # [1, num_classes]
