import torch
import torch.nn as nn
from transformers import ViTConfig, ViTImageProcessor, ViTForImageClassification


class ViT_PT(nn.Module):

  def __init__(self, in_channels, num_classes, base_model_name="google/vit-base-patch16-224"):
    super().__init__()
    config = ViTConfig.from_pretrained(base_model_name)
    config.num_channels = in_channels
    config.num_labels = num_classes
    config.ignore_mismatched_sizes = True
    self.config = config
    self.tokenizer = ViTImageProcessor.from_pretrained(
        base_model_name, do_resize=False, do_rescale=True, do_normalize=False)
    self.base_model = ViTForImageClassification(config)

  def forward(self, x):
    assert x.shape[2] == self.config.image_size and x.shape[3] == self.config.image_size
    # x: [bs, in_channels, img_size, img_size]
    # Resize
    inputs = self.tokenizer.preprocess(images=x, data_format="channels_first",
                                       input_data_format="channels_first", return_tensors='pt')

    outputs = self.base_model(**inputs)
    return outputs.logits


if __name__ == '__main__':
  # Create a dummy input image (batch size 1, 3 channels, 224x224 pixels)
  dummy_input = torch.randn(1, 3, 224, 224)
  # Instantiate the Vision Transformer
  model_args = {
      "in_channels": 3,
      "num_classes": 1000,
      "base_model_name": "google/vit-base-patch16-224"
  }
  model = ViT_PT(**model_args)
  print(f"Model config: {model.config}")
  print(f"Model classifier: {model.base_model.classifier}")

  # Forward pass
  output = model(dummy_input)
  print("Output shape:", output.shape)  # [1, num_classes]
