import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import EncoderBlock


class ConvolutionalStem(nn.Module):
    """
    Convolutional stem to efficiently reduce high-resolution images before feeding them to the transformer
    """
    def __init__(self, in_channels, embed_dim, img_size=1024):
        super().__init__()
        # A series of convolutional layers to gradually reduce the resolution
        # Using a sequence of conv layers with strides to reduce to 64x64
        self.conv_stem = nn.Sequential(
            # Stage 1: 1024x1024 -> 512x512
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2: 512x512 -> 256x256
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3: 256x256 -> 128x128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Stage 4: 128x128 -> 64x64
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Calculate the number of patches after convolution stem
        self.final_resolution = img_size // 16  # After 4 stride-2 convolutions
        self.num_patches = self.final_resolution ** 2
        
    def forward(self, x):
        # x: [bs, in_channels, img_size, img_size]
        x = self.conv_stem(x)  # [bs, embed_dim, img_size//16, img_size//16]
        x = x.flatten(2)  # [bs, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [bs, num_patches, embed_dim]
        return x


class ViT_Pre(nn.Module):
    """
    Vision Transformer with a convolutional stem for high-resolution images
    """
    def __init__(
            self,
            img_size=1024,
            in_channels=1,  # Grayscale images
            num_classes=15,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            dropout=0.0,
            attn_dropout=0.0,
            use_pretrained_blocks=True
    ):
        super().__init__()
        # Convolutional stem to handle high-resolution input
        self.conv_stem = ConvolutionalStem(in_channels, embed_dim, img_size)
        num_patches = self.conv_stem.num_patches
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Create transformer blocks - either from scratch or based on pretrained model
        mlp_dim = embed_dim * mlp_ratio
        if use_pretrained_blocks:
            # Load a pretrained ViT model (without classification head)
            pretrained_vit = models.vit_b_16(weights='DEFAULT')
            # Use the pretrained encoder blocks
            self.transformer_layers = pretrained_vit.encoder.layers
            # We need to adjust position embeddings later since our num_patches is different
        else:
            # Create transformer blocks from scratch
            self.transformer_layers = nn.ModuleList([
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout,
                    attention_dropout_rate=attn_dropout
                )
                for _ in range(depth)
            ])
        # Layer normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def interpolate_pos_encoding(self, x):
        """
        Interpolate positional encoding to match the number of patches.
        Useful when using pretrained position embeddings with different sized inputs.
        """
        npatch = x.shape[1] - 1
        pos_embed = self.pos_embed
        # Separate class token and patch embeddings
        class_pos_embed = pos_embed[:, 0:1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        # Interpolate patch embeddings if sizes don't match
        if npatch != patch_pos_embed.shape[1]:
            dim = x.shape[-1]
            patch_height = int(self.conv_stem.final_resolution)
            patch_width = int(self.conv_stem.final_resolution)
            # Reshape the position embedding to a 2D grid
            patch_pos_embed = patch_pos_embed.reshape(1, int(patch_pos_embed.shape[1]**0.5), 
                                                   int(patch_pos_embed.shape[1]**0.5), dim)
            # Use bilinear interpolation to resize to new dimensions
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.permute(0, 3, 1, 2),
                size=(patch_height, patch_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(1, -1, dim)
            # Combine class token with resized patch embeddings
            pos_embed = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        return pos_embed
        
    def forward(self, x):
        # x: [bs, in_channels, img_size, img_size]
        bs = x.shape[0]
        # Process the input image through convolutional stem
        x = self.conv_stem(x)  # [bs, num_patches, embed_dim]
        # Add class token
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # [bs, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [bs, 1 + num_patches, embed_dim]
        # Add positional embeddings (with potential interpolation)
        pos_embed = self.interpolate_pos_encoding(x)
        x = x + pos_embed
        x = self.pos_drop(x)
        # Pass through transformer blocks
        for block in self.transformer_layers:
            x = block(x)
        # Apply layer normalization
        x = self.norm(x)
        # Use class token for classification
        x_cls = x[:, 0]  # [bs, embed_dim]
        # Classification head
        logits = self.head(x_cls)
        return logits


if __name__ == '__main__':
    # Create a dummy input image (batch size 1, 1 channel (grayscale), 1024x1024 pixels)
    dummy_input = torch.randn(16, 1, 224, 224)
    # Instantiate the Vision Transformer with convolutional stem
    model_args = {
        "img_size": 224,
        "in_channels": 1,  # Grayscale
        "num_classes": 15,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "use_pretrained_blocks": True  # Use pretrained ViT blocks
    }
    model = ViT_Pre(**model_args)
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [1, num_classes]
