import torch
import torch.nn as nn


class Patch_Embedding(nn.Module):
    """
    Module to split image into patches and embed them
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: [bs, in_channel, img_size, img_size]
        x = self.proj(x)  # [bs, embed_dim, height/patch_size, width/patch_size]
        x = x.flatten(2)  # [bs, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [bs, num_patches, embed_dim]
        return x


class Transformer_Layer(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        ) 
    def forward(self, x):
        # x: [seq_length, batch_size, embed_dim]
        # Self-attention block
        x_norm = self.norm1(x)
        x_att, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout1(x_att)
        # MLP block
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class ViT(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=1,
            num_classes=15,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_dim=768*4,
            dropout=0.0
    ):
        super().__init__()
        self.patch_embed = Patch_Embedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for all patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Stack transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    def forward(self, x):
        # x: [bs, in_channels, img_size, img_size]
        bs = x.shape[0]
        x = self.patch_embed(x)  # [bs, num_patches, embed_dim]
        # Expand class token and concatenate with patch embeddings
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # [bs, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [bs, 1 + num_patches, embed_dim]
        # Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects shape [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        for block in self.transformer_layers:
            x = block(x)
        x = self.norm(x)
        # Use the output corresponding to the class token for classification
        cls_token_final = x[0]  # shape: [bs, embed_dim]
        logits = self.head(cls_token_final)
        return logits


if __name__ == '__main__':
    # Create a dummy input image (batch size 1, 3 channels, 224x224 pixels)
    dummy_input = torch.randn(1, 3, 224, 224)
    # Instantiate the Vision Transformer
    model_args = {
        "img_size": 64,
        "patch_size": 16,
        "in_channels": 3,
        "num_classes": 1000,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_dim": 768 * 4,
        "dropout": 0.0
    }
    model = ViT(**model_args)
    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [1, num_classes]
