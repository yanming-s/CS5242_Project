import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from time import time
import wandb

from models.vit import Transformer_Layer, Patch_Embedding


class ViT_Encoder(nn.Module):
    """
    ViT Encoder for VAE
    """
    def __init__(self, img_size, patch_size, in_channels, latent_dim,
                 embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        self.patch_embed = Patch_Embedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # Learnable positional embeddings for all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Stack transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # VAE specific: project to latent space
        self.ll_mu = nn.Linear(embed_dim * num_patches, latent_dim)
        self.ll_var = nn.Linear(embed_dim * num_patches, latent_dim)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        # x: [bs, in_channels, img_size, img_size]
        bs = x.shape[0]
        x = self.patch_embed(x)  # [bs, num_patches, embed_dim]
        # Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects shape [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        for block in self.transformer_layers:
            x = block(x)
        x = self.norm(x)
        # Reshape for the VAE latent space projection
        x = x.transpose(0, 1)  # Back to [bs, num_patches, embed_dim]
        x = x.reshape(bs, -1)  # [bs, num_patches * embed_dim]
        # Get latent space parameters
        mu = self.ll_mu(x)
        log_var = self.ll_var(x)
        return mu, log_var


class ViT_Decoder(nn.Module):
    """
    ViT Decoder for VAE
    """
    def __init__(self, img_size, patch_size, out_channels, latent_dim,
                 embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        # Project from latent space to patches
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim * self.num_patches)
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Stack transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Projection to image patches
        self.patch_proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, z):
        # z: [bs, latent_dim]
        bs = z.shape[0]
        # Project from latent space to patch embeddings
        x = self.latent_to_embed(z)  # [bs, num_patches * embed_dim]
        x = x.view(bs, self.num_patches, -1)  # [bs, num_patches, embed_dim]
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        for block in self.transformer_layers:
            x = block(x) 
        x = self.norm(x)
        # Back to [bs, num_patches, embed_dim]
        x = x.transpose(0, 1)
        # Project to patch pixels
        x = self.patch_proj(x)  # [bs, num_patches, patch_size * patch_size * out_channels]
        # Reshape to image
        patches_side = self.img_size // self.patch_size
        x = x.view(bs, patches_side, patches_side, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(bs, self.out_channels, self.img_size, self.img_size)
        # Apply sigmoid to get values in [0, 1]
        x = torch.sigmoid(x)
        return x


class ViT_VAE(nn.Module):
    """
    Vision Transformer VAE
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=3, latent_dim=512,
                 embed_dim=512, encoder_depth=6, decoder_depth=6,
                 num_heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = ViT_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        # Decoder
        self.decoder = ViT_Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=in_channels,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encoder
        mu, log_var = self.encoder(x)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        # Decoder
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def encode(self, x):
        """
        Encode input to latent space
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def decode(self, z):
        """
        Decode from latent space
        """
        return self.decoder(z)
    
    def sample(self, num_samples=1, device='cuda'):
        """
        Sample from the latent space and decode
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(recon_x, x, mu, log_var, kld_weight=0.01):
    """
    VAE loss function = Reconstruction loss + KL divergence
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Total loss
    loss = recon_loss + kld_weight * kld_loss
    return loss, recon_loss, kld_loss


class Train_Module(LightningModule):
    """
    PyTorch Lightning module for ViT-VAE Training
    """
    def __init__(self, model, train_loader, val_loader, test_loader, lr=1e-4,
                 kld_weight=0.01, scheduler="plateau", save_ckpt=True):
        super().__init__()
        # Module setup
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.kld_weight = kld_weight
        if scheduler not in ["cosine", "plateau"]:
            raise ValueError(f"Invalid scheduler: {scheduler}. Must be 'cosine' or 'plateau'.")
        self.scheduler = scheduler
        self.save_ckpt = save_ckpt
        # Running loss
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        # Running time
        self.epoch_start_time = None
    
    def training_step(self, batch, _):
        noisy_x, clean_x = batch
        recon_x, mu, log_var = self.model(noisy_x)
        loss, _, _ = vae_loss(recon_x, clean_x, mu, log_var, self.kld_weight)
        self.train_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.print(
            f"Epoch {self.current_epoch + 1} / {self.trainer.max_epochs}"
            f" - Train Loss: {avg_loss.item():.6f}"
            f" - Time: {(time() - self.epoch_start_time) / 60:.2f} mins"
        )
        wandb.log({"train_loss": avg_loss.item()})
        self.train_step_outputs.clear()
        # Save model
        if self.save_ckpt and (self.current_epoch + 1) % 50 == 0:
            save_path = f"checkpoints/model_5_{self.current_epoch + 1}.pth"
            torch.save(self.model.state_dict(), save_path)

    def validation_step(self, batch, _):
        noisy_x, clean_x = batch
        recon_x, mu, log_var = self.model(noisy_x)
        loss, _, _ = vae_loss(recon_x, clean_x, mu, log_var, self.kld_weight)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_step_outputs).mean()
        self.print(f"\n>>> Val Loss: {avg_loss.item():.6f}\n")
        self.log("val_loss", avg_loss.item())
        wandb.log({"val_loss": avg_loss.item()})
        self.val_step_outputs.clear()

    def test_step(self, batch, _):
        noisy_x, clean_x = batch
        recon_x, mu, log_var = self.model(noisy_x)
        loss, _, _ = vae_loss(recon_x, clean_x, mu, log_var, self.kld_weight)
        self.test_step_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        self.print(f"\n### Test Loss: {avg_loss.item():.6f}")
        wandb.log({"test_loss": avg_loss.item()})
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        if self.scheduler == "cosine":
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6,
            )
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
        else:
            # Reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=False
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
