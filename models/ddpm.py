import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np
import os
import os.path as osp
from time import time
import wandb

from models.vit import Transformer_Layer, Patch_Embedding


class ViT_Denoise_Net(nn.Module):
    """
    Vision Transformer model to predict noise in DDPM
    """
    def __init__(self, img_size, patch_size, in_channels, time_embed_dim,
                 embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Patch embedding
        self.patch_embed = Patch_Embedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # Learnable positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Stack transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Output projection to predict noise
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels),
        )
        # Image reconstruction parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, t_emb):
        # x: [bs, in_channels, img_size, img_size]
        # t_emb: [bs, time_embed_dim]
        bs = x.shape[0]
        # Embed patches
        x = self.patch_embed(x)  # [bs, num_patches, embed_dim]
        # Process time embedding
        time_emb = self.time_embed(t_emb)  # [bs, embed_dim]
        time_emb = time_emb.unsqueeze(1)  # [bs, 1, embed_dim]
        # Add positional embeddings and time embeddings
        x = x + self.pos_embed
        x = x + time_emb  # Broadcasting time embedding to all patches
        x = self.pos_drop(x)
        # Transformer expects shape [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        for block in self.transformer_layers:
            x = block(x)
        x = self.norm(x)
        # Back to [bs, num_patches, embed_dim]
        x = x.transpose(0, 1)
        # Project to noise prediction
        patches = self.out_proj(x)  # [bs, num_patches, patch_size * patch_size * in_channels]
        # Reshape to image
        patches_side = self.img_size // self.patch_size
        x = patches.view(bs, patches_side, patches_side, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(bs, self.in_channels, self.img_size, self.img_size)
        return x


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model with Vision Transformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, time_embed_dim=128,
                 embed_dim=512, depth=6, num_heads=8, mlp_dim=512*4, dropout=0.0,
                 timesteps=250, beta_schedule="linear"):
        super().__init__()
        # Diffusion parameters
        self.timesteps = timesteps
        # Define beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == "cosine":
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        # Pre-calculate diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
        # Time embedding
        self.time_embed_dim = time_embed_dim
        # Noise prediction network
        self.noise_predictor = ViT_Denoise_Net(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
    
    def get_time_embedding(self, t, device):
        """
        Create sinusoidal time embeddings
        """
        half_dim = self.time_embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_embed_dim % 2 == 1:  # Zero pad if odd dimension
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb
    
    def noising_process(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) = N(sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        # Reshape for proper broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_sample(self, x_t, t, t_index):
        """
        Reverse diffusion sampling process for one timestep
        Sample from p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])
        # Use the model to predict the mean
        t_emb = self.get_time_embedding(t, x_t.device)
        predicted_noise = self.noise_predictor(x_t, t_emb)
        # No noise when t == 0
        if t_index == 0:
            sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
            return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * predicted_noise
        # Mean of the posterior
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, batch_size, img_size, in_channels, device):
        """
        Generate samples
        """
        shape = (batch_size, in_channels, img_size, img_size)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    def forward(self, x):
        """
        Forward pass of the DDPM model - training
        Randomly sample a timestep t for each image in the batch
        """
        batch_size = x.shape[0]
        device = x.device
        # Sample a random timestep for each image
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        # Generate random noise
        noise = torch.randn_like(x)
        # Forward diffusion process
        x_noisy, target = self.noising_process(x, t, noise)
        # Get time embeddings
        t_emb = self.get_time_embedding(t, device)
        # Predict noise using the ViT model
        predicted_noise = self.noise_predictor(x_noisy, t_emb)
        return predicted_noise, target


class DDPM_Module(LightningModule):
    """
    PyTorch Lightning module for DDPM Training
    """
    def __init__(self, model: DDPM, train_loader, lr=5e-5, scheduler="plateau",
                 save_ckpt=True, save_every_epoch=10, save_dir="checkpoints", max_grad_norm=1.0):
        super().__init__()
        # Module setup
        self.model = model
        self.train_loader = train_loader
        # Learning settings
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        if scheduler not in ["cosine", "plateau"]:
            raise ValueError(f"Invalid scheduler: {scheduler}. Must be 'cosine' or 'plateau'.")
        self.scheduler = scheduler
        # Save settings
        self.save_dir = save_dir
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        self.save_ckpt = save_ckpt
        self.save_every_epoch = save_every_epoch
        # Running loss
        self.train_step_outputs = []
        # Running time
        self.epoch_start_time = None
    
    def train_dataloader(self):
        return self.train_loader
    
    def training_step(self, batch, _):
        x = batch
        predicted_noise, target = self.model(x)
        loss = F.mse_loss(predicted_noise, target)
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
        self.log("train_loss", avg_loss.item())
        self.train_step_outputs.clear()
        # Save model
        if self.save_ckpt and (self.current_epoch + 1) % self.save_every_epoch == 0:
            save_path = osp.join(self.save_dir, f"ddpm_epoch_{self.current_epoch + 1}.pth")
            torch.save(self.model.state_dict(), save_path)
    
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
                },
                "gradient_clip_val": self.max_grad_norm
            }
        else:
            # Reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
                "gradient_clip_val": self.max_grad_norm
            }
