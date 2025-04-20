import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block used in UNet
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Time embedding for the UNet model
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in DDPM
    """
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        # Downsampling path
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        self.down4 = Down(base_channels*8, base_channels*8)
        # Time embeddings for different levels
        self.time_mlp1 = nn.Linear(time_emb_dim, base_channels*2)
        self.time_mlp2 = nn.Linear(time_emb_dim, base_channels*4)
        self.time_mlp3 = nn.Linear(time_emb_dim, base_channels*8)
        self.time_mlp4 = nn.Linear(time_emb_dim, base_channels*8)
        # Upsampling path with skip connections
        self.up1 = Up(base_channels*16, base_channels*4)
        self.up2 = Up(base_channels*8, base_channels*2)
        self.up3 = Up(base_channels*4, base_channels)
        self.up4 = Up(base_channels*2, base_channels)
        # Final convolution
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = x2 + self.time_mlp1(t_emb)[:, :, None, None]
        x3 = self.down2(x2)
        x3 = x3 + self.time_mlp2(t_emb)[:, :, None, None]
        x4 = self.down3(x3)
        x4 = x4 + self.time_mlp3(t_emb)[:, :, None, None]
        x5 = self.down4(x4)
        x5 = x5 + self.time_mlp4(t_emb)[:, :, None, None]
        # Upsampling with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final convolution
        logits = self.outc(x)
        return logits


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(
        self,
        model,
        beta_start=1e-4,
        beta_end=0.02,
        num_timesteps=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process: q(x_t | x_0)
        Takes an image and a timestep as input and returns a noisy version of the image
        """
        noise = torch.randn_like(x_0)
        mean = self.sqrt_alphas_cumprod[t][:, None, None, None] * x_0
        variance = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return mean + variance * noise, noise
    
    def sample_timesteps(self, n):
        """
        Sample timesteps uniformly for training
        """
        return torch.randint(low=1, high=self.num_timesteps, size=(n,), device=self.device)

    def p_losses(self, x_0, t, noise=None):
        """
        Training loss calculation
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        # Add noise to the input image according to the timestep
        x_noisy, target = self.forward_diffusion(x_0, t)
        # Predict the noise with the model
        predicted_noise = self.model(x_noisy, t)
        # Calculate the loss
        loss = F.mse_loss(predicted_noise, target)
        return loss
    
    def train_step(self, x, optimizer):
        """
        Perform a single training step
        """
        self.model.train()
        optimizer.zero_grad()
        # Sample random timesteps
        batch_size = x.shape[0]
        t = self.sample_timesteps(batch_size)
        # Calculate loss
        loss = self.p_losses(x, t)
        # Backpropagation
        loss.backward()
        optimizer.step()
        return loss.item()


if __name__ == "__main__":
    # Example usage
    unet = UNet(in_channels=1, out_channels=1)
    ddpm = DDPM(unet)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
    # Dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_0 = torch.randn(32, 1, 224, 224).to(device)
    loss = ddpm.train_step(x_0, optimizer)
    print(f"Training loss: {loss}")
