import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import InterpolationMode

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.utils as vutils
from pathlib import Path

import torchxrayvision as xrv
import torch.nn.functional  as F
from torch.utils.data import random_split
import copy
from torchmetrics.image.fid import FrechetInceptionDistance

LATENT_DIM = 100
IMG_HEIGHT = 224
CHANNELS = 1
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
LAMBDA_L1 = 1



def unnormalize(img):
    return (img + 1) / 2



def evaluate_denoising(generator, test_loader, epoch=None, fid_metric=None):
    generator.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    output_dir = Path(f"eval_outputs/epoch_{epoch}") if epoch is not None else Path("eval_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_saved = False

    if fid_metric is not None:
        fid_metric.reset()

    with torch.no_grad():
        for clean, noisy in test_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            denoised = generator(noisy).clamp(-1, 1)

            if not image_saved:
                clean_img = unnormalize(clean[0].cpu())
                noisy_img = unnormalize(noisy[0].cpu())
                denoised_img = unnormalize(denoised[0].cpu())
                vutils.save_image(noisy_img, output_dir / "noisy_0.png")
                vutils.save_image(clean_img, output_dir / "clean_0.png")
                vutils.save_image(denoised_img, output_dir / "denoised_0.png")
                image_saved = True

            # For FID: convert grayscale to 3-channel RGB
            clean_fid = (unnormalize(clean).clamp(0, 1) * 255).to(torch.uint8)
            denoised_fid = (unnormalize(denoised).clamp(0, 1) * 255).to(torch.uint8)

            if clean_fid.shape[1] == 1:
                clean_fid = clean_fid.repeat(1, 3, 1, 1)
                denoised_fid = denoised_fid.repeat(1, 3, 1, 1)

            if fid_metric is not None:
                fid_metric.update(clean_fid, real=True)
                fid_metric.update(denoised_fid, real=False)

            # PSNR/SSIM
            denoised_np = denoised_fid.cpu().permute(0, 2, 3, 1).numpy()
            clean_np = clean_fid.cpu().permute(0, 2, 3, 1).numpy()

            for i in range(denoised_np.shape[0]):
                psnr_total += psnr(clean_np[i], denoised_np[i], data_range=1.0)
                ssim_total += ssim(clean_np[i], denoised_np[i], data_range=1.0, channel_axis=2)
                count += 1

    generator.train()
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    fid_score = fid_metric.compute().item() if fid_metric is not None else None

    return avg_psnr, avg_ssim, fid_score


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS * 2, 64, kernel_size=3, stride=2),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def center_crop_to_match(self, src, ref):
        _, _, h, w = ref.size()
        return TF.center_crop(src, [h, w])

    def forward(self, noisy, clean_or_fake):
        if noisy.size()[2:] != clean_or_fake.size()[2:]:
            clean_or_fake = self.center_crop_to_match(clean_or_fake, noisy)

        x = torch.cat([noisy, clean_or_fake], dim=1)
        return self.net(x)



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

noise_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_HEIGHT)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1),
    transforms.Normalize([0.5], [0.5])
])



class NIH_DenoiseDataset(Dataset):
    def __init__(self, base_dataset, is_train=True, transform=None, noise_transform=None):
        """
        Args:
            base_dataset: Instance of xrv.datasets.NIH_Dataset
            is_train: True = train/val mode (filter by train_val_list.txt)
                      False = test mode (filter by test_list.txt)
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.noise_transform = noise_transform

        self.filter_by_split(is_train)

    def filter_by_split(self, is_train):
        list_file = "archive/train_val_list.txt" if is_train else "archive/test_list.txt"
        with open(list_file, "r") as f:
            filenames = set(line.strip() for line in f)

        # Filter by list
        filtered_csv = self.base_dataset.csv[
            self.base_dataset.csv["Image Index"].isin(filenames)
        ].copy()

        # Keep only PA view
        if is_train:
            filtered_csv = filtered_csv[filtered_csv["View Position"] == "PA"].reset_index(drop=True)

        self.base_dataset.csv = filtered_csv

    def __len__(self):
        return len(self.base_dataset.csv)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image, label = sample["img"],  sample["lab"]
        image=image[0]
        clean = self.transform(image)
        noisy = self.noise_transform(image)
        return clean, noisy


if __name__ == "__main__":
    base_dataset = xrv.datasets.NIH_Dataset(
        imgpath="archive/images",
        csvpath="archive/Data_Entry_2017.csv",
        bbox_list_path="archive/BBox_List_2017.csv",
        views=["PA", "AP"],
        unique_patients= False,
        transform=None 
    )

    # Datasets
    train_dataset = NIH_DenoiseDataset(copy.deepcopy(base_dataset), is_train=True, transform=transform, noise_transform=noise_transform)
    # test_dataset = NIH_DenoiseDataset(copy.deepcopy(base_dataset), is_train=False, transform=transform, noise_transform=noise_transform)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    writer = SummaryWriter(log_dir="runs/gan")


    warmup_epochs = 5

    def get_warmup_scheduler(optimizer, warmup_epochs, base_lr=1.0, decay_step=15, decay_gamma=0.5):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  
            else:
                return base_lr * (decay_gamma ** ((epoch - warmup_epochs) // decay_step))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler_g = get_warmup_scheduler(optimizer_g, warmup_epochs, base_lr=1e-4)
    scheduler_d = get_warmup_scheduler(optimizer_d, warmup_epochs, base_lr=1e-4)


    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        for i, (clean, noisy) in enumerate(pbar):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            fake_clean = generator(noisy).detach()
            d_real = discriminator(noisy, clean)
            d_fake = discriminator(noisy, fake_clean)

            real = torch.ones_like(d_real) 
            fake = torch.zeros_like(d_fake)
            d_loss = criterion(d_real, real) + criterion(d_fake, fake)

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            fake_clean = generator(noisy)
            adv = discriminator(noisy, fake_clean)
            adv_loss = criterion(adv, real) 
            l1 = F.l1_loss(fake_clean, clean)
            g_loss = adv_loss + LAMBDA_L1 * l1

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()


            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            pbar.set_postfix({"D Loss": f"{d_loss.item():.4f}", "G Loss": f"{g_loss.item():.4f}"})
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        

        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 5 == 0:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict()
            }, os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))

            fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
            avg_psnr, avg_ssim,fid_score  = evaluate_denoising(generator, val_loader, epoch=epoch+1,fid_metric=fid_metric)
            writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
            writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
            writer.add_scalar("Eval/FID", fid_score, epoch)

            # print(f"Eval at Epoch {epoch+1}: PSNR = {avg_psnr:.2f}, SSIM = {avg_ssim:.4f}, FID = {fid_score:.2f}")

            print(f"Saved generator checkpoint at epoch {epoch+1}")
    writer.close()
