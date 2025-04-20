import os
import os.path as osp
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from time import time
import warnings

from models.ddpm import UNet, DDPM
from dataset.dataloader import get_binary_dataloader

warnings.filterwarnings("ignore", category=UserWarning)


def train_one_epoch(ddpm_model, loader, optimizer, max_grad_norm, device, epoch):
    ddpm_model.model.train()
    running_loss = 0.0
    start = time()
    for images in loader:
        images = images.to(device)
        optimizer.zero_grad()
        batch_size = images.shape[0]
        t = ddpm_model.sample_timesteps(batch_size)
        loss = ddpm_model.p_losses(images, t)
        loss.backward()
        nn.utils.clip_grad_norm_(ddpm_model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item()
    duration = time() - start
    avg_loss = running_loss / len(loader)
    logging.info(f"epoch {epoch} - loss {avg_loss:.4f} - time {duration:.2f}s")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, split):
    model.eval()
    running_loss = 0.0
    for images in loader:
        images = images.to(device)
        batch_size = images.shape[0]
        t = model.sample_timesteps(batch_size)
        loss = model.p_losses(images, t)
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    logging.info(f"{split} loss {avg_loss:.4f}")
    return avg_loss


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ViT on multi-label classification")
    parser.add_argument("--data_dir", type=str, default="data_tensor", help="Root tensor directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--check_every", type=int, default=5, help="Check model every N epochs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels for UNet")
    parser.add_argument("--time_emb_dim", type=int, default=256, help="Time embedding dimension")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    # Set logging configuration
    log_dir = osp.join(
        "logs",
        datetime.now().strftime("%Y-%m-%d"),
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(
        log_dir,
        f"ddpm-{datetime.now().strftime('%H-%M-%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Get data loaders
    logging.info("Loading data...")
    torch.manual_seed(args.seed)
    train_loader = get_binary_dataloader(
        args.data_dir,
        split="train",
        batch_size=args.batch_size,
        shuffle=True
    )
    test_normal = get_binary_dataloader(
        args.data_dir,
        split="test_normal",
        batch_size=args.batch_size,
        shuffle=False
    )
    test_abnormal = get_binary_dataloader(
        args.data_dir,
        split="test_abnormal",
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initialize model, loss function, and optimizer
    logging.info("Initializing model...")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.index}")
        torch.cuda.set_device(args.index)
    unet = UNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim
    )
    ddpm = DDPM(
        model=unet,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_timesteps=args.num_timesteps,
        device=device
    )
    optimizer = optim.Adam(ddpm.parameters(), lr=args.lr)
    max_grad_norm = 1.0
    
    # Main training loop
    logging.info("Starting training...")
    best_eval_loss = float("inf")
    no_improve_epochs = 0
    early_stop_patience = 2
    ckpt_dir = osp.join(args.ckpt_dir, f"ddpm-{datetime.now().strftime('%Y-%m-%d')}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(ddpm, train_loader, optimizer, max_grad_norm, device, epoch)
        # Gradually improve the gradient clipping threshold
        if epoch > 10:
            max_grad_norm = min(max_grad_norm + 0.1, 5.0)

        # Periodic checkpoint saving
        if epoch % args.save_every == 0:
            torch.save(
                ddpm.state_dict(),
                osp.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
            )
            logging.info(f"Checkpoint saved at epoch {epoch}")
        
        # Periodic evaluation
        if epoch % args.check_every == 0:
            eval_loss = evaluate(ddpm, test_normal, device, "eval normal")
            evaluate(ddpm, test_abnormal, device, "eval abnormal")
            # Early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stop_patience:
                    logging.info("Early stopping triggered.")
                    break

    # Save the final model
    torch.save(
        ddpm.state_dict(),
        osp.join(ckpt_dir, "final_model.pt")
    )
    logging.info("Final model saved.")


if __name__ == "__main__":
    main()
