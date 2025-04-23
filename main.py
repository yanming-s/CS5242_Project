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

from models.vit import ViT
from models.vit_conv import ViTWithConvStem
from dataset.dataloader import get_multilabel_dataloader
from dataset.lazy_dataloader import get_lazy_dataloader

warnings.filterwarnings("ignore", category=UserWarning)


def train_one_epoch(model, loader, criterion, optimizer, max_grad_norm, device, epoch):
    model.train()
    running_loss = 0.0
    start = time()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item()
    duration = time() - start
    avg_loss = running_loss / len(loader)
    logging.info(f"epoch {epoch} - loss {avg_loss:.4f} - time {duration:.2f}s")
    return avg_loss


@torch.no_grad()
def validate_and_test(model, loader, criterion, device, split):
    model.eval()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    logging.info(f"{split} loss {avg_loss:.4f}")
    return avg_loss


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ViT on multi-label classification")
    parser.add_argument("--data_dir", type=str, default="data_tensor", help="Root tensor directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=768*4, help="MLP hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of labels")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lazy", type=bool, default=True, help="Use lazy loading dataset")
    parser.add_argument("--use_pretrained", type=bool, default=True, help="Use pretrained model")
    args = parser.parse_args()
    # Set logging configuration
    log_dir = osp.join(
        "logs",
        datetime.now().strftime("%Y-%m-%d"),
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(
        log_dir,
        f"vit-{datetime.now().strftime('%H-%M-%S')}.log"
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
    if args.lazy:
        train_loader = get_lazy_dataloader(
            args.data_dir,
            split="train",
            chunk_size=args.batch_size,
            max_chunks_in_ram=25,
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = get_lazy_dataloader(
            args.data_dir,
            split="val",
            chunk_size=args.batch_size,
            max_chunks_in_ram=25,
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = get_lazy_dataloader(
            args.data_dir,
            split="test",
            chunk_size=args.batch_size,
            max_chunks_in_ram=25,
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        train_loader = get_multilabel_dataloader(
            args.data_dir,
            split="train",
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = get_multilabel_dataloader(
            args.data_dir,
            split="val",
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = get_multilabel_dataloader(
            args.data_dir,
            split="test",
            batch_size=args.batch_size,
            shuffle=False
        )
    # Initialize model, loss function, and optimizer
    logging.info("Initializing model...")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.index}")
        torch.cuda.set_device(args.index)
    if args.use_pretrained:
        model_args = {
            "img_size": args.img_size,
            "in_channels": 1,
            "num_classes": args.num_classes,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_dim // args.embed_dim,
            "dropout": args.dropout,
            "use_pretrained_blocks": True
        }
        model = ViTWithConvStem(**model_args).to(device)
    else:
        model = ViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=1,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout
        ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    max_grad_norm = 1.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Main training loop
    logging.info("Starting training...")
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop_patience = 10
    ckpt_dir = osp.join(args.ckpt_dir, f"vit-{datetime.now().strftime('%Y-%m-%d')}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm, device, epoch)
        # Gradually improve the gradient clipping threshold
        if epoch > 10:
            max_grad_norm = min(max_grad_norm + 0.1, 5.0)

        val_loss = validate_and_test(model, val_loader, criterion, device, "val")
        scheduler.step(val_loss)
        # Early stopping if no improvement or learning rate is too low
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(
                model.state_dict(),
                osp.join(ckpt_dir, "best_model.pt")
            )
            logging.info(f"Best model saved at epoch {epoch} with val loss {val_loss:.4f}")
        else:
            no_improve_epochs += 1
        current_lr = optimizer.param_groups[0]['lr']
        if no_improve_epochs >= early_stop_patience or current_lr < 1e-6:
            logging.info(f"Early stopping triggered at epoch {epoch}.")
            break

        # Periodic checkpoint saving
        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                osp.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
            )
            logging.info(f"Checkpoint saved at epoch {epoch}")
    
    # Testing
    logging.info("Testing...")
    test_loss = validate_and_test(model, test_loader, criterion, device, "test")
    logging.info(f"Final Test Loss: {test_loss:.4f}")
    # Save the final model
    torch.save(
        model.state_dict(),
        osp.join(ckpt_dir, "final_model.pt")
    )
    logging.info("Final model saved.")


if __name__ == "__main__":
    main()
