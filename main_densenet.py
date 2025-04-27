import os
import os.path as osp
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime
from time import time
import warnings
from models.gan import Generator
# from models.vit import ViT
import torchxrayvision as xrv
from dataset.dataloader import get_multilabel_dataloader


warnings.filterwarnings("ignore", category=UserWarning)

def load_gan(path="gan_checkpoints/generator_epoch_5.pt"):
    gan_model = Generator()
    checkpt = torch.load(path)
    gan_model.load_state_dict(checkpt["generator_state_dict"])
    return gan_model

class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_ratio = (self.last_epoch + 1) / self.warmup_epochs
            lr = self.base_lr * warmup_ratio
        else:
            # Linear decay
            decay_ratio = (self.last_epoch - self.warmup_epochs + 1) / (self.total_epochs - self.warmup_epochs + 1)
            lr = self.base_lr - (self.base_lr - self.final_lr) * decay_ratio

        return [lr for _ in self.optimizer.param_groups]



def train_one_epoch(model, loader, criterion, optimizer, max_grad_norm, device, epoch, gan_model):
    model.train()
    running_loss = 0.0
    start = time()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        # if torch.rand(1).item() > 0.4:
        #     with torch.no_grad():
        #         images= gan_model(images)

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    parser = argparse.ArgumentParser(description="Train Densnet on multi-label classification")
    parser.add_argument("--data_dir", type=str, default="data_tensor", help="Root tensor directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--gan_path", type=str, default="gan_checkpoints/generator_epoch_5.pth", help="GAN Checkpoint directory")
    parser.add_argument("--split_type", type=str, default="original", help="dataset_type")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of labels")
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
    train_loader = get_multilabel_dataloader(
        args.data_dir,
        split="train",
        split_type=args.split_type,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = get_multilabel_dataloader(
        args.data_dir,
        split="val",
        split_type=args.split_type,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = get_multilabel_dataloader(
        args.data_dir,
        split="test",
        split_type=args.split_type,
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initialize model, gan model, loss function, and optimizer
    logging.info("Initializing model...")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.index}")
        torch.cuda.set_device(args.index)
    model = xrv.models.DenseNet(weights="densenet121-res224-nih",
                                apply_sigmoid=True).to(device)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, args.num_classes).to(device)
    model.op_threshs = None

    gan_model = load_gan(args.gan_path).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    max_grad_norm = 1.0
    scheduler =  WarmupLinearLR(
        optimizer,
        warmup_epochs=5,
        total_epochs=args.epochs +1 ,
        base_lr = args.lr,
        final_lr= 1e-7
    )
    
    # Main training loop
    logging.info("Starting training...")
    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop_patience = 5
    ckpt_dir = osp.join(args.ckpt_dir, f"densenet-{datetime.now().strftime('%Y-%m-%d')}")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm, device, epoch, gan_model)
        # Gradually improve the gradient clipping threshold
        if epoch > 10:
            max_grad_norm = min(max_grad_norm + 0.1, 5.0)

        val_loss = validate_and_test(model, val_loader, criterion, device, "val")
        scheduler.step(val_loss)
        # Early stopping if no improvement or learning rate is too low
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        current_lr = optimizer.param_groups[0]['lr']
        if no_improve_epochs >= early_stop_patience or current_lr < 1e-7:
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
