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
def validate(model, loader, criterion, device, split):
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


@torch.no_grad()
def test(model, loader, device, label_dict_path):
    """
    Calculate overall accuracy and recall for the disease class,
    using the 'No Finding' label bit to distinguish normal vs. abnormal.
    """
    model.eval()
    # Load and parse label dictionary into a name→index map
    with open(label_dict_path, "r") as f:
        lines = f.read().splitlines()
    label_to_idx = {}
    for line in lines:
        name, idx = line.split(":")
        label_to_idx[name.strip()] = int(idx.strip())
    # Get index of the 'No Finding' (normal) class
    no_find_idx = label_to_idx["No Finding"]
    # Prepare counters
    total_samples = 0
    correct_preds = 0
    true_positive = 0    # correctly predicted diseased
    false_negative = 0   # diseased samples predicted as normal
    sigmoid = nn.Sigmoid()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        # Model outputs logits for each class
        outputs = model(images)
        probs = sigmoid(outputs)
        # Predicted normal if P(No Finding) >= 0.5
        pred_normal = probs[:, no_find_idx] >= 0.5
        # Ground-truth normal where target bit is 1
        actual_normal = targets[:, no_find_idx] == 1
        # Update accuracy count
        correct_preds += (pred_normal == actual_normal).sum().item()
        total_samples += targets.size(0)
        # For disease (positive) samples (actual_normal == False)
        disease_mask = ~actual_normal
        # Predicted disease if not predicted normal
        pred_disease = ~pred_normal
        true_positive += (disease_mask & pred_disease).sum().item()
        false_negative += (disease_mask & pred_normal).sum().item()
    # Compute metrics
    accuracy = correct_preds / total_samples if total_samples > 0 else 0.0
    recall = (true_positive / (true_positive + false_negative)
              if (true_positive + false_negative) > 0 else 0.0)
    # Log results
    logging.info(f"Test Accuracy: {accuracy * 100:.3f}")
    logging.info(f"Disease Recall: {recall * 100:.3f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train ViT on multi-label classification")
    parser.add_argument("--img_size", type=int,
                        default=224, help="Input image size")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/users/nus/e0945822/scratch/checkpoints", help="Checkpoint directory")
    parser.add_argument("--split_type", type=str, default="balanced", choices=[
                        "balanced", "rare_first", "original", "binary"], help="Split type for dataset")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--test_only", action="store_true",
                        help="Test only without training")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Absolute path to the checkpoint for testing")
    parser.add_argument("--model", choices=["alexnet", "vgg16", "resnet18", "resnet34", "vit_pt"],
                        help="The model to run")
    args = parser.parse_args()
    # Set logging configuration
    log_dir = osp.join(
        "logs",
        datetime.now().strftime("%Y-%m-%d"),
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(
        log_dir,
        f"{args.model}" + f"{'-test' if args.test_only else ''}" +
        f"-{args.split_type}-{datetime.now().strftime('%H-%M-%S')}.log"
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
    if args.img_size == 224:
        data_dir = "/home/users/nus/e0945822/scratch/5242data/data_224/"
        if not os.path.exists(data_dir):
            data_dir = "data_tensor"
        train_loader = get_multilabel_dataloader(
            data_dir,
            split_type=args.split_type,
            split="train",
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = get_multilabel_dataloader(
            data_dir,
            split_type=args.split_type,
            split="val",
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = get_multilabel_dataloader(
            data_dir,
            split_type=args.split_type,
            split="test",
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        raise ValueError("Invalid image size. Choose either 224 or 1024.")

    # Initialize model, loss function, and optimizer
    logging.info("Initializing model...")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.index}")
        torch.cuda.set_device(args.index)

    model_args = {
        "in_channels": 1,
        "num_classes": 15
    }

    if args.model == 'alexnet':
        from models.alexnet import AlexNet
        model = AlexNet(**model_args).to(device)
    elif args.model == 'vgg16':
        from models.vgg import VGG
        model = VGG(**model_args, variant='16').to(device)
    elif args.model == 'resnet18' or 'resnet34':
        from models.resnet import ResNet
        variant = args.model.split('resnet')[0]
        model = ResNet(**model_args, variant=variant).to(device)
    elif args.model == 'vit_pt':
        from models.vit_pt import ViT_PT
        model = ViT_PT(**model_args).to(device)

    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
        logging.info(f"Loaded model from {args.ckpt_path}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    max_grad_norm = 1.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Main training loop
    if not args.test_only:
        logging.info("Starting training...")
        best_val_loss = float("inf")
        no_improve_epochs = 0
        early_stop_patience = 10
        ckpt_dir = osp.join(args.ckpt_dir, f"{args.model}" +
                            f"-{args.split_type}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        os.makedirs(ckpt_dir, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, criterion,
                            optimizer, max_grad_norm, device, epoch)
            # Gradually improve the gradient clipping threshold
            if epoch > 10:
                max_grad_norm = min(max_grad_norm + 0.1, 5.0)

            val_loss = validate(model, val_loader, criterion, device, "val")
            scheduler.step(val_loss)
            # Early stopping if no improvement or learning rate is too low
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                torch.save(
                    model.state_dict(),
                    osp.join(ckpt_dir, "best_model.pt")
                )
                logging.info(
                    f"Best model saved at epoch {epoch} with val loss {val_loss:.4f}")
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

        # Save the final model
        torch.save(
            model.state_dict(),
            osp.join(ckpt_dir, "final_model.pt")
        )
        logging.info("Final model saved.")

    # Testing
    logging.info("Testing...")
    # Load the best model for testing
    if not args.test_only:
        ckpt_path = osp.join(ckpt_dir, "best_model.pt")
    else:
        if args.ckpt_path is None:
            raise ValueError("Please provide a checkpoint path for testing.")
        ckpt_path = args.ckpt_path
    model.load_state_dict(torch.load(ckpt_path))
    logging.info(f"Loaded tested model from {ckpt_path}")
    model.to(device)
    test(model, test_loader, device, data_dir + "/label_dict.txt")


if __name__ == "__main__":
    main()
