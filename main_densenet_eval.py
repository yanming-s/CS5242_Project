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



@torch.no_grad()
def validate_and_test(model, loader, criterion, device, split):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()
        correct = (preds == targets).float().sum()
        
        total_correct += correct.item()
        total_samples += targets.numel()

        loss = criterion(outputs, targets)
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    accuracy = total_correct / total_samples * 100.0
    # logging.info(f"{split} loss {avg_loss:.4f}")
    print(f"Final Evaluation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"accuracy: {accuracy:.4f}")
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
    print(f"Test Accuracy: {accuracy * 100:.3f}")
    print(f"Disease Recall: {recall * 100:.3f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Eval Densnet on multi-label classification")
    parser.add_argument("--data_dir", type=str, default="data_tensor", help="Root tensor directory")
    parser.add_argument("--split_type", type=str, default="original", help="dataset_type")
    parser.add_argument("--model_path", type=str, default="checkpoints/densenet-2025-04-22/final_model.pt", help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of labels")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Get data loaders
    logging.info("Loading data...")
    test_loader = get_multilabel_dataloader(
        args.data_dir,
        split_type=args.split_type,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        
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
    checkpt = torch.load(args.model_path)
    model.load_state_dict(checkpt)

    # criterion = nn.BCEWithLogitsLoss()
    
    # Testing
    logging.info("Testing...")
    test_loss = validate_and_test(model, test_loader, criterion, device, "test")
    logging.info(f"Final Test Loss: {test_loss:.4f}")
    data_dir = args.data_dir 
    # test(model, test_loader, device, data_dir + "/label_dict.txt")

if __name__ == "__main__":
    main()
