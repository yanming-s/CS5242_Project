import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader


class MultiLabelDataset(Dataset):
    """
    A dataset class for multi-label classification.
    Args:
        data_dir (str): Directory containing the dataset.
    """
    def __init__(self, data_dir):
        images_path = osp.join(data_dir, "images.pt")
        labels_path = osp.join(data_dir, "labels.pt")
        if not osp.exists(images_path) or not osp.exists(labels_path):
            raise FileNotFoundError(f"Missing images.pt or labels.pt in {data_dir}")
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        assert len(self.images) == len(self.labels), \
            "Number of images and labels must match"
    def __len__(self):
        return self.images.size(0)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class BinaryDataset(Dataset):
    """
    A dataset class for binary classification.
    Args:
        data_dir (str): Directory containing the dataset.
    """
    def __init__(self, data_dir):
        images_path = osp.join(data_dir, "images.pt")
        if not osp.exists(images_path):
            raise FileNotFoundError(f"Missing images.pt in {data_dir}")
        self.images = torch.load(images_path)
    def __len__(self):
        return self.images.size(0)
    def __getitem__(self, idx):
        return self.images[idx]


def get_multilabel_dataloader(
        root_dir,
        split_type="balanced",
        split="train",
        batch_size=64,
        shuffle=True,
        num_workers=4
):
    """
    Get a DataLoader for the multi-label dataset.
    Args:
        root_dir (str): Root directory of the dataset.
        split_type (str): Type of split for the dataset.
        split (str): Split name (train/val/test).
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        DataLoader: DataLoader for the specified dataset split.
    """
    data_dir = osp.join(root_dir, split_type, split)
    ds = MultiLabelDataset(data_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_binary_dataloader(
        root_dir,
        split_type="binary",
        split="train",
        batch_size=64,
        shuffle=True,
        num_workers=4
):
    """
    Get a DataLoader for the binary dataset.
    Args:
        root_dir (str): Root directory of the dataset.
        split_type (str): Type of split for the dataset.
        split (str): Split name (train/val/test).
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        DataLoader: DataLoader for the specified dataset split.
    """
    data_dir = osp.join(root_dir, split_type, split)
    ds = BinaryDataset(data_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
