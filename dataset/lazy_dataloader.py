import os
import os.path as osp
import re
import glob
import torch
from torch.utils.data import Dataset, DataLoader


class LazyChunkDataset(Dataset):
    def __init__(self, save_root, split, chunk_size=32, max_chunks_in_ram=25):
        super().__init__()
        self.save_dir = os.path.join(save_root, split)
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks_in_ram
        # Get sorted list of chunk files
        self.image_chunks = sorted(
            glob.glob(os.path.join(self.save_dir, "images_chunk_*.pt")),
            key=lambda x: int(re.search(r'images_chunk_(\d+).pt', x).group(1))
        )
        self.label_chunks = sorted(
            glob.glob(os.path.join(self.save_dir, "labels_chunk_*.pt")),
            key=lambda x: int(re.search(r'labels_chunk_(\d+).pt', x).group(1))
        )
        # Validate chunks
        assert len(self.image_chunks) == len(self.label_chunks), "Mismatched image/label chunks"
        self.num_chunks = len(self.image_chunks)
        # Calculate total samples
        if self.num_chunks == 0:
            self.total_samples = 0
        else:
            # Load last chunk to get final count
            last_img_chunk = torch.load(self.image_chunks[-1])
            self.total_samples = (self.num_chunks - 1) * chunk_size + len(last_img_chunk)
        # LRU buffer management
        self.buffer = {}  # {chunk_idx: (images, labels)}
        self.access_order = []

    def __len__(self):
        return self.total_samples

    def _get_chunk_position(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError()
        chunk_idx = idx // self.chunk_size
        if chunk_idx < self.num_chunks - 1:
            pos = idx % self.chunk_size
        else:
            pos = idx - (self.num_chunks - 1) * self.chunk_size
        return chunk_idx, pos

    def _load_chunk(self, chunk_idx):
        """Load chunk into buffer with LRU management"""
        if chunk_idx not in self.buffer:
            # Load from disk
            img_chunk = torch.load(self.image_chunks[chunk_idx])
            lbl_chunk = torch.load(self.label_chunks[chunk_idx])
            # Add to buffer
            self.buffer[chunk_idx] = (img_chunk, lbl_chunk)
            self.access_order.append(chunk_idx)
            # Enforce max buffer size
            while len(self.buffer) > self.max_chunks:
                oldest = self.access_order.pop(0)
                if oldest in self.buffer:
                    del self.buffer[oldest]
        else:
            # Update access order
            self.access_order.remove(chunk_idx)
            self.access_order.append(chunk_idx)

    def __getitem__(self, idx):
        chunk_idx, pos = self._get_chunk_position(idx)
        if chunk_idx not in self.buffer:
            self._load_chunk(chunk_idx)
        images, labels = self.buffer[chunk_idx]
        return images[pos], labels[pos]


def get_lazy_dataloader(
        save_root,
        split_type="balanced",
        split="train",
        chunk_size=32,
        max_chunks_in_ram=25,
        batch_size=32,
        shuffle=True,
        num_workers=4
):
    """
    Get a DataLoader for the lazy chunked dataset.
    Args:
        save_root (str): Root directory where dataset chunks are stored.
        split_type (str): Split strategy (e.g., "balanced", "binary").
        split (str): Split name (train/val/test).
        chunk_size (int): Size of each data chunk.
        max_chunks_in_ram (int): Maximum number of chunks to keep in memory.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        DataLoader: DataLoader for the specified dataset split.
    """
    data_dir = osp.join(save_root, split_type)
    dataset = LazyChunkDataset(
        save_root=data_dir,
        split=split,
        chunk_size=chunk_size,
        max_chunks_in_ram=max_chunks_in_ram,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
