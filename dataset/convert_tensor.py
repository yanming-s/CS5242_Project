import os
import os.path as osp
from glob import glob
import argparse
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


CHUNK_SIZE = 32


def collect_image_paths(root_dir):
    """
    Collect all image paths from the dataset.
    Args:
        root_dir (str): The root directory of the dataset.
    Returns:
        dict: A dictionary mapping image names to their paths.
    """
    image_dict = {}
    folders = glob(os.path.join(root_dir, 'images_*', 'images'))
    for folder in folders:
        all_images = glob(os.path.join(folder, '*.png'))
        for path in all_images:
            name = os.path.basename(path)
            image_dict[name] = path
    return image_dict


def convert_img_to_tensor(split_df, image_dict, save_root, split_type):
    """
    Convert images to tensor format and save them.
    Args:
        split_df (pd.DataFrame): DataFrame containing image names and labels.
        image_dict (dict): Dictionary mapping image names to their paths.
        save_root (str): Directory to save the converted tensors.
        split_type (str): Type of split for the dataset.
    """
    output_size = 224
    save_dir = osp.join(save_root, split_type)
    os.makedirs(save_dir, exist_ok=True)
    indices = split_df["Image Index"].values.tolist()
    if len(indices) == 0:
        torch.save(torch.empty((0, 1, output_size, output_size)), osp.join(save_dir, "images.pt"))
    tensor_list = []
    transform = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    for i in tqdm(range(len(indices)), desc=f"Converting {split_type} images"):
        image = indices[i]
        path = image_dict[image]
        img = Image.open(path).convert("L")
        tensor = transform(img)
        tensor_list.append(tensor)
    tensors = torch.stack(tensor_list)
    torch.save(tensors, osp.join(save_dir, "images.pt"))


def convert_img_to_tensor_chunk(split_df, image_dict, save_root, split_type, chunk_size=32):
    """
    Convert images to tensor format and save them in chunks.
    Args:
        split_df (pd.DataFrame): DataFrame containing image names and labels.
        image_dict (dict): Dictionary mapping image names to their paths.
        save_root (str): Directory to save the converted tensors.
        split_type (str): Type of split for the dataset.
        chunk_size (int): Number of images per chunk.
    """
    output_size = 1024
    save_dir = osp.join(save_root, split_type)
    os.makedirs(save_dir, exist_ok=True)
    indices = split_df["Image Index"].values.tolist()
    num_images = len(indices)
    if num_images == 0:
        torch.save(torch.empty((0, 1, output_size, output_size)), osp.join(save_dir, "images_chunk_0.pt"))
        return
    transform = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    chunk = []
    chunk_idx = 0
    for i in tqdm(range(num_images), desc=f"Converting {split_type} images in chunks"):
        image = indices[i]
        path = image_dict[image]
        img = Image.open(path).convert("L")
        tensor = transform(img)
        chunk.append(tensor)
        if len(chunk) == chunk_size or i == num_images - 1:
            tensors = torch.stack(chunk)
            torch.save(tensors, osp.join(save_dir, f"images_chunk_{chunk_idx}.pt"))
            chunk = []
            chunk_idx += 1


def convert_label_to_tensor(split_df, label_dict, save_root, split_type):
    """
    Convert labels to tensor format and save them.
    Args:
        split_df (pd.DataFrame): DataFrame containing labels.
        label_dict (dict): Dictionary mapping labels to indices.
        save_root (str): Directory to save the converted tensors.
        split_type (str): Type of split for the dataset.
    """
    save_dir = osp.join(save_root, split_type)
    os.makedirs(save_dir, exist_ok=True)
    lbl_series = split_df["Finding Labels"].tolist()
    num_samples = len(lbl_series)
    num_classes = len(label_dict)
    if num_samples == 0:
        # Deal with empty dataset
        empty = torch.empty((0, num_classes), dtype=torch.float32)
        torch.save(empty, osp.join(save_dir, "labels.pt"))
        return
    tensor_list = []
    for lbls in tqdm(lbl_series, desc=f"Converting {split_type} labels"):
        # Multi-hot encoding for multi-label classification
        mh = torch.zeros(num_classes, dtype=torch.float32)
        for l in lbls:
            if l and l in label_dict:
                mh[label_dict[l]] = 1.0
        tensor_list.append(mh)
    labels_tensor = torch.stack(tensor_list)
    torch.save(labels_tensor, osp.join(save_dir, "labels.pt"))


def convert_label_to_tensor_chunk(split_df, label_dict, save_root, split_type, chunk_size=32):
    """
    Convert labels to tensor format and save them in chunks.
    Args:
        split_df (pd.DataFrame): DataFrame containing labels.
        label_dict (dict): Dictionary mapping labels to indices.
        save_root (str): Directory to save the converted tensors.
        split_type (str): Type of split for the dataset.
        chunk_size (int): Number of labels per chunk.
    """
    save_dir = osp.join(save_root, split_type)
    os.makedirs(save_dir, exist_ok=True)
    lbl_series = split_df["Finding Labels"].tolist()
    num_samples = len(lbl_series)
    num_classes = len(label_dict)
    if num_samples == 0:
        empty = torch.empty((0, num_classes), dtype=torch.float32)
        torch.save(empty, osp.join(save_dir, "labels_chunk_0.pt"))
        return
    chunk = []
    chunk_idx = 0
    for i, lbls in enumerate(tqdm(lbl_series, desc=f"Converting {split_type} labels in chunks")):
        mh = torch.zeros(num_classes, dtype=torch.float32)
        for l in lbls:
            if l and l in label_dict:
                mh[label_dict[l]] = 1.0
        chunk.append(mh)
        if len(chunk) == chunk_size or i == num_samples - 1:
            tensors = torch.stack(chunk)
            torch.save(tensors, osp.join(save_dir, f"labels_chunk_{chunk_idx}.pt"))
            chunk = []
            chunk_idx += 1


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert images to tensor format.")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory of the dataset.")
    parser.add_argument("--output_dir", type=str, default="data_tensor", help="Output directory for tensors.")
    parser.add_argument(
        "--split_type",
        type=str,
        default="balanced",
        choices=[
            "balanced",
            "rare_first",
            "original",
            "binary"
        ],
        help="Type of split to use."
    )
    parser.add_argument(
        "--use_chunk",
        action="store_true",
        help="Whether to use chunked tensor conversion (default: True)."
    )
    args = parser.parse_args()
    root_dir = args.root_dir
    split_type = args.split_type
    use_chunk = args.use_chunk
    if not use_chunk:
        output_dir = "data_224"
        if osp.exists("data_tensor"):
            output_dir = "data_tensor"
    else:
        output_dir = "data_1024"
    save_dir = os.path.join(output_dir, split_type)
    os.makedirs(save_dir, exist_ok=True)
    image_dict = collect_image_paths(root_dir)
    # Load the CSV file and split the dataset
    df = pd.read_csv(os.path.join(root_dir, "Data_Entry_2017.csv"))
    if split_type in ["balanced", "original", "rare_first"]:
        if split_type == "balanced":
            from sample import multilabel_balanced_split
            train_df, val_df, test_df = multilabel_balanced_split(df)
        elif split_type == "rare_first":
            from sample import multilabel_rare_first_split
            train_df, val_df, test_df = multilabel_rare_first_split(df)
        else:
            from sample import multilabel_split
            train_df, val_df, test_df = multilabel_split(df)
        # Check whether the dataset has already been split
        if osp.exists(osp.join(save_dir, "train", "images.pt")) or osp.exists(osp.join(save_dir, "train", "images_chunk_0.pt")):
            print("Dataset has already been split.")
            return
        # Generate unified label dict from all Finding Labels
        label_dict = None
        all_labels = set()
        for item in df["Finding Labels"].tolist():
            lbls = item.split("|")
            all_labels.update([l for l in lbls if l])
        all_labels = sorted(all_labels)
        label_dict = {label: idx for idx, label in enumerate(all_labels)}
        # Save label dict to a file
        label_dict_path = os.path.join(output_dir, "label_dict.txt")
        with open(label_dict_path, "w") as f:
            for label, idx in label_dict.items():
                f.write(f"{label}: {idx}\n")
        if use_chunk:
            convert_img_to_tensor_chunk(train_df, image_dict, save_dir, "train", chunk_size=CHUNK_SIZE)
            convert_label_to_tensor_chunk(train_df, label_dict, save_dir, "train", chunk_size=CHUNK_SIZE)
            convert_img_to_tensor_chunk(val_df, image_dict, save_dir, "val", chunk_size=CHUNK_SIZE)
            convert_label_to_tensor_chunk(val_df, label_dict, save_dir, "val", chunk_size=CHUNK_SIZE)
            convert_img_to_tensor_chunk(test_df, image_dict, save_dir, "test", chunk_size=CHUNK_SIZE)
            convert_label_to_tensor_chunk(test_df, label_dict, save_dir, "test", chunk_size=CHUNK_SIZE)
        else:
            convert_img_to_tensor(train_df, image_dict, save_dir, "train")
            convert_label_to_tensor(train_df, label_dict, save_dir, "train")
            convert_img_to_tensor(val_df, image_dict, save_dir, "val")
            convert_label_to_tensor(val_df, label_dict, save_dir, "val")
            convert_img_to_tensor(test_df, image_dict, save_dir, "test")
            convert_label_to_tensor(test_df, label_dict, save_dir, "test")
    else:
        from sample import binary_split
        train_df, test_normal_df, test_abnormal_df = binary_split(df)
        # Check whether the dataset has already been split
        if osp.exists(osp.join(save_dir, "train", "images.pt")) or osp.exists(osp.join(save_dir, "train", "images_chunk_0.pt")):
            print("Dataset has already been split.")
            return
        if use_chunk:
            convert_img_to_tensor_chunk(train_df, image_dict, save_dir, "train", chunk_size=CHUNK_SIZE)
            convert_img_to_tensor_chunk(test_normal_df, image_dict, save_dir, "test_normal", chunk_size=CHUNK_SIZE)
            convert_img_to_tensor_chunk(test_abnormal_df, image_dict, save_dir, "test_abnormal", chunk_size=CHUNK_SIZE)
        else:
            convert_img_to_tensor(train_df, image_dict, save_dir, "train")
            convert_img_to_tensor(test_normal_df, image_dict, save_dir, "test_normal")
            convert_img_to_tensor(test_abnormal_df, image_dict, save_dir, "test_abnormal")


if __name__ == "__main__":
    main()