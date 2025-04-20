import os
import os.path as osp
from glob import glob
import argparse
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


CHUNK_SIZE = 64
OUTPUT_SIZE = 224
T = transforms.Compose([
    transforms.Resize((OUTPUT_SIZE, OUTPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


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
    save_dir = osp.join(save_root, split_type)
    os.makedirs(save_dir, exist_ok=True)
    indices = split_df["Image Index"].values.tolist()
    if len(indices) == 0:
        torch.save(torch.empty((0, 1, OUTPUT_SIZE, OUTPUT_SIZE)), osp.join(save_dir, "images.pt"))
    tensor_list = []
    for i in tqdm(range(len(indices)), desc=f"Converting {split_type} images"):
        image = indices[i]
        path = image_dict[image]
        img = Image.open(path).convert("L")
        tensor = T(img)
        tensor_list.append(tensor)
    tensors = torch.stack(tensor_list)
    torch.save(tensors, osp.join(save_dir, "images.pt"))


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
    args = parser.parse_args()
    root_dir = args.root_dir
    output_dir = args.output_dir
    split_type = args.split_type
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
        if osp.exists(osp.join(save_dir, "train", "images.pt")):
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
        if osp.exists(osp.join(save_dir, "train", "images.pt")):
            print("Dataset has already been split.")
            return
        convert_img_to_tensor(train_df, image_dict, save_dir, "train")
        convert_img_to_tensor(test_normal_df, image_dict, save_dir, "test_normal")
        convert_img_to_tensor(test_abnormal_df, image_dict, save_dir, "test_abnormal")


if __name__ == "__main__":
    main()
