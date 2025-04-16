import os
import pandas as pd
from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
import random
from collections import Counter
import seaborn as sns
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob
import os

def balanced_multilabel_split(df, train_size=20000, test_size=10000, seed=42):
    random.seed(seed)

    # Convert string labels to list
    if isinstance(df['Finding Labels'].iloc[0], str):
        df['Finding Labels'] = df['Finding Labels'].str.split('|')

    # Count label frequencies
    label_counter = Counter(label for labels in df['Finding Labels'] for label in labels)
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1])  # rarest first

    # Build mapping
    image_labels = dict(zip(df['Image Index'], df['Finding Labels']))
    label_to_images = defaultdict(set)
    for img, labels in image_labels.items():
        for label in labels:
            label_to_images[label].add(img)

    # Sets to track assignment
    train_images = set()
    test_images = set()
    used_images = set()

    def add_image(img, to_train):
        target_set = train_images if to_train else test_images
        if img not in used_images:
            target_set.add(img)
            used_images.add(img)

    for label, _ in sorted_labels:
        candidate_images = list(label_to_images[label] - used_images)
        random.shuffle(candidate_images)

        n_total = len(candidate_images)

        # Skip if we've hit capacity
        if len(train_images) >= train_size and len(test_images) >= test_size:
            break

        remaining_train = train_size - len(train_images)
        remaining_test = test_size - len(test_images)
        available = min(n_total, remaining_train + remaining_test)

        if available == 0:
            continue

        n_train = int(available * 0.8)
        n_test = available - n_train

        # Adjust to stay within global limits
        n_train = min(n_train, remaining_train)
        n_test = min(n_test, remaining_test)

        # Assign images
        for img in candidate_images[:n_train]:
            add_image(img, to_train=True)
        for img in candidate_images[n_train:n_train + n_test]:
            add_image(img, to_train=False)

    return list(train_images), list(test_images)

def binary_balanced_split(df, train_ratio=0.8, seed=42):
    """
    Manually split a balanced dataset into train and test.
    Returns train_df and test_df.
    """
    random.seed(seed)

    # Ensure 'binary_label' column exists
    if 'binary_label' not in df.columns:
        df['binary_label'] = df['Finding Labels'].apply(lambda labels: 0 if labels == ['No Finding'] else 1)

    # Separate into normal and abnormal
    df_normal = df[df['binary_label'] == 0].copy()
    df_abnormal = df[df['binary_label'] == 1].copy()

    # Match dataset sizes by downsampling the larger group
    min_size = min(len(df_normal), len(df_abnormal))
    df_normal = df_normal.sample(min_size, random_state=seed)
    df_abnormal = df_abnormal.sample(min_size, random_state=seed)

    # Shuffle both
    df_normal = df_normal.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_abnormal = df_abnormal.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Calculate split indices
    n_train = int(train_ratio * min_size)

    # Split each group
    train_normal = df_normal.iloc[:n_train]
    test_normal = df_normal.iloc[n_train:]

    train_abnormal = df_abnormal.iloc[:n_train]
    test_abnormal = df_abnormal.iloc[n_train:]

    # Combine splits
    train_df = pd.concat([train_normal, train_abnormal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([test_normal, test_abnormal]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, test_df

def get_label_distribution(subset_df):
    all_labels = [label for labels in subset_df['Finding Labels'] for label in labels]
    return Counter(all_labels)

def get_all_image_paths(root_dir):
    """
    Scans all `images_*/images/*.png` subfolders and returns a dictionary
    mapping image filename to full path. Fits to the structure of the dataset when downloaded
    """
    image_dict = {}
    # Match all folders like images_001, images_002, ..., images_012
    folders = glob(os.path.join(root_dir, 'images_*', 'images'))

    for folder in folders:
        all_images = glob(os.path.join(folder, '*.png'))
        for path in all_images:
            name = os.path.basename(path)
            image_dict[name] = path

    return image_dict

def plot_class_balance(train_df, test_df):
    # Get label distributions
    train_counts = get_label_distribution(train_df)
    test_counts = get_label_distribution(test_df)

    # Combine into a DataFrame for seaborn
    all_labels = sorted(set(train_counts.keys()).union(test_counts.keys()))
    data = {
        'Class': all_labels,
        'Train': [train_counts.get(label, 0) for label in all_labels],
        'Test': [test_counts.get(label, 0) for label in all_labels],
    }
    balance_df = pd.DataFrame(data)

    melt_df = pd.melt(balance_df, id_vars='Class', value_vars=['Train', 'Test'],
                      var_name='Dataset', value_name='Image Count')

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melt_df, x='Class', y='Image Count', hue='Dataset')
    plt.title('Class Balance in Train vs Test Sets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

class NIHDataset_modified(Dataset):
    def __init__(self, df, root_dir, sizeimagesout):
        self.sizeimagesout = sizeimagesout
        self.root_dir = root_dir
        df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|') if isinstance(x, str) else x)
        image_dict = get_all_image_paths(root_dir)

        self.image_names = [name for name in df['Image Index'] if name in image_dict]
        self.image_paths = [image_dict[name] for name in self.image_names]
        self.labels = df.set_index('Image Index').loc[self.image_names]['Finding Labels'].tolist()
        self.ages = df.set_index('Image Index').loc[self.image_names]['Patient Age'].tolist()
        self.transform = T.Compose([
                                      #transforms.RandomRotation(fnd(), expand=True),
                                      T.RandomRotation(10, expand=True),
                                      T.Resize(self.sizeimagesout),
                                      T.RandomHorizontalFlip(),
                                      T.RandomVerticalFlip(),
                                      #transforms.ColorJitter(0.01),
                                      #transforms.RandomAffine(10),
                                      T.ToTensor()
                                     ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, self.labels[index], self.ages[index]
    
class NIHBinaryDataset(Dataset):
    def __init__(self, df, root_dir, sizeimagesout):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.sizeimagesout = sizeimagesout

        # Make sure label is binary
        if 'binary_label' not in self.df.columns:
            self.df['binary_label'] = self.df['Finding Labels'].apply(lambda labels: 0 if labels == ['No Finding'] else 1)

        # Map images to full paths
        self.image_paths = get_all_image_paths(root_dir)  # This should return {filename: full_path}
        self.df = self.df[self.df['Image Index'].isin(self.image_paths.keys())]

        # Filter image paths accordingly
        self.df['full_path'] = self.df['Image Index'].map(self.image_paths)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row['full_path']).convert("RGB")
        image = T.Resize(self.sizeimagesout)(image)
        label = row['binary_label']
        age = row['Patient Age']

        return image, label, age

if __name__ == "__main__":
    df = pd.read_csv('Data_Entry_2017.csv')
    df = df[df['Patient Age'] <= 100].reset_index(drop=True)
    df['Finding Labels'] = df['Finding Labels'].str.split('|')

    root_dir = 'C:/Users/e1498134/Documents/Courses/CS5242/Project'
    size_out = (256, 256)


    ## Dataset constitution:
    # training set = 20000 
    # test set ~ 8000 
    # includes the age in the data (img, label[], age)
    train_ids, test_ids = balanced_multilabel_split(df, train_size=20000, test_size=10000)

    train_df = df[df['Image Index'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['Image Index'].isin(test_ids)].reset_index(drop=True)

    train_dataset = NIHDataset_modified(train_df, root_dir, size_out)
    test_dataset = NIHDataset_modified(test_df, root_dir, size_out)

    '''print("Train label distribution:")
    print(get_label_distribution(train_df))
    print("\nTest label distribution:")
    print(get_label_distribution(test_df))
    plot_class_balance(train_df, test_df)
    img, label = train_dataset[0]
    print(f"Image size: {img.size}, Number of labels: {len(label)}")
    imgplot = plt.imshow(np.squeeze(img))
    plt.show()'''

    ## Dataset constitution for Transformer: Label ∈ (Normal, Abnormal)
    # training set = 82800 
    # test set = 20702  
    # includes the age in the data (img, label, age)

    train_df_binary, test_df_binary = binary_balanced_split(df)
    train_dataset = NIHBinaryDataset(train_df_binary, root_dir,size_out)
    test_dataset = NIHBinaryDataset(test_df_binary, root_dir, size_out)

    #print(f"Train set: {len(train_df_binary)} (Normal: {sum(train_df_binary['binary_label']==0)}, Abnormal: {sum(train_df_binary['binary_label']==1)})")
    #print(f"Test set: {len(test_df_binary)} (Normal: {sum(test_df_binary['binary_label']==0)}, Abnormal: {sum(test_df_binary['binary_label']==1)})")
    
