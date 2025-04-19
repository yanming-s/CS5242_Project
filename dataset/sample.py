import os
import os.path as osp
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def multilabel_split(df, train_size=20000, val_size=2500, test_size=2500, seed=42):
    """
    Splits a multilabel dataset into stratified training, validation, and test sets.

    This function takes a DataFrame containing multilabel annotations and splits it into
    training, validation, and test subsets, ensuring that the distribution of labels is
    preserved across all splits. The splitting is performed in three stages using
    multilabel stratified shuffle splitting to maintain label proportions. The function
    supports specifying the desired size for each split and uses a random seed for
    reproducibility.
    
    Args:
        df (pd.DataFrame): Input DataFrame with at least "Image Index" and "Finding Labels" columns.
        train_size (int, optional): Number of samples in the training set. Default is 20,000.
        val_size (int, optional): Number of samples in the validation set. Default is 2,500.
        test_size (int, optional): Number of samples in the test set. Default is 2,500.
        seed (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
        tuple: A tuple containing three DataFrames: (df_train, df_val, df_test), corresponding
               to the stratified training, validation, and test splits.
    """
    random.seed(seed)
    df = df[["Image Index", "Finding Labels"]].copy()
    if isinstance(df["Finding Labels"].iloc[0], str):
        df["Finding Labels"] = df["Finding Labels"].str.split("|")
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    Y_full = mlb.fit_transform(df["Finding Labels"])   # (N, L)
    idx_full = df.index.to_numpy()
    total = len(df)
    subsz = train_size + val_size + test_size
    # Select a stratified subsample of the desired total size from the full dataset
    msss0 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=subsz / total,
        test_size=(total - subsz) / total,
        random_state=seed
    )
    sub_i, _ = next(msss0.split(idx_full, Y_full))
    df_sub = df.iloc[sub_i].reset_index(drop=True)
    Y_sub  = Y_full[sub_i]
    # Split the subsample into training and temporary (validation + test) sets
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size / subsz,
        test_size=(val_size + test_size) / subsz,
        random_state=seed
    )
    tr_i, tmp_i = next(msss1.split(df_sub.index.to_numpy(), Y_sub))
    df_train = df_sub.iloc[tr_i].reset_index(drop=True)
    df_tmp   = df_sub.iloc[tmp_i].reset_index(drop=True)
    Y_tmp    = Y_sub[tmp_i]
    # Split the temporary set into validation and test sets
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=val_size / (val_size + test_size),
        test_size=test_size / (val_size + test_size),
        random_state=seed
    )
    v_i, te_i = next(msss2.split(df_tmp.index.to_numpy(), Y_tmp))
    df_val  = df_tmp.iloc[v_i].reset_index(drop=True)
    df_test = df_tmp.iloc[te_i].reset_index(drop=True)
    return df_train, df_val, df_test


def multilabel_balanced_split(df, train_size=20000, val_size=2500, test_size=2500, seed=42):
    """
    Splits a multilabel dataset into balanced training, validation, and test sets.

    This function attempts to ensure that each label is represented as evenly as possible across the splits,
    by iteratively sampling indices associated with each label until the desired split sizes are reached.
    If there are not enough samples to perfectly balance all labels, the function fills the remaining slots
    with random samples from the unused pool.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least "Image Index" and "Finding Labels" columns.
                           "Finding Labels" should be a string of labels separated by '|' or a list of labels.
        train_size (int): Number of samples in the training set.
        val_size (int): Number of samples in the validation set.
        test_size (int): Number of samples in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, val_df, test_df)
            train_df (pd.DataFrame): Training set DataFrame.
            val_df (pd.DataFrame): Validation set DataFrame.
            test_df (pd.DataFrame): Test set DataFrame.
    """
    random.seed(seed)
    df = df[["Image Index", "Finding Labels"]].copy()
    if isinstance(df["Finding Labels"].iloc[0], str):
        df["Finding Labels"] = df["Finding Labels"].str.split('|')
    total_size = train_size + val_size + test_size
    # Construct a mapping of labels to their indices
    label2idxs = defaultdict(list)
    for idx, row in df.iterrows():
        for label in row['Finding Labels']:
            label2idxs[label].append(idx)
    labels = list(label2idxs.keys())
    # Records the index of the last sampled item for each label
    label_ptr = {l: 0 for l in labels}
    # Main sampling loop
    sampled_idx = set()
    last_sampled_count = 0
    cut_pos = []
    newly_sampled_idx = []
    total_idxs = set(range(len(df)))
    while len(sampled_idx) < total_size:
        for label in labels:
            ptr = label_ptr[label]
            total_label_idx = label2idxs[label]
            # Skip labels that have been fully traversed
            if ptr >= len(total_label_idx):
                continue
            while ptr < len(total_label_idx):
                xi = total_label_idx[ptr]
                ptr += 1
                if xi not in sampled_idx:
                    sampled_idx.add(xi)
                    newly_sampled_idx.append(xi)
                    break
            label_ptr[label] = ptr
        # Check if have exhausted all labels
        any_exhausted = False
        for l in labels:
            if label_ptr[l] >= len(label2idxs[l]):
                any_exhausted = True
                break
        if any_exhausted or len(sampled_idx) >= total_size:
            # Record the cut point
            cut_pos.append((last_sampled_count, len(sampled_idx)))
            last_sampled_count = len(sampled_idx)
    # Fill in the remaining samples
    remaining_needed = total_size - len(sampled_idx)
    if remaining_needed > 0:
        remain_pool = list(total_idxs - sampled_idx)
        random.shuffle(remain_pool)
        for xi in remain_pool[:remaining_needed]:
            sampled_idx.add(xi)
            newly_sampled_idx.append(xi)
        if remaining_needed > 0:
            cut_pos.append((last_sampled_count, len(sampled_idx)))
            last_sampled_count = len(sampled_idx)
    # Assign the sampled indices to the corresponding dataset
    sampled_idx_list = list(sampled_idx)
    random.shuffle(sampled_idx_list)
    train_ids = sampled_idx_list[:train_size]
    val_ids = sampled_idx_list[train_size: train_size + val_size]
    test_ids = sampled_idx_list[train_size + val_size: train_size + val_size + test_size]
    train_df = df.loc[train_ids].reset_index(drop=True)
    val_df = df.loc[val_ids].reset_index(drop=True)
    test_df = df.loc[test_ids].reset_index(drop=True)
    return train_df, val_df, test_df


def binary_split(df, train_size=20000, test_normal_size=2500, test_abnormal_size=2500, seed=42):
    """
    Splits a DataFrame containing medical findings into training and test sets for binary classification.
    This function separates the input DataFrame into normal and abnormal samples based on the 'Finding Labels' column,
    assigns a binary label (0 for 'No Finding', 1 otherwise), shuffles the data, and creates three splits:
    a training set of normal samples, a test set of normal samples, and a test set of abnormal samples.
    Args:
        df (pd.DataFrame): Input DataFrame with a 'Finding Labels' column containing lists of findings.
        train_size (int, optional): Number of normal samples to include in the training set. Default is 20,000.
        test_normal_size (int, optional): Number of normal samples to include in the test set. Default is 2,500.
        test_abnormal_size (int, optional): Number of abnormal samples to include in the test set. Default is 2,500.
        seed (int, optional): Random seed for reproducibility. Default is 42.
    Returns:
        tuple: A tuple containing three DataFrames:
            - train_df (pd.DataFrame): Training set of normal samples.
            - test_normal_df (pd.DataFrame): Test set of normal samples.
            - test_abnormal_df (pd.DataFrame): Test set of abnormal samples.
    Raises:
        ValueError: If there are not enough normal or abnormal samples to satisfy the requested split sizes.
    """
    random.seed(seed)
    # Ensure 'binary_label' column exists
    if 'binary_label' not in df.columns:
        df['binary_label'] = df['Finding Labels'].apply(lambda labels: 0 if labels == ['No Finding'] else 1)
    # Separate into normal and abnormal
    df_normal = df[df['binary_label'] == 0].copy()
    df_abnormal = df[df['binary_label'] == 1].copy()
    # Shuffle both datasets
    df_normal = df_normal.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_abnormal = df_abnormal.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Check if we have enough data
    if len(df_normal) < train_size:
        raise ValueError(f"Not enough normal samples: have {len(df_normal)}, need {train_size}")
    if len(df_abnormal) < (test_normal_size + test_abnormal_size):
        raise ValueError(f"Not enough abnormal samples: have {len(df_abnormal)}, need {test_normal_size, test_abnormal_size}")
    # Create the splits
    train_df = df_normal.iloc[:train_size].reset_index(drop=True)
    test_normal_df = df_normal.iloc[train_size:train_size+test_normal_size].reset_index(drop=True)
    test_abnormal_df = df_abnormal.iloc[:test_abnormal_size].reset_index(drop=True)
    return train_df, test_normal_df, test_abnormal_df


def show_split_distribution(train_df, val_df, test_df, original_df=None, save_name="label_distribution.png"):
    """
    Visualizes the relative percentage distribution of labels across train, validation and test sets.

    Args:
        train_df: Training set DataFrame with 'Finding Labels' column
        val_df: Validation set DataFrame with 'Finding Labels' column
        test_df: Test set DataFrame with 'Finding Labels' column
        original_df: Optional - Original dataset for comparison
        save_path: File path to save the plot
    """
    # Function to count labels in a dataframe
    def count_labels(df):
        label_counts = defaultdict(int)
        for labels in df['Finding Labels']:
            for label in labels:
                label_counts[label] += 1
        return label_counts
    # Count labels in each split
    train_counts = count_labels(train_df)
    val_counts = count_labels(val_df)
    test_counts = count_labels(test_df)
    # If original dataset is provided, count those labels too
    if original_df is not None:
        # Ensure 'Finding Labels' are lists
        if isinstance(original_df["Finding Labels"].iloc[0], str):
            orig_df_copy = original_df.copy()
            orig_df_copy["Finding Labels"] = orig_df_copy["Finding Labels"].str.split('|')
        else:
            orig_df_copy = original_df
        orig_counts = count_labels(orig_df_copy)
    # Get all unique labels
    all_labels = set()
    for counts in [train_counts, val_counts, test_counts]:
        all_labels.update(counts.keys())
    all_labels = sorted(list(all_labels))
    # Prepare data for plotting
    train_values = [train_counts.get(label, 0) for label in all_labels]
    val_values = [val_counts.get(label, 0) for label in all_labels]
    test_values = [test_counts.get(label, 0) for label in all_labels]
    # Calculate percentages
    train_size = sum(train_values)
    val_size = sum(val_values)
    test_size = sum(test_values)
    train_pct = [count/train_size*100 for count in train_values]
    val_pct = [count/val_size*100 for count in val_values]
    test_pct = [count/test_size*100 for count in test_values]
    # For original dataset if provided
    if original_df is not None:
        orig_values = [orig_counts.get(label, 0) for label in all_labels]
        orig_size = sum(orig_values)
        orig_pct = [count/orig_size*100 for count in orig_values]
    # Setup for plot
    plt.figure(figsize=(15, 6))
    x = np.arange(len(all_labels))
    width = 0.2
    plt.bar(x - width*1.5, train_pct, width, label='Train')
    plt.bar(x - width/2, val_pct, width, label='Validation')
    plt.bar(x + width/2, test_pct, width, label='Test')
    if original_df is not None:
        plt.bar(x + width*1.5, orig_pct, width, label='Original')
    plt.ylabel('Percentage (%)')
    plt.title('Relative Label Distribution')
    plt.xticks(x, all_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_dir = "imgs"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, save_name)
    plt.savefig(save_path)
    # Print summary statistics
    print("Dataset sizes:")
    print(f"Train: {len(train_df)} samples with {sum(train_values)} labels (avg {sum(train_values)/len(train_df):.2f} labels per sample)")
    print(f"Validation: {len(val_df)} samples with {sum(val_values)} labels (avg {sum(val_values)/len(val_df):.2f} labels per sample)")
    print(f"Test: {len(test_df)} samples with {sum(test_values)} labels (avg {sum(test_values)/len(test_df):.2f} labels per sample)")
    if original_df is not None:
        print(f"Original: {len(original_df)} samples with {sum(orig_values)} labels (avg {sum(orig_values)/len(original_df):.2f} labels per sample)")


if __name__ == "__main__":
    df = pd.read_csv("data/Data_Entry_2017.csv")
    # Split corrosponding to the original distribution
    train_df, val_df, test_df = multilabel_split(df)
    show_split_distribution(train_df, val_df, test_df, df, save_name="label_distribution.png")
    # Split with balanced distribution
    train_df, val_df, test_df = multilabel_balanced_split(df)
    show_split_distribution(train_df, val_df, test_df, df, save_name="balanced_label_distribution.png")
