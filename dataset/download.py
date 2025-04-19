import os
import kaggle
import kagglehub
import argparse


def download_dataset_kaggle(download_dir="./data"):
    print("Starting download of NIH Chest X-ray dataset...")
    os.makedirs(download_dir, exist_ok=True)
    dataset = "nih-chest-xrays/data"
    kaggle.api.dataset_download_files(dataset, path=download_dir, unzip=True)
    print("Download completed.")


def download_dataset_kagglehub():
    print("Starting download of NIH Chest X-ray dataset...")
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the NIH Chest X-ray dataset.")
    parser.add_argument("--dir", type=str, default="./data", help="Directory to save the dataset.")
    parser.add_argument("--use_kaggle_api", action="store_true", help="Use kagglehub for downloading.")
    args = parser.parse_args()
    download_dir = args.dir
    use_kaggle_api = args.use_kaggle_api
    if not use_kaggle_api:
        downloaded_path = download_dataset_kagglehub()
        print(f"The dataset has been downloaded to {downloaded_path}.")
        print('Please manually move the dataset to the correct folder under "./data"!')
    else:
        download_dataset_kaggle(download_dir)
