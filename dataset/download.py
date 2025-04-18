import os
import kaggle


def download_dataset(download_dir="./data"):
    print("Starting download of NIH Chest X-ray dataset...")
    os.makedirs(download_dir, exist_ok=True)
    dataset = "nih-chest-xrays/data"
    kaggle.api.dataset_download_files(dataset, path=download_dir, unzip=True)
    print("Download completed.")


if __name__ == "__main__":
    download_dataset()
