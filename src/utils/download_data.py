"""
Download datasets from Kaggle if they don't exist locally.
Requires Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import os
import subprocess
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_path, unzip=True):
    """Download a dataset from Kaggle if it doesn't exist."""
    output_dir = Path(output_path)

    # Check if data already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Data already exists at {output_path}")
        return True

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download using Kaggle API
    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)]
        if unzip:
            cmd.append("--unzip")

        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install it with: pip install kaggle")
        return False

def main():
    """Download all required datasets."""
    print("Checking for required datasets...\n")

    # Download IMDB data
    download_kaggle_dataset(
        dataset_name="ishikajohari/imdb-data-with-descriptions",
        output_path="data/imdb"
    )

    # Download Goodreads data
    download_kaggle_dataset(
        dataset_name="ishikajohari/best-books-10k-multi-genre-data",
        output_path="data/goodreads"
    )

    print("\n✓ All datasets ready!")

if __name__ == "__main__":
    main()
