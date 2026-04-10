"""
Download the AoPS dataset from Kaggle into dataset/aops.csv.

Usage:
    python download_data.py

Requires KAGGLE_USERNAME and KAGGLE_KEY set in .env or environment.
"""
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

import kagglehub

DATASET_PATH = os.path.join("dataset", "aops.csv")

if os.path.exists(DATASET_PATH):
    print(f"Dataset already exists at {DATASET_PATH}, skipping download.")
else:
    print("Downloading from Kaggle...")
    kaggle_dir = kagglehub.dataset_download("imbishal7/math-olympiad-problems-and-solutions-aops")
    src = os.path.join(kaggle_dir, "aops.csv")
    os.makedirs("dataset", exist_ok=True)
    shutil.copy(src, DATASET_PATH)
    print(f"Saved to {DATASET_PATH} ({os.path.getsize(DATASET_PATH) / 1e6:.1f} MB)")
