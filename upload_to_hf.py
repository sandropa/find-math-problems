"""
One-time script to upload aops.csv to Hugging Face Hub.

Usage:
    python upload_to_hf.py

Reads HUGGINGFACE_KEY and repo is hardcoded as sandropa/aops-problems.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

DATASET_PATH = os.path.join("dataset", "aops.csv")
REPO_ID = "sandropa/aops-problems"
HF_TOKEN = os.environ["HUGGINGFACE_KEY"]

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"{DATASET_PATH} not found. Run download_data.py first.")

api = HfApi(token=HF_TOKEN)

print(f"Creating repo {REPO_ID} (if it doesn't exist)...")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False, token=HF_TOKEN)

print(f"Uploading {DATASET_PATH} ({os.path.getsize(DATASET_PATH) / 1e6:.1f} MB)...")
api.upload_file(
    path_or_fileobj=DATASET_PATH,
    path_in_repo="aops.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{REPO_ID}")
