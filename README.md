# Find Math Problems

Search and analyze math competition problems from AoPS and other sources.

## Datasets

### AoPS Dataset
43,218 problems scraped from Art of Problem Solving, stored in `dataset/aops.csv`.

Columns: `category`, `contest`, `link`, `name`, `source`, `problem`, `problem_html`, `solution_1`–`solution_5`, `which`

To download the dataset programmatically:

```python
import kagglehub

path = kagglehub.dataset_download("imbishal7/math-olympiad-problems-and-solutions-aops")
print("Path to dataset files:", path)
```

### Math Problems Dataset
100,000 problems with solutions, stored in `dataset/math_problems.parquet`.

To download:

```python
import kagglehub

path = kagglehub.dataset_download("YOUR_DATASET_SLUG_HERE")
print("Path to dataset files:", path)
```

## Setup

### 1. Create conda environment

```bash
conda create -n math_problems python=3.11
conda activate math_problems
pip install pandas pyarrow kagglehub jupyter
```

### 2. Configure Kaggle API key

Required for downloading datasets via `kagglehub`.

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
2. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, set environment variables:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```
