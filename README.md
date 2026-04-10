# AoPS Problem Finder

A conversational search tool for math competition problems from [Art of Problem Solving](https://artofproblemsolving.com). Describe a problem in any language and the agent finds it in the dataset with a direct AoPS link.

## Dataset

43,218 problems scraped from AoPS, hosted on [Hugging Face](https://huggingface.co/datasets/sandropa/aops-problems).

Columns: `category`, `contest`, `link`, `name`, `source`, `problem`, `problem_html`, `solution_1`–`solution_5`, `which`

## Local development

### 1. Create environment

```bash
conda create -n math_problems python=3.11
conda activate math_problems
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file:

```
OPENROUTER_KEY=sk-or-...
```

### 3. Download the dataset

```bash
python download_data.py
```

This requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in your `.env`. Get them from [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**.

Alternatively, the app will auto-download from Hugging Face on first run if the local file is missing.

### 4. Run the app

```bash
streamlit run app.py
```

Or test the agent directly from the terminal:

```bash
python agent/agent.py
```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Required secrets:

```toml
OPENROUTER_KEY = "sk-or-..."
HF_DATASET_REPO = "sandropa/aops-problems"
```

## Architecture

- `agent/tools.py` — pandas search tools over `problem_html` column
- `agent/agent.py` — LangChain agent using OpenRouter (`anthropic/claude-haiku-4-5`)
- `app.py` — Streamlit chat UI, dataset cached in memory per session
- `download_data.py` — one-time Kaggle download
- `upload_to_hf.py` — one-time Hugging Face upload
