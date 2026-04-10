import os
import re
import pandas as pd
from langchain_core.tools import tool

_df: pd.DataFrame | None = None

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "aops.csv")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "sandropa/aops-problems")


def ensure_dataset() -> str:
    """Return path to aops.csv. Uses local file if present, otherwise downloads from HF Hub."""
    if os.path.exists(DATASET_PATH):
        return DATASET_PATH

    if not HF_DATASET_REPO:
        raise RuntimeError(
            "Dataset not found locally and HF_DATASET_REPO is not set. "
            "Run upload_to_hf.py first and set HF_DATASET_REPO in your environment."
        )

    print(f"Downloading dataset from Hugging Face ({HF_DATASET_REPO})...")
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=HF_DATASET_REPO, filename="aops.csv", repo_type="dataset")
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    import shutil
    shutil.copy(src, DATASET_PATH)
    print(f"Dataset ready at {DATASET_PATH}")
    return DATASET_PATH


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(ensure_dataset())
    return _df


def _clean_html(html: str) -> str:
    """Extract LaTeX from alt= attributes and strip remaining HTML tags."""
    if not isinstance(html, str):
        return ""
    text = re.sub(r'<img[^>]+alt="([^"]+)"[^>]*>', r'\1', html)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


@tool
def search_problems(query: str) -> str:
    """Search the AoPS math competition dataset for problems matching a query.

    The query should contain key mathematical terms, numbers, or LaTeX expressions
    from the problem. Returns up to 5 matching problems with their index, contest,
    name, and a short preview.

    Example queries:
    - "minimize a/x where y^2 - 1 = a^2(x^2 - 1)"
    - "100x100 board nice cell token"
    - "system x_1 + 2x_2 harmonic"
    """
    df = get_df()

    terms = [t.strip() for t in re.split(r'\s+', query) if len(t.strip()) > 2]

    mask = pd.Series([True] * len(df))
    for term in terms[:4]:
        escaped = re.escape(term)
        try:
            term_mask = df['problem_html'].str.contains(escaped, case=False, regex=True, na=False)
        except Exception:
            term_mask = df['problem_html'].str.contains(term, case=False, regex=False, na=False)
        mask = mask & term_mask

    results = df[mask].head(5)

    if results.empty:
        for term in terms:
            escaped = re.escape(term)
            try:
                results = df[df['problem_html'].str.contains(escaped, case=False, regex=True, na=False)].head(5)
            except Exception:
                results = df[df['problem_html'].str.contains(term, case=False, regex=False, na=False)].head(5)
            if not results.empty:
                break

    if results.empty:
        return "No problems found matching that query."

    out = []
    for idx, row in results.iterrows():
        preview = _clean_html(row['problem_html'])[:200]
        source = row.get('source', '')
        link = f"https://artofproblemsolving.com{source}" if isinstance(source, str) and source.startswith('/') else source
        out.append(
            f"[{idx}] {row.get('contest', '?')} — {row.get('name', '?')}\n"
            f"  Preview: {preview}...\n"
            f"  Link: {link}"
        )

    return "\n\n".join(out)


@tool
def get_problem_details(row_index: int) -> str:
    """Get the full problem statement and AoPS link for a specific row index.

    Use the index returned by search_problems. Returns the full problem text,
    contest info, and AoPS community link.
    """
    df = get_df()

    if row_index not in df.index:
        return f"No row found at index {row_index}."

    row = df.loc[row_index]
    problem_text = _clean_html(row['problem_html'])
    source = row.get('source', '')
    link = f"https://artofproblemsolving.com{source}" if isinstance(source, str) and source.startswith('/') else source

    return (
        f"Contest: {row.get('contest', '?')}\n"
        f"Name: {row.get('name', '?')}\n"
        f"Category: {row.get('category', '?')}\n"
        f"Link: {link}\n\n"
        f"Problem:\n{problem_text}"
    )
