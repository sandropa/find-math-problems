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


def _search_column(df: pd.DataFrame, column: str, terms: list[str], limit: int = 5) -> pd.DataFrame:
    """Search a column for rows matching all given terms. Falls back to single-term matches."""
    mask = pd.Series([True] * len(df))
    for term in terms[:5]:
        try:
            term_mask = df[column].str.contains(re.escape(term), case=False, regex=True, na=False)
        except Exception:
            term_mask = df[column].str.contains(term, case=False, regex=False, na=False)
        mask = mask & term_mask

    results = df[mask].head(limit)

    if results.empty:
        # Fallback: try pairs of terms
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pair_mask = pd.Series([True] * len(df))
                for t in [terms[i], terms[j]]:
                    try:
                        pair_mask = pair_mask & df[column].str.contains(re.escape(t), case=False, regex=True, na=False)
                    except Exception:
                        pair_mask = pair_mask & df[column].str.contains(t, case=False, regex=False, na=False)
                results = df[pair_mask].head(limit)
                if not results.empty:
                    return results

    if results.empty:
        # Last resort: any single term
        for term in terms:
            try:
                results = df[df[column].str.contains(re.escape(term), case=False, regex=True, na=False)].head(limit)
            except Exception:
                results = df[df[column].str.contains(term, case=False, regex=False, na=False)].head(limit)
            if not results.empty:
                break

    return results


def _format_results(results: pd.DataFrame) -> str:
    if results.empty:
        return "No problems found matching that query."

    out = []
    for idx, row in results.iterrows():
        preview = _clean_html(row['problem_html'])[:300]
        source = row.get('source', '')
        link = f"https://artofproblemsolving.com{source}" if isinstance(source, str) and source.startswith('/') else source
        out.append(
            f"[{idx}] {row.get('contest', '?')} — {row.get('name', '?')}\n"
            f"  Preview: {preview}...\n"
            f"  Link: {link}"
        )

    return "\n\n".join(out)


@tool
def search_problems(terms: list[str]) -> str:
    """Search the AoPS dataset for math competition problems.

    Pass a list of search terms. Each term is matched against the problem's HTML content
    (which includes LaTeX in alt= attributes of img tags).

    IMPORTANT: The dataset stores math as LaTeX in HTML. Effective search terms are:
    - LaTeX expressions: "ac=bd", "\\frac{a}{b}", "x^2", "x_1", "\\geq"
    - English math words: "triangle", "circle", "prime", "integer", "maximum"
    - Numbers that appear in the problem: "100", "2024", "4"
    - Contest names: "IMO", "USAMO", "Putnam"

    DO NOT search with words in other languages — the dataset is in English/LaTeX.

    Examples:
        terms=["ac=bd", "\\frac{a}{b}", "\\frac{c}{d}"]
        terms=["100", "nice", "cell", "token"]
        terms=["y^2-1", "a^2", "x^2-1", "minimize"]
    """
    df = get_df()

    if not terms:
        return "Please provide at least one search term."

    results = _search_column(df, 'problem_html', terms)
    return _format_results(results)


@tool
def search_by_contest(contest_name: str) -> str:
    """Search for problems by contest name.

    Examples: "CentroAmerican", "IMO Shortlist", "All Russian", "JBMO"
    """
    df = get_df()
    mask = df['contest'].str.contains(contest_name, case=False, regex=False, na=False)
    results = df[mask].head(10)
    return _format_results(results)


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
