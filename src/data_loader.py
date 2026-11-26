"""Load the project's data files.

Default source is the repository's own data folder. When the notebook runs
in Colab without a clone, files are fetched from the GitHub repo's raw URL.
Upload / Google Drive options live in the notebook (they need Colab APIs).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SYN_DIR = DATA_DIR / "synthetic"

SOURCE_FILES = {
    "fir_reports": "fir_reports.json",
    "social_media": "social_media_posts.json",
    "cdr": "cdr_records.csv",
    "criminal_history": "criminal_history.csv",
    "financial": "financial_transactions.csv",
}
CASES_FILE = "fir_cases.csv"


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            return pd.DataFrame(json.load(fh))
    # phone numbers / accounts are identifiers, never numbers
    return pd.read_csv(path, dtype={"caller": str, "callee": str, "phone": str,
                                    "complainant_phone": str})


def download_from_github(raw_base_url: str, dest: Path = DATA_DIR) -> None:
    """Download the 5 source files + case register from a GitHub raw URL.

    raw_base_url example:
    https://raw.githubusercontent.com/<user>/fir-crime-insights/main/data
    """
    (dest / "synthetic").mkdir(parents=True, exist_ok=True)
    targets = [f"synthetic/{f}" for f in SOURCE_FILES.values()] + [CASES_FILE]
    for rel in targets:
        out = dest / rel
        if not out.exists():
            urllib.request.urlretrieve(f"{raw_base_url.rstrip('/')}/{rel}", out)


def load_sources(syn_dir: Path = SYN_DIR) -> dict[str, pd.DataFrame]:
    """Load the 5 provided synthetic files into a dict of DataFrames."""
    missing = [f for f in SOURCE_FILES.values() if not (syn_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files in {syn_dir}: {missing}")
    return {key: _read(syn_dir / fname) for key, fname in SOURCE_FILES.items()}


def load_cases(path: Path = DATA_DIR / CASES_FILE, regenerate_if_missing: bool = True) -> pd.DataFrame:
    """Load the FIR case register; regenerate it from the seeded script if absent."""
    if not path.exists() and regenerate_if_missing:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen", DATA_DIR / "generate_fir_cases.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        gen.generate().to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return _read(path)


def describe_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, non-null, missing count and % (mid-term stage A)."""
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    out.index.name = f"{name} column"
    return out
