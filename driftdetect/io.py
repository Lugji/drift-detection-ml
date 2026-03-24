from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from disk into a pandas DataFrame.

    Purpose in DriftDetect:
    - This is the single place where we decide how files are read.
    - Baseline building, detection, simulation, and coverage all rely on this function.
    - It directly supports REQ-001: ingest CSV and Parquet.

    Inputs:
    - path: file path to a dataset (expects .csv or .parquet)

    Output:
    - pandas.DataFrame containing the dataset rows/columns

    Error behavior:
    - If the file does not exist -> FileNotFoundError
    - If file extension is unsupported -> ValueError
    - If Parquet reading fails (usually missing pyarrow) -> RuntimeError with helpful message
    """
    p = Path(path)

    # 1) Validate that the file actually exists.
    if not p.exists():
        raise FileNotFoundError(str(p))

    # 2) Decide how to read based on file extension.
    suf = p.suffix.lower()

    # CSV is universally supported and easy to inspect/debug.
    if suf == ".csv":
        return pd.read_csv(p)

    # Parquet is common in data engineering (faster, preserves types).
    # Pandas requires an engine like pyarrow to read it.
    if suf == ".parquet":
        try:
            return pd.read_parquet(p, engine="pyarrow")
        except Exception as e:
            # Make the error message human-friendly: most common cause is missing pyarrow.
            raise RuntimeError(
                "Failed to read Parquet. Make sure pyarrow is installed: pip install pyarrow"
            ) from e

    # If someone passes another extension, we explicitly reject it.
    raise ValueError(f"Unsupported file type: {suf} (use .csv or .parquet)")
