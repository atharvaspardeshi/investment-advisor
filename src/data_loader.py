"""
Data loading + cleaning for both Kaggle datasets.

Why a separate loader?  Kaggle CSVs often have messy column names
("Return_1yr (%)", "Expense ratio"), numbers stored as text ("1,234.5", "-"),
and dates in different formats. Cleaning everything in ONE place means the
models, notebook and web app all see the same clean data.
"""
import glob
import os
import re

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean_col(name: str) -> str:
    """'Return_1yr (%)' -> 'return_1yr'   |   'Expense ratio' -> 'expense_ratio'"""
    name = str(name).strip().lower()
    name = re.sub(r"\(.*?\)", "", name)          # drop "(%)", "(Cr)"
    name = re.sub(r"[^a-z0-9]+", "_", name)      # spaces/symbols -> _
    return name.strip("_")


def to_number(series: pd.Series) -> pd.Series:
    """Convert text like '1,234.5', '12%', '-' into float (NaN if impossible)."""
    if series.dtype.kind in "if":
        return series.astype(float)
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(s.str.strip(), errors="coerce")


# ----------------------------------------------------------------------------
# 1) MUTUAL FUNDS  (ravibarnawal/mutual-funds-india-detailed)
# ----------------------------------------------------------------------------
# Different versions of the CSV use slightly different headers, so we map
# every known spelling to ONE standard name.
MF_SYNONYMS = {
    "scheme_name": ["scheme_name", "scheme", "fund_name", "name"],
    "min_sip": ["min_sip", "minimum_sip"],
    "min_lumpsum": ["min_lumpsum", "minimum_lumpsum"],
    "expense_ratio": ["expense_ratio", "expense"],
    "fund_size_cr": ["fund_size_cr", "fund_size", "aum", "aum_cr"],
    "fund_age_yr": ["fund_age_yr", "fund_age", "age"],
    "fund_manager": ["fund_manager", "manager"],
    "sortino": ["sortino", "sortino_ratio"],
    "alpha": ["alpha"],
    "sd": ["sd", "standard_deviation", "std_dev", "std"],
    "beta": ["beta"],
    "sharpe": ["sharpe", "sharpe_ratio"],
    "risk_level": ["risk_level", "risk"],
    "amc_name": ["amc_name", "amc", "fund_house"],
    "rating": ["rating"],
    "category": ["category"],
    "sub_category": ["sub_category", "subcategory"],
    "returns_1yr": ["returns_1yr", "return_1yr", "return_1_yr", "1yr_return"],
    "returns_3yr": ["returns_3yr", "return_3yr", "return_3_yr", "3yr_return"],
    "returns_5yr": ["returns_5yr", "return_5yr", "return_5_yr", "5yr_return"],
}
MF_NUMERIC = ["min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr", "fund_age_yr",
              "sortino", "alpha", "sd", "beta", "sharpe", "risk_level", "rating",
              "returns_1yr", "returns_3yr", "returns_5yr"]


def load_mutual_funds(path: str) -> pd.DataFrame:
    """Load + clean the mutual fund CSV. `path` = file or folder containing it."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV found in {path}")
        path = files[0]

    df = pd.read_csv(path)
    df.columns = [clean_col(c) for c in df.columns]

    # rename to standard names
    rename = {}
    for std, options in MF_SYNONYMS.items():
        for opt in options:
            if opt in df.columns and std not in rename.values():
                rename[opt] = std
                break
    df = df.rename(columns=rename)

    for col in MF_NUMERIC:
        if col in df.columns:
            df[col] = to_number(df[col])

    for col in ["category", "sub_category", "amc_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # basic cleaning
    df = df.drop_duplicates(subset=["scheme_name"]) if "scheme_name" in df else df.drop_duplicates()
    if "returns_3yr" in df.columns:
        df = df.dropna(subset=["returns_3yr"])       # target for our model
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2) NSE INDICES  (vinay20/stock-market-nse-indices-daily-data2000)
# ----------------------------------------------------------------------------
def _find_col(cols, keywords, exclude=()):
    for c in cols:                      # exact match first ("close" before "low_close")
        if c in keywords:
            return c
    for c in cols:
        if any(k in c for k in keywords) and not any(e in c for e in exclude):
            return c
    return None


def load_nse_indices(folder: str) -> pd.DataFrame:
    """
    Returns a WIDE table: one row per date, one column per index (closing value).

    Works for both layouts Kaggle datasets usually have:
      a) one CSV per index  (e.g. NIFTY 50.csv, NIFTY BANK.csv)
      b) one big CSV with an 'Index Name' column
    """
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    series = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [clean_col(c) for c in df.columns]
        date_col = _find_col(df.columns, ["date", "timestamp"])
        close_col = _find_col(df.columns, ["close"], exclude=["prev", "low_", "high_"])
        if date_col is None or close_col is None:
            print(f"[skip] {os.path.basename(f)}: no date/close column")
            continue

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, format="mixed")
        df[close_col] = to_number(df[close_col])
        df = df.dropna(subset=[date_col, close_col])

        name_col = _find_col(df.columns, ["index_name", "index", "symbol", "name"],
                             exclude=[date_col])
        if name_col is not None and df[name_col].nunique() > 1:      # layout (b)
            for name, g in df.groupby(name_col):
                series[str(name).strip().upper()] = g.set_index(date_col)[close_col]
        else:                                                        # layout (a)
            name = os.path.splitext(os.path.basename(f))[0]
            name = re.sub(r"[_\-]+", " ", name).strip().upper()
            series[name] = df.set_index(date_col)[close_col]

    wide = pd.DataFrame({k: v[~v.index.duplicated()] for k, v in series.items()})
    return wide.sort_index()


def pick_benchmark(prices: pd.DataFrame) -> str:
    """Use NIFTY 50 as the market benchmark if present, else the longest series."""
    for c in prices.columns:
        if re.sub(r"[^A-Z0-9]", "", c) in ("NIFTY50", "NIFTY", "NIFTY50INDEX", "NSEI", "CNXNIFTY"):
            return c
    return prices.notna().sum().idxmax()
