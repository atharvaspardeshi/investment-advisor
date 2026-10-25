"""
Creates SMALL FAKE data with the same shape as the Kaggle datasets.
Only for testing that the code runs. For the real project, download the
Kaggle CSVs into data/ (see README) - never submit results from fake data.

    python scripts/make_sample_data.py
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
os.makedirs("data/nse", exist_ok=True)
os.makedirs("data/mutual_funds", exist_ok=True)

# --- NSE indices: ONE CSV with a SYMBOL column (same layout as the Kaggle file) ---
dates = pd.bdate_range("2000-01-03", "2025-05-16")
frames = []
for name, mu, sig in [("NIFTY 50", 0.12, 0.20), ("NIFTY BANK", 0.14, 0.27),
                      ("NIFTY IT", 0.13, 0.25), ("NIFTY MIDCAP 100", 0.15, 0.24),
                      ("NIFTYJR", 0.14, 0.24)]:
    r = rng.normal(mu / 252, sig / np.sqrt(252), len(dates))
    close = 1000 * np.exp(np.cumsum(r))
    frames.append(pd.DataFrame({"DATETIME": dates.strftime("%d-%m-%Y"), "SYMBOL": name,
                                "OPEN": close * 0.998, "HIGH": close * 1.006, "LOW": close * 0.994,
                                "CLOSE": close.round(2), "VOLUME": 0,
                                "PREVIOUS_CLOSE": np.r_[np.nan, close[:-1]].round(2),
                                "LOW_CLOSE": 0.5, "HIGH_CLOSE": 0.5}))
pd.concat(frames).to_csv("data/nse/NSE_Daily_indices_data.csv", index=False)

# --- Mutual funds ---
n = 800
cats = rng.choice(["Equity", "Debt", "Hybrid", "Solution Oriented", "Other"], n, p=[.45, .3, .15, .05, .05])
sub = {"Equity": ["Large Cap Mutual Funds", "Mid Cap Mutual Funds", "Small Cap Mutual Funds", "ELSS Mutual Funds"],
       "Debt": ["Liquid Mutual Funds", "Gilt Mutual Funds", "Corporate Bond Mutual Funds"],
       "Hybrid": ["Aggressive Hybrid Mutual Funds", "Balanced Advantage Mutual Funds"],
       "Solution Oriented": ["Retirement Funds"], "Other": ["Index Funds"]}
risk = {"Equity": [5, 6], "Debt": [1, 2, 3], "Hybrid": [3, 4, 5], "Solution Oriented": [4, 5, 6], "Other": [5, 6]}
base = {"Equity": 16, "Debt": 6.5, "Hybrid": 12, "Solution Oriented": 13, "Other": 14}
rows = []
for i in range(n):
    c = cats[i]
    sd = {"Equity": 15, "Debt": 2, "Hybrid": 9}.get(c, 12) + rng.normal(0, 2)
    er = abs(rng.normal(1.0, 0.5))
    r3 = base[c] + rng.normal(0, 3) - 1.2 * er + 0.1 * sd
    rows.append({
        "scheme_name": f"Sample {c} Fund {i}", "min_sip": rng.choice([100, 500, 1000]),
        "min_lumpsum": rng.choice([500, 1000, 5000]), "expense_ratio": round(er, 2),
        "fund_size_cr": round(abs(rng.normal(3000, 4000)), 1), "fund_age_yr": rng.integers(1, 20),
        "fund_manager": "X", "sortino": round(r3 / 8 + rng.normal(0, .3), 2),
        "alpha": round(rng.normal(1, 3), 2), "sd": round(sd, 2), "beta": round(abs(rng.normal(.9, .2)), 2),
        "sharpe": round(r3 / max(sd, 1) + rng.normal(0, .2), 2), "risk_level": rng.choice(risk[c]),
        "amc_name": f"AMC {rng.integers(1, 30)}", "rating": rng.integers(0, 6), "category": c,
        "sub_category": rng.choice(sub[c]), "returns_1yr": round(r3 + rng.normal(0, 6), 1),
        "returns_3yr": round(r3, 1), "returns_5yr": round(r3 + rng.normal(0, 2), 1) if rng.random() > .2 else None,
    })
pd.DataFrame(rows).to_csv("data/mutual_funds/comprehensive_mutual_funds_data.csv", index=False)
print("Sample data written to data/nse and data/mutual_funds")
