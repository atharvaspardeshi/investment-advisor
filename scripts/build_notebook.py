"""Builds notebooks/Investment_Advisory.ipynb (run: python scripts/build_notebook.py)."""
import json
import os

REPO = "https://github.com/atharvaspardeshi/investment-advisor.git"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, tag=None):
    cells.append({"cell_type": "code", "metadata": {"tags": [tag] if tag else []},
                  "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# 📈 ML-based Investment Advisory System
**Datasets (Kaggle):**
1. [NSE 30 Indices Daily Data (2000-)](https://www.kaggle.com/datasets/vinay20/stock-market-nse-indices-daily-data2000), daily OHLC of NSE indices
2. [Mutual Funds India - Detailed](https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed), ~800 schemes with returns, risk and cost

**Pipeline**
| Part | What | Method |
|---|---|---|
| A | Market analysis | CAGR, volatility, Sharpe, drawdown + **ML classifier** for next-month direction |
| B | Fund analysis | **ML regressor** predicting 3-yr return + weighted fund score |
| C | Advisor | Risk-profile questionnaire → asset allocation → fund picks → SIP projection |

> ⚠️ Academic project, not SEBI-registered financial advice.
""")

md("## 0. Setup\nClone the repo (code lives in `src/`) and download both datasets with `kagglehub` (pre-installed on Colab, no API key needed for public datasets).")
code(f"""
!git clone -q {REPO}
%cd investment-advisor
""", tag="colab")
code("""
import kagglehub
NSE_DIR = kagglehub.dataset_download("vinay20/stock-market-nse-indices-daily-data2000")
MF_DIR  = kagglehub.dataset_download("ravibarnawal/mutual-funds-india-detailed")
print(NSE_DIR, MF_DIR)
""", tag="colab")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.data_loader import load_mutual_funds, load_nse_indices, pick_benchmark
from src.market_model import index_stats, make_features, train_market_model, outlook_label
from src.fund_model import train_fund_model, score_funds
from src.advisor import Investor, risk_profile, recommend, project_wealth
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")
""")

# ---------------- Part A ----------------
md("""
# Part A: NSE Market Analysis
## A1. Load data
The Kaggle file is in **long format** (one row per date per index, `SYMBOL` column).
`load_nse_indices` pivots it to **wide format**: one column per index, so comparing indices is easy.
""")
code("""
prices = load_nse_indices(NSE_DIR)
print(prices.shape)
print("Indices:", list(prices.columns))
prices.tail()
""")
md("""
## A2. Risk vs return of every index
- **CAGR**: average yearly growth
- **Volatility**: yearly std-dev of daily returns (risk)
- **Sharpe** = (CAGR − 6.5% risk-free) / volatility → return per unit of risk
- **Max drawdown**: worst fall from a peak (e.g. 2008, 2020 crash)
""")
code("""
stats = index_stats(prices)
stats
""")
code("""
top = stats.head(15)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=top, x="volatility_%", y="cagr_%", size="sharpe", hue="sharpe", palette="viridis", legend=False)
for _, r in top.iterrows():
    plt.annotate(r["index"], (r["volatility_%"], r["cagr_%"]), fontsize=8)
plt.title("Risk vs Return of NSE indices"); plt.show()
""")
code("""
bench = pick_benchmark(prices)          # NIFTY 50 if present; change manually if needed: bench = "..."
print("Benchmark:", bench)
p = prices[bench].dropna()
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
p.plot(ax=ax[0], title=f"{bench} closing value")
(p / p.cummax() - 1).plot(ax=ax[1], color="crimson", title="Drawdown from peak")
plt.tight_layout(); plt.show()
""")
code("""
# correlation of daily returns: low correlation = better diversification
top_idx = stats["index"].head(10)
sns.heatmap(prices[top_idx].pct_change().corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation of daily returns"); plt.show()
""")
md("""
## A3. ML model: will the market be UP after 1 month?
**Features** (only past data, no peeking into the future): 5/21/63-day returns (momentum), 21/63-day volatility,
price vs 50/200-day moving average, RSI-14, drawdown.
**Target**: 1 if price after 21 trading days > today, else 0.

**Why a time-based split, not `train_test_split(shuffle=True)`?** Shuffling puts future rows in training, so the model
"knows" the future and accuracy looks fake-high. We train on the first 80% of years and test on the last 20%.

**Why compare against a baseline?** Markets go up ~60% of months, so a model that always says "UP" already gets ~60%.
Our model is only useful if it beats that, mainly on **ROC-AUC** (ranking ability).
""")
code("""
feat = make_features(p).dropna()
print(feat.shape); print("UP share:", round(feat.target.mean(), 3))
feat.head()
""")
code("""
mkt = train_market_model(p)
print("Test period:", mkt["test_period"])
mkt["results"]
""")
code("""
if mkt["feature_importance"] is not None:
    mkt["feature_importance"].plot.barh(figsize=(7, 4), title="Feature importance"); plt.show()
outlook = outlook_label(mkt["prob_up"])
print(f"Best model: {mkt['best_model_name']} | as of {mkt['as_of']} | P(UP next month) = {mkt['prob_up']:.2f} → {outlook}")
""")
md("""
**Interpretation:** market direction is very hard to predict (efficient-market hypothesis), so accuracy close to
the baseline is *expected*. That is why the advisor uses this signal only for a **small tilt (max 10%)** in allocation,
never for the full decision.
""")

# ---------------- Part B ----------------
md("""
# Part B: Mutual Fund Analysis
## B1. Load + clean
Columns like `sortino`, `alpha`, `sd` are stored as text with `-` for missing values. The loader converts them to numbers.
""")
code("""
funds = load_mutual_funds(MF_DIR)
print(funds.shape)
funds.head()
""")
code("""
funds.isna().sum().sort_values(ascending=False).head(10)
""")
code("""
funds.describe().T
""")
md("## B2. EDA")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
funds["category"].value_counts().plot.bar(ax=ax[0], title="Funds per category")
sns.boxplot(data=funds, x="category", y="returns_3yr", ax=ax[1]); ax[1].set_title("3-yr return by category")
plt.tight_layout(); plt.show()
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=funds, x="risk_level", y="returns_3yr", ax=ax[0]); ax[0].set_title("Higher risk level → higher & wider returns?")
sns.scatterplot(data=funds, x="expense_ratio", y="returns_3yr", hue="category", alpha=.6, ax=ax[1]); ax[1].set_title("Cost vs return")
plt.tight_layout(); plt.show()
""")
code("""
num = funds.select_dtypes("number")
plt.figure(figsize=(11, 8))
sns.heatmap(num.corr(), cmap="coolwarm", annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation matrix"); plt.show()
""")
md("""
## B3. ML model: predict 3-year return
**Inputs:** expense ratio, fund size, age, std-dev, beta, risk level, rating, min SIP, 1-yr return, category, sub-category.

**Data leakage, an important point:** `sharpe`, `sortino`, `alpha` and `returns_5yr` are *computed from* the same past returns,
so they would "give away" the answer. We **exclude** them from the model and use them only in the final ranking score.

**Pipeline:** median-impute + scale numbers, most-frequent-impute + one-hot categories → model.
Using a `Pipeline` means the preprocessing is learned from training data only (no leakage) and the same steps run on new data.

**Models:** Linear Regression (simple baseline) vs Random Forest (handles non-linear patterns). Evaluated with
5-fold cross-validation R² + test R² and MAE.
""")
code("""
fm = train_fund_model(funds)
print("Best:", fm["best_model_name"])
fm["results"]
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(fm["y_test"], fm["pred_test"], alpha=.5)
lim = [fm["y_test"].min(), fm["y_test"].max()]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Actual 3-yr return %"); ax[0].set_ylabel("Predicted"); ax[0].set_title("Actual vs Predicted")
if fm["feature_importance"] is not None:
    fm["feature_importance"].head(12).plot.barh(ax=ax[1], title="Top features")
plt.tight_layout(); plt.show()
""")
md("""
## B4. Fund score (0-100)
`35% predicted return + 25% Sharpe + 15% Sortino + 15% low expense ratio + 10% rating` (each min-max scaled to 0-1).
Combining ML prediction with risk-adjusted metrics avoids picking funds that only had one lucky year.
""")
code("""
scored = score_funds(funds, fm)
scored[["scheme_name", "category", "sub_category", "risk_level", "returns_3yr", "pred_return_3yr", "sharpe", "expense_ratio", "fund_score"]].head(15)
""")

# ---------------- Part C ----------------
md("""
# Part C: The Advisor
## C1. Risk profiling
Score out of 16 from: age, horizon, income stability, reaction to a 20% fall, experience.
→ 5 profiles from **Conservative** to **Aggressive**, each with an equity/hybrid/debt split and a maximum fund risk level (1-6, the SEBI riskometer scale used in the dataset).
Rule-based because there is no labelled "correct profile" data; this is how real AMC questionnaires work.
""")
code("""
investors = {
    "Student, 21, 10-yr goal":        Investor(age=21, horizon_years=10, monthly_sip=3000, experience=0),
    "Professional, 35, 15-yr goal":   Investor(age=35, horizon_years=15, monthly_sip=20000, income_stability=3, reaction_to_fall=3, experience=2),
    "Retiree, 62, 3-yr goal":         Investor(age=62, horizon_years=3, lumpsum=500000, income_stability=1, reaction_to_fall=0, experience=1),
}
pd.DataFrame({k: risk_profile(v) for k, v in investors.items()}).T
""")
md("## C2. Full recommendation")
code("""
for name, inv in investors.items():
    rec = recommend(scored, inv, outlook)
    proj = project_wealth(rec["picks"], inv)
    print(f"\\n=== {name} → {rec['profile']['profile']} | market: {outlook} | allocation: {rec['allocation']}")
    display(rec["picks"])
    if proj:
        print(f"Expected {proj['expected_return_%']}% p.a. | invested ₹{proj['invested_₹']:,} → ₹{proj['future_value_₹']:,}")
""")
code("""
inv = investors["Professional, 35, 15-yr goal"]
proj = project_wealth(recommend(scored, inv, outlook)["picks"], inv)
proj["yearly"].set_index("year").plot(figsize=(9, 4), title="SIP growth projection (₹)"); plt.show()
""")

md("""
# Conclusion
- **Market model:** direction of NIFTY 50 is only slightly predictable; used as a small tactical tilt, compared honestly against a baseline.
- **Fund model:** fund characteristics (category, risk, cost, volatility, 1-yr return) explain a good part of 3-yr returns, with leakage-prone features removed.
- **Advisor:** combines a transparent risk profile, ML market outlook and ML-based fund scores into a personalised allocation + SIP plan.

**Limitations:** fund data is a single snapshot (past returns ≠ future returns); no tax/exit-load modelling; risk-free rate fixed at 6.5%.
**Future scope:** NAV time-series per fund, LSTM forecasting, portfolio optimisation (Markowitz), live data via APIs.

Web app: `streamlit run app.py`
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 4}
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/Investment_Advisory.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
