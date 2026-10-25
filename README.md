# 📈 ML-based Investment Advisory System

An investment advisor for Indian investors that combines **NSE market data** and **mutual fund data** with machine learning to recommend a personalised portfolio and SIP plan.

> Academic mini-project (Machine Learning / Data Science). Not SEBI-registered financial advice.

## Datasets
| Dataset | Content | Used for |
|---|---|---|
| [NSE 30 Indices Daily Data (2000-)](https://www.kaggle.com/datasets/vinay20/stock-market-nse-indices-daily-data2000) | Daily OHLC for 30 NSE indices (`NSE_Daily_indices_data.csv`) | Index risk/return analysis + market-direction model |
| [Mutual Funds India - Detailed](https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed) | ~800 schemes: returns, risk level, Sharpe, expense ratio, etc. (`comprehensive_mutual_funds_data.csv`) | Fund return model + fund ranking |

## How it works
```
Investor answers 5 questions ──► Risk profile (Conservative … Aggressive)
                                        │
NSE data ──► ML classifier ──► Market outlook (Bullish / Neutral / Bearish) ──► small allocation tilt
                                        │
Fund data ──► ML regressor ──► Predicted 3-yr return ──► Fund score (0-100)
                                        │
                                        ▼
              Asset allocation  +  Top funds per bucket  +  SIP wealth projection
```

| Part | Method | Why |
|---|---|---|
| **A. Market** | CAGR, volatility, Sharpe, max drawdown per index. Logistic Regression vs Random Forest predicting if NIFTY 50 is up after 21 trading days, time-based split, compared to an "always UP" baseline | Direction (not exact price) is the honest framing; a time split avoids future leakage |
| **B. Funds** | Linear Regression vs Random Forest predicting `returns_3yr` inside an sklearn `Pipeline` (impute → scale/one-hot → model), 5-fold CV | Sharpe/Sortino/Alpha/5-yr return are **excluded** from inputs because they are computed from returns (data leakage) |
| **C. Advisor** | Scored risk questionnaire → equity/hybrid/debt split + max SEBI risk level → top funds by score that fit min SIP/lumpsum → future value | Transparent rules where no labelled data exists; ML outlook shifts allocation by max 10% |

**Fund score** = 35% predicted return + 25% Sharpe + 15% Sortino + 15% low expense ratio + 10% rating (each min-max scaled).

## Project structure
```
investment-advisor/
├── notebooks/Investment_Advisory.ipynb   # full walkthrough: EDA, models, results (run on Colab)
├── src/
│   ├── data_loader.py     # loading + cleaning both datasets
│   ├── market_model.py    # Part A
│   ├── fund_model.py      # Part B
│   └── advisor.py         # Part C
├── app.py                 # Streamlit web app
├── main.py                # run the whole pipeline from terminal, saves charts to outputs/
├── scripts/
│   ├── download_data.py   # downloads Kaggle data into data/
│   ├── make_sample_data.py# fake data, only to test that code runs
│   └── build_notebook.py  # regenerates the notebook
└── requirements.txt
```

## Run it
### Option 1: Google Colab (easiest)
Open `notebooks/Investment_Advisory.ipynb` in Colab (File → Open notebook → GitHub → paste this repo URL) and run all cells. It clones the repo and downloads the data automatically.

### Option 2: Local
```bash
git clone https://github.com/atharvaspardeshi/investment-advisor.git
cd investment-advisor
pip install -r requirements.txt
python scripts/download_data.py      # gets both Kaggle datasets into data/
python main.py                       # trains models, saves results + charts to outputs/
streamlit run app.py                 # opens the web app in the browser
```
Use `python main.py --benchmark "NIFTY BANK"` to model a different index.

## Results (real Kaggle data, run on Google Colab)

### Market model: will NIFTY 50 be up after 21 trading days?
Test period: Oct 2019 to Apr 2025 (time-based split, train on 1997-2019).

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (always UP) | 0.631 | 0.773 | 0.500 |
| Logistic Regression | 0.610 | 0.758 | 0.525 |
| **Random Forest** (selected) | 0.622 | 0.767 | **0.546** |

The market went up in 63% of months, so "always UP" already scores 63% accuracy. The Random Forest ranks up/down months slightly better than chance (AUC 0.546) but doesn't beat the baseline on accuracy. This matches the efficient-market hypothesis and is why the advisor uses the outlook only as a small (max 10%) allocation tilt.

### Fund model: predict 3-year return (768 funds)
| Model | 5-fold CV R² | Test R² | Test MAE |
|---|---|---|---|
| Linear Regression | 0.804 | 0.677 | 3.21% |
| **Random Forest** (selected) | 0.852 | **0.897** | **2.21%** |

Top features: standard deviation (0.74 importance), risk level, 1-year return, small-cap sub-category. Higher-volatility funds earned higher 3-year returns in this period.

### Index highlights (30 NSE indices)
- NIFTY 50 (1997-2025): 12.4% CAGR, 22.7% volatility, worst fall −59.9%.
- Best Sharpe among long-history indices: NIFTY IT (0.52) and NIFTY Midcap (0.51). Weakest: Realty, Media, Metal, PSU Bank.

### Sample advice
| Investor | Profile | Equity / Hybrid / Debt | Expected return | Invested → projected |
|---|---|---|---|---|
| Student, 21, ₹3,000 SIP, 10 yrs | Moderately Aggressive | 65 / 15 / 20 | 14.0% | ₹3.6L → ₹7.5L |
| Professional, 35, ₹20,000 SIP, 15 yrs | Aggressive | 80 / 10 / 10 | 14.5% | ₹36L → ₹1.18 Cr |
| Retiree, 62, ₹5L lumpsum, 3 yrs | Moderately Conservative | 30 / 25 / 45 | 12.3% | ₹5L → ₹7.1L |

Projections use model-predicted returns capped at 15%. The 2021-24 bull run inflates past 3-year returns, so real future returns are likely lower.

## Limitations & future scope
- Fund data is a single snapshot; past returns don't guarantee future returns.
- Market direction is close to random (efficient-market hypothesis), so it's used only as a small tilt.
- No tax, exit load or inflation modelling; risk-free rate fixed at 6.5%.
- Future: per-fund NAV history, LSTM forecasting, Markowitz portfolio optimisation, live data via APIs.

## Team
Atharva Pardeshi, T.Y. AI&DS, DYPCOE Akurdi
