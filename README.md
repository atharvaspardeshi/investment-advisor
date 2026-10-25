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

## Results
Run `python main.py` and copy the tables from `outputs/market_model_results.csv` and `outputs/fund_model_results.csv` here.

| Model | Metric | Value |
|---|---|---|
| Market: Logistic Regression | ROC-AUC (test) | _fill_ |
| Market: Random Forest | ROC-AUC (test) | _fill_ |
| Funds: Linear Regression | R² / MAE (test) | _fill_ |
| Funds: Random Forest | R² / MAE (test) | _fill_ |

## Limitations & future scope
- Fund data is a single snapshot; past returns don't guarantee future returns.
- Market direction is close to random (efficient-market hypothesis), so it's used only as a small tilt.
- No tax, exit load or inflation modelling; risk-free rate fixed at 6.5%.
- Future: per-fund NAV history, LSTM forecasting, Markowitz portfolio optimisation, live data via APIs.

## Team
Atharva Pardeshi, T.Y. AI&DS, DYPCOE Akurdi
