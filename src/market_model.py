"""
Part A — MARKET ANALYSIS on NSE index data.

1. Risk/return statistics for every index (CAGR, volatility, Sharpe, max drawdown)
2. ML model: predict if the benchmark (NIFTY 50) will be UP over the next ~1 month
   -> used as a "market outlook" signal by the advisor.

Why classification (UP/DOWN) and not exact price prediction?
  Exact prices are almost impossible to predict and a regression on price looks
  great (R² ≈ 0.99) only because tomorrow's price ≈ today's price. Direction is an
  honest, standard framing and gives a probability we can act on.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRADING_DAYS = 252
RISK_FREE = 0.065          # ~6.5% Indian govt bond yield (annual)
HORIZON = 21               # trading days ≈ 1 month


# ----------------------------------------------------------------------------
# 1) Index statistics
# ----------------------------------------------------------------------------
def index_stats(prices: pd.DataFrame, min_years: float = 3) -> pd.DataFrame:
    rows = []
    for col in prices.columns:
        if "VIX" in col.upper():        # volatility index: not investable, skip
            continue
        p = prices[col].dropna()
        years = len(p) / TRADING_DAYS
        if years < min_years:
            continue
        r = p.pct_change().dropna()
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
        vol = r.std() * np.sqrt(TRADING_DAYS)
        drawdown = (p / p.cummax() - 1).min()
        rows.append({
            "index": col,
            "start": p.index[0].date(),
            "end": p.index[-1].date(),
            "years": round(years, 1),
            "cagr_%": round(cagr * 100, 2),
            "volatility_%": round(vol * 100, 2),
            "sharpe": round((cagr - RISK_FREE) / vol, 2) if vol > 0 else np.nan,
            "max_drawdown_%": round(drawdown * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2) Feature engineering
# ----------------------------------------------------------------------------
def rsi(price: pd.Series, n: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def make_features(price: pd.Series) -> pd.DataFrame:
    """Only PAST information is used for each row (no look-ahead)."""
    df = pd.DataFrame(index=price.index)
    r = price.pct_change()
    for n in (5, 21, 63):
        df[f"ret_{n}d"] = price.pct_change(n)                  # momentum
    df["vol_21d"] = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["vol_63d"] = r.rolling(63).std() * np.sqrt(TRADING_DAYS)
    df["price_vs_ma50"] = price / price.rolling(50).mean() - 1
    df["price_vs_ma200"] = price / price.rolling(200).mean() - 1
    df["rsi_14"] = rsi(price)
    df["drawdown"] = price / price.cummax() - 1
    # target: 1 if price is higher after HORIZON days
    df["target"] = (price.shift(-HORIZON) > price).astype(int)
    df.loc[price.shift(-HORIZON).isna(), "target"] = np.nan
    return df


# ----------------------------------------------------------------------------
# 3) Train + evaluate
# ----------------------------------------------------------------------------
def train_market_model(price: pd.Series, test_size: float = 0.2):
    """
    Time-based split (train on past, test on future).
    Why not random train_test_split?  Shuffling leaks future data into training
    and gives fake-high accuracy. In finance we must always split by time.
    """
    data = make_features(price.dropna())
    latest_features = data.drop(columns="target").dropna().iloc[[-1]]   # today's row
    data = data.dropna()

    X, y = data.drop(columns="target"), data["target"].astype(int)
    split = int(len(X) * (1 - test_size))
    # gap of HORIZON rows so training targets don't overlap the test period
    X_train, y_train = X.iloc[:split - HORIZON], y.iloc[:split - HORIZON]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    models = {
        "Baseline (always UP)": None,
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                min_samples_leaf=20, random_state=42, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, model in models.items():
        if model is None:
            pred = np.ones(len(y_test), dtype=int)
            proba = np.full(len(y_test), y_train.mean())
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            fitted[name] = model
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3) if y_test.nunique() > 1 else np.nan,
        })

    results = pd.DataFrame(results)
    best_name = results[results.model != "Baseline (always UP)"].sort_values("roc_auc").iloc[-1]["model"]
    best = fitted[best_name]
    best.fit(X, y)                                       # refit on ALL data for live use
    prob_up = float(best.predict_proba(latest_features)[0, 1])

    importance = None
    if hasattr(best, "feature_importances_"):
        importance = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "results": results,
        "best_model_name": best_name,
        "model": best,
        "prob_up": prob_up,
        "as_of": latest_features.index[-1].date(),
        "feature_importance": importance,
        "test_period": (X_test.index[0].date(), X_test.index[-1].date()),
    }


def outlook_label(prob_up: float) -> str:
    if prob_up >= 0.60:
        return "Bullish"
    if prob_up <= 0.45:
        return "Bearish"
    return "Neutral"
