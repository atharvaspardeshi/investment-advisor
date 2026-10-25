"""
Part B — MUTUAL FUND MODEL.

Goal: predict a fund's 3-year return from its characteristics, then rank funds
by a score that mixes predicted return, risk-adjusted performance and cost.

Leakage note (important for viva!):
  sharpe, sortino, alpha and returns_5yr are CALCULATED from past returns, so
  they already "contain" returns_3yr. Using them as inputs would make the model
  look perfect without learning anything. So we leave them OUT of the model and
  only use them later in the ranking score.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "returns_3yr"
NUM_FEATURES = ["expense_ratio", "fund_size_cr", "fund_age_yr", "sd", "beta",
                "risk_level", "rating", "min_sip", "returns_1yr"]
CAT_FEATURES = ["category", "sub_category"]


def _available(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def build_pipeline(model, num_cols, cat_cols):
    """
    Pipeline = preprocessing + model in one object.
    Why? The same steps run on train, test and new data automatically, and the
    imputer/scaler are fit ONLY on training data (no leakage).
    """
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def train_fund_model(funds: pd.DataFrame):
    num_cols, cat_cols = _available(funds, NUM_FEATURES), _available(funds, CAT_FEATURES)
    data = funds.dropna(subset=[TARGET])
    X, y = data[num_cols + cat_cols], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
    }
    rows, fitted = [], {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, est in candidates.items():
        pipe = build_pipeline(est, num_cols, cat_cols)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"model": name,
                     "cv_r2_mean": round(cv_r2.mean(), 3),
                     "test_r2": round(r2_score(y_test, pred), 3),
                     "test_mae_%": round(mean_absolute_error(y_test, pred), 2)})
        fitted[name] = pipe

    results = pd.DataFrame(rows)
    best_name = results.sort_values("test_r2").iloc[-1]["model"]
    best = fitted[best_name]
    pred_test = best.predict(X_test)                    # saved for the actual-vs-predicted plot
    best.fit(X, y)                                      # refit on all data

    importance = None
    if best_name == "Random Forest":
        names = best.named_steps["prep"].get_feature_names_out()
        importance = (pd.Series(best.named_steps["model"].feature_importances_, index=names)
                      .sort_values(ascending=False))
        importance.index = [i.split("__", 1)[-1] for i in importance.index]

    return {"results": results, "best_model_name": best_name, "model": best,
            "features": num_cols + cat_cols, "feature_importance": importance,
            "y_test": y_test, "pred_test": pred_test}


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(0.5, index=s.index)


def score_funds(funds: pd.DataFrame, model_bundle) -> pd.DataFrame:
    """
    Fund score (0-100) = weighted mix of normalised metrics:
      35% predicted 3yr return (ML)   25% Sharpe   15% Sortino
      15% low expense ratio           10% rating
    Weights are a design choice (explain in report); edit them freely.
    """
    df = funds.copy()
    df["pred_return_3yr"] = model_bundle["model"].predict(df[model_bundle["features"]])

    parts = {"pred_return_3yr": 0.35, "sharpe": 0.25, "sortino": 0.15, "rating": 0.10}
    score = pd.Series(0.0, index=df.index)
    total_w = 0
    for col, w in parts.items():
        if col in df.columns:
            score += w * _minmax(df[col].fillna(df[col].median()))
            total_w += w
    if "expense_ratio" in df.columns:
        score += 0.15 * (1 - _minmax(df["expense_ratio"].fillna(df["expense_ratio"].median())))
        total_w += 0.15
    df["fund_score"] = (score / total_w * 100).round(1)
    return df.sort_values("fund_score", ascending=False)
