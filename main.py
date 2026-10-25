"""
Run the full pipeline from the command line:

    python main.py

Reads data/  ->  trains both models  ->  saves charts, tables and models to outputs/
->  prints a sample recommendation.
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.advisor import Investor, project_wealth, recommend
from src.data_loader import load_mutual_funds, load_nse_indices, pick_benchmark
from src.fund_model import score_funds, train_fund_model
from src.market_model import index_stats, outlook_label, train_market_model

OUT = "outputs"


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=130)
    plt.close()


def main(nse_dir, mf_path, benchmark=None):
    os.makedirs(OUT, exist_ok=True)

    # ---------------- Part A: market ----------------
    prices = load_nse_indices(nse_dir)
    print(f"Loaded {prices.shape[1]} indices, {prices.index.min().date()} -> {prices.index.max().date()}")
    stats = index_stats(prices)
    stats.to_csv(f"{OUT}/index_stats.csv", index=False)
    print("\nTop indices by Sharpe ratio:\n", stats.head(10).to_string(index=False))

    print("Indices found:", ", ".join(prices.columns))
    bench = benchmark or pick_benchmark(prices)
    print("Benchmark used:", bench)
    (prices[bench].dropna()).plot(figsize=(10, 4), title=f"{bench} - closing value")
    save_fig("benchmark_price.png")

    top = stats.head(15)
    plt.figure(figsize=(8, 6))
    plt.scatter(top["volatility_%"], top["cagr_%"])
    for _, r in top.iterrows():
        plt.annotate(r["index"], (r["volatility_%"], r["cagr_%"]), fontsize=7)
    plt.xlabel("Volatility (%)"); plt.ylabel("CAGR (%)"); plt.title("Risk vs Return of NSE indices")
    save_fig("risk_return_indices.png")

    mkt = train_market_model(prices[bench])
    mkt["results"].to_csv(f"{OUT}/market_model_results.csv", index=False)
    outlook = outlook_label(mkt["prob_up"])
    print(f"\nMarket model ({bench}, test {mkt['test_period'][0]} -> {mkt['test_period'][1]}):")
    print(mkt["results"].to_string(index=False))
    print(f"Best: {mkt['best_model_name']} | P(UP next month) = {mkt['prob_up']:.2f} -> {outlook}")
    if mkt["feature_importance"] is not None:
        mkt["feature_importance"].plot.barh(figsize=(7, 4), title="Market model - feature importance")
        save_fig("market_feature_importance.png")
    joblib.dump(mkt["model"], f"{OUT}/market_model.joblib")

    # ---------------- Part B: funds ----------------
    funds = load_mutual_funds(mf_path)
    print(f"\nLoaded {len(funds)} mutual funds")
    fm = train_fund_model(funds)
    fm["results"].to_csv(f"{OUT}/fund_model_results.csv", index=False)
    print(fm["results"].to_string(index=False))

    plt.figure(figsize=(5, 5))
    plt.scatter(fm["y_test"], fm["pred_test"], alpha=0.5)
    lims = [min(fm["y_test"].min(), fm["pred_test"].min()), max(fm["y_test"].max(), fm["pred_test"].max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual 3yr return (%)"); plt.ylabel("Predicted (%)")
    plt.title(f"{fm['best_model_name']} - actual vs predicted")
    save_fig("fund_actual_vs_predicted.png")
    if fm["feature_importance"] is not None:
        fm["feature_importance"].head(12).plot.barh(figsize=(7, 5), title="Fund model - feature importance")
        save_fig("fund_feature_importance.png")
    joblib.dump(fm, f"{OUT}/fund_model.joblib")

    scored = score_funds(funds, fm)
    scored.to_csv(f"{OUT}/scored_funds.csv", index=False)

    # ---------------- Part C: sample advice ----------------
    inv = Investor(age=21, horizon_years=10, monthly_sip=5000, lumpsum=0,
                   income_stability=2, reaction_to_fall=2, experience=1)
    rec = recommend(scored, inv, outlook)
    proj = project_wealth(rec["picks"], inv)
    print(f"\nSample investor -> {rec['profile']['profile']} (score {rec['profile']['score']}/16)")
    print("Allocation:", rec["allocation"])
    print(rec["picks"].to_string(index=False))
    if proj:
        print(f"Expected return {proj['expected_return_%']}% | invested ₹{proj['invested_₹']:,} "
              f"-> ₹{proj['future_value_₹']:,} in {inv.horizon_years} yrs")
    print(f"\nAll outputs saved in ./{OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nse", default="data/nse", help="folder with NSE index CSVs")
    p.add_argument("--mf", default="data/mutual_funds", help="mutual fund CSV or folder")
    p.add_argument("--benchmark", default=None, help="index name to use as market benchmark")
    a = p.parse_args()
    main(a.nse, a.mf, a.benchmark)
