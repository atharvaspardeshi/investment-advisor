"""
Web app:   streamlit run app.py
"""
import glob
import os
import shutil

import pandas as pd
import streamlit as st

from src.advisor import Investor, project_wealth, recommend
from src.data_loader import load_mutual_funds, load_nse_indices, pick_benchmark
from src.fund_model import score_funds, train_fund_model
from src.market_model import index_stats, outlook_label, train_market_model

st.set_page_config(page_title="Investment Advisor", page_icon="📈", layout="wide")


DATASETS = {"data/nse": "vinay20/stock-market-nse-indices-daily-data2000",
            "data/mutual_funds": "ravibarnawal/mutual-funds-india-detailed"}


def ensure_data():
    """On a fresh server (e.g. Streamlit Cloud) data/ is empty, so download it from Kaggle once."""
    for folder, slug in DATASETS.items():
        if not glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
            import kagglehub
            path = kagglehub.dataset_download(slug)
            os.makedirs(folder, exist_ok=True)
            for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
                shutil.copy(f, folder)


@st.cache_resource(show_spinner="Downloading data and training models (first load takes ~1 min)...")
def load_everything(nse_dir="data/nse", mf_path="data/mutual_funds"):
    ensure_data()
    prices = load_nse_indices(nse_dir)
    bench = pick_benchmark(prices)
    mkt = train_market_model(prices[bench])
    funds = load_mutual_funds(mf_path)
    fm = train_fund_model(funds)
    return prices, bench, index_stats(prices), mkt, fm, score_funds(funds, fm)


prices, bench, stats, mkt, fm, scored = load_everything()
outlook = outlook_label(mkt["prob_up"])

st.title("📈 ML-based Investment Advisor")
st.caption("NSE index data + Indian mutual funds | For academic purposes only, not SEBI-registered advice.")

tab_adv, tab_mkt, tab_funds, tab_models = st.tabs(["Get advice", "Market", "Fund explorer", "Model performance"])

# ---------------- Advice ----------------
with tab_adv:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Your profile")
        age = st.slider("Age", 18, 75, 22)
        horizon = st.slider("Investment horizon (years)", 1, 30, 10)
        sip = st.number_input("Monthly SIP (₹)", 0, 1_000_000, 5000, step=500)
        lump = st.number_input("One-time lumpsum (₹)", 0, 100_000_000, 0, step=5000)
        income = st.radio("Income", ["Unstable", "Stable", "Very stable"], index=1, horizontal=True)
        fall = st.radio("If your portfolio falls 20% in a month, you would...",
                        ["Sell everything", "Sell some", "Hold", "Buy more"], index=2)
        exp = st.radio("Investing experience", ["None", "Some", "Experienced"], index=1, horizontal=True)
        go = st.button("Get recommendation", type="primary", use_container_width=True)

    with right:
        if go:
            if sip == 0 and lump == 0:
                st.warning("Enter a SIP or lumpsum amount.")
                st.stop()
            inv = Investor(age=age, horizon_years=horizon, monthly_sip=sip, lumpsum=lump,
                           income_stability=["Unstable", "Stable", "Very stable"].index(income) + 1,
                           reaction_to_fall=["Sell everything", "Sell some", "Hold", "Buy more"].index(fall),
                           experience=["None", "Some", "Experienced"].index(exp))
            rec = recommend(scored, inv, outlook)
            prof = rec["profile"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk profile", prof["profile"])
            c2.metric("Risk score", f"{prof['score']} / 16")
            c3.metric("Market outlook", outlook, f"P(up) = {mkt['prob_up']:.0%}", delta_color="off")

            st.subheader("Asset allocation")
            alloc = pd.DataFrame({"weight_%": {k: v * 100 for k, v in rec["allocation"].items()}})
            st.bar_chart(alloc)

            st.subheader("Recommended funds")
            if rec["picks"].empty:
                st.info("No funds match these amounts. Try a higher SIP/lumpsum.")
            else:
                st.dataframe(rec["picks"], use_container_width=True, hide_index=True)
                proj = project_wealth(rec["picks"], inv)
                st.subheader("Wealth projection")
                p1, p2, p3 = st.columns(3)
                p1.metric("Expected return", f"{proj['expected_return_%']}% p.a.")
                p2.metric("Total invested", f"₹{proj['invested_₹']:,}")
                p3.metric(f"Value after {horizon} yrs", f"₹{proj['future_value_₹']:,}")
                st.line_chart(proj["yearly"].set_index("year"))
        else:
            st.info("Fill your profile on the left and click **Get recommendation**.")

# ---------------- Market ----------------
with tab_mkt:
    st.subheader(f"{bench} closing value")
    st.line_chart(prices[bench].dropna())
    st.subheader("Risk & return of NSE indices")
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.scatter_chart(stats, x="volatility_%", y="cagr_%", color="index")

# ---------------- Fund explorer ----------------
with tab_funds:
    cats = sorted(scored["category"].dropna().unique()) if "category" in scored else []
    chosen = st.multiselect("Category", cats, default=cats)
    view = scored[scored["category"].isin(chosen)] if cats else scored
    cols = [c for c in ["scheme_name", "category", "sub_category", "risk_level", "fund_score",
                        "returns_1yr", "returns_3yr", "pred_return_3yr", "sharpe",
                        "expense_ratio", "min_sip"] if c in view]
    st.dataframe(view[cols].head(200), use_container_width=True, hide_index=True)

# ---------------- Models ----------------
with tab_models:
    st.subheader(f"Market direction model (next 21 trading days, {bench})")
    st.dataframe(mkt["results"], hide_index=True)
    st.caption(f"Test period: {mkt['test_period'][0]} to {mkt['test_period'][1]} | best: {mkt['best_model_name']}")
    st.subheader("Fund 3-year return model")
    st.dataframe(fm["results"], hide_index=True)
    if fm["feature_importance"] is not None:
        st.bar_chart(fm["feature_importance"].head(12))
