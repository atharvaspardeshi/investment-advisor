"""
Web app:   streamlit run app.py
"""
import glob
import os
import shutil

import pandas as pd
import streamlit as st

from src import calculator as calc
from src.advisor import Investor, explain_advice, inr, project_wealth, recommend
from src.chatbot import KnowledgeBase, answer
from src.data_loader import load_mutual_funds, load_nse_indices, pick_benchmark
from src.fund_model import score_funds, train_fund_model
from src.goal_planner import Goal, plan_goal
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


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_kb():
    return KnowledgeBase("data/knowledge", "data/docs")


def secret_key():
    """Gemini key from Streamlit secrets or an environment variable (never hard-code keys in code)."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


prices, bench, stats, mkt, fm, scored = load_everything()
kb = load_kb()
outlook = outlook_label(mkt["prob_up"])

st.title("📈 ML-based Investment Advisor")
st.caption("NSE index data + Indian mutual funds | For academic purposes only, not SEBI-registered advice.")

(tab_adv, tab_calc, tab_goal, tab_chat,
 tab_mkt, tab_funds, tab_models) = st.tabs(["🎯 Get advice", "🧮 Calculators", "🏁 Goal planner", "🤖 AI assistant",
                                             "📊 Market", "🔍 Fund explorer", "📈 Model performance"])

# ---------------- Advice ----------------
with tab_adv:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Your profile")
        age = st.slider("Age", 18, 75, 22)
        horizon = st.slider("Investment horizon (years)", 1, 30, 10)
        sip_amt = st.number_input("Monthly SIP (₹)", 0, 1_000_000, 5000, step=500)
        lump = st.number_input("One-time lumpsum (₹)", 0, 100_000_000, 0, step=5000)
        income = st.radio("Income", ["Unstable", "Stable", "Very stable"], index=1, horizontal=True)
        fall = st.radio("If your portfolio falls 20% in a month, you would...",
                        ["Sell everything", "Sell some", "Hold", "Buy more"], index=2)
        exp = st.radio("Investing experience", ["None", "Some", "Experienced"], index=1, horizontal=True)
        go = st.button("Get recommendation", type="primary", use_container_width=True)

    with right:
        if go:
            if sip_amt == 0 and lump == 0:
                st.warning("Enter a SIP or lumpsum amount.")
                st.stop()
            inv = Investor(age=age, horizon_years=horizon, monthly_sip=sip_amt, lumpsum=lump,
                           income_stability=["Unstable", "Stable", "Very stable"].index(income) + 1,
                           reaction_to_fall=["Sell everything", "Sell some", "Hold", "Buy more"].index(fall),
                           experience=["None", "Some", "Experienced"].index(exp))
            rec = recommend(scored, inv, outlook)
            prof = rec["profile"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk profile", prof["profile"])
            c2.metric("Risk score", f"{prof['score']} / 16")
            c3.metric("Market outlook", outlook, f"P(up) = {mkt['prob_up']:.0%}", delta_color="off")

            proj = project_wealth(rec["picks"], inv) if not rec["picks"].empty else None
            # remembered so the AI assistant can answer questions about this plan
            st.session_state["profile_note"] = (
                f"{prof['profile']} investor, age {age}, {horizon}-year horizon, SIP ₹{sip_amt}/month, "
                f"lumpsum ₹{lump}; allocation {rec['allocation']}; funds: "
                + ", ".join(rec["picks"]["scheme_name"].tolist() if not rec["picks"].empty else []))

            st.subheader("📝 Your advice in simple words")
            for i, (heading, text) in enumerate(explain_advice(rec, proj, inv, outlook)):
                if heading == "Important":
                    st.caption(text)
                else:
                    with st.expander(heading, expanded=i < 2):
                        st.markdown(text)

            st.subheader("Asset allocation")
            alloc = pd.DataFrame({"weight_%": {k: v * 100 for k, v in rec["allocation"].items()}})
            st.bar_chart(alloc)

            st.subheader("Recommended funds")
            if rec["picks"].empty:
                st.info("No funds match these amounts. Try a higher SIP/lumpsum.")
            else:
                st.dataframe(rec["picks"], use_container_width=True, hide_index=True)
                st.subheader("Wealth projection")
                p1, p2, p3 = st.columns(3)
                p1.metric("Expected return", f"{proj['expected_return_%']}% p.a.")
                p2.metric("Total invested", f"₹{proj['invested_₹']:,}")
                p3.metric(f"Value after {horizon} yrs", f"₹{proj['future_value_₹']:,}")
                st.line_chart(proj["yearly"].set_index("year"))
        else:
            st.info("Fill your profile on the left and click **Get recommendation**.")

# ---------------- Calculators ----------------
with tab_calc:
    kind = st.radio("Calculator", ["SIP", "Lumpsum", "SWP (monthly withdrawal)"], horizontal=True)
    c_in, c_out = st.columns([1, 2])
    with c_in:
        if kind == "SIP":
            m = st.number_input("Monthly investment (₹)", 100, 10_000_000, 5000, step=500, key="c_sip")
            step = st.slider("Increase SIP every year by (%)", 0, 25, 0, key="c_step",
                             help="Step-up SIP: raise your SIP each year as your income grows")
        elif kind == "Lumpsum":
            m = st.number_input("One-time investment (₹)", 1000, 1_000_000_000, 100000, step=10000, key="c_lump")
        else:
            m = st.number_input("Starting amount / corpus (₹)", 10000, 1_000_000_000, 5_000_000, step=100000, key="c_corpus")
            wd = st.number_input("Withdraw every month (₹)", 500, 10_000_000, 30000, step=1000, key="c_wd")
        rate = st.slider("Expected return (% per year)", 1.0, 20.0, 12.0 if kind != "SWP (monthly withdrawal)" else 8.0,
                         0.5, key="c_rate")
        yrs = st.slider("Time period (years)", 1, 40, 10, key="c_yrs")
    with c_out:
        if kind == "SIP":
            df = calc.sip(m, rate, yrs, step)
            last = df.iloc[-1]
            a, b, c = st.columns(3)
            a.metric("You invest", inr(last["invested"]))
            b.metric("Gains", inr(last["value"] - last["invested"]))
            c.metric("Total value", inr(last["value"]))
            st.line_chart(df.set_index("year")[["invested", "value"]])
            st.markdown(f"Investing **{inr(m)}/month**" + (f", increased by {step}% every year," if step else "")
                        + f" for **{yrs} years** at **{rate}%** could grow to about **{inr(last['value'])}**. "
                          f"Of this, **{inr(last['value'] - last['invested'])}** is growth from compounding.")
        elif kind == "Lumpsum":
            df = calc.lumpsum(m, rate, yrs)
            last = df.iloc[-1]
            a, b, c = st.columns(3)
            a.metric("You invest", inr(m))
            b.metric("Gains", inr(last["value"] - m))
            c.metric("Total value", inr(last["value"]))
            st.line_chart(df.set_index("year")[["invested", "value"]])
            st.markdown(f"**{inr(m)}** invested once at **{rate}%** for **{yrs} years** could become "
                        f"**{inr(last['value'])}**, about **{last['value'] / m:.1f}x** your money.")
        else:
            df = calc.swp(m, wd, rate, yrs)
            last = df.iloc[-1]
            a, b, c = st.columns(3)
            a.metric("Total withdrawn", inr(last["withdrawn"]))
            b.metric("Balance left", inr(last["balance"]))
            years_lasted = int((df["balance"] > 0).sum() - 1)
            c.metric("Money lasts", f"{years_lasted}+ yrs" if last["balance"] > 0 else f"~{years_lasted + 1} yrs")
            st.line_chart(df.set_index("year")[["balance"]])
            if last["balance"] > 0:
                st.success(f"Withdrawing {inr(wd)}/month, your money still has {inr(last['balance'])} left after {yrs} years.")
            else:
                st.warning(f"At {inr(wd)}/month your money runs out in about {years_lasted + 1} years. "
                           "Withdraw less or start with a bigger amount.")
    st.caption("Monthly compounding with the effective monthly rate (more accurate than dividing yearly return by 12). "
               "Returns are assumptions, not guarantees.")

# ---------------- Goal planner ----------------
with tab_goal:
    st.subheader("Plan your goals")
    st.write("Add your goals below (edit the table directly). The planner adjusts for inflation, chooses a safe "
             "mix for each goal's time-frame, and tells you how much to invest every month.")
    default_goals = pd.DataFrame([
        {"Goal": "Emergency fund", "Cost today (₹)": 300000, "Years away": 1, "Already saved (₹)": 50000},
        {"Goal": "Bike / car", "Cost today (₹)": 800000, "Years away": 5, "Already saved (₹)": 0},
        {"Goal": "Higher studies", "Cost today (₹)": 2000000, "Years away": 4, "Already saved (₹)": 100000},
        {"Goal": "Retirement", "Cost today (₹)": 30000000, "Years away": 30, "Already saved (₹)": 0},
    ])
    goals_df = st.data_editor(default_goals, num_rows="dynamic", use_container_width=True, key="goals",
                              column_config={"Years away": st.column_config.NumberColumn(min_value=1, max_value=50, step=1)})
    g1, g2 = st.columns(2)
    infl = g1.slider("Inflation (% per year)", 0.0, 12.0, 6.0, 0.5, help="Long-run Indian inflation is about 5-6%")
    gstep = g2.slider("Increase SIPs every year by (%)", 0, 20, 0, help="Step-up: makes today's SIP smaller")

    plans = []
    for _, r in goals_df.dropna(subset=["Goal", "Cost today (₹)", "Years away"]).iterrows():
        if float(r["Cost today (₹)"]) <= 0 or int(r["Years away"]) <= 0:
            continue
        plans.append(plan_goal(Goal(str(r["Goal"]), float(r["Cost today (₹)"]), int(r["Years away"]),
                                    float(r.get("Already saved (₹)") or 0), infl), gstep))
    if plans:
        total = sum(p["monthly_sip"] for p in plans)
        st.metric("Total monthly investment needed", inr(total) + "/month")
        table = pd.DataFrame([{
            "Goal": p["goal"], "Years": p["years"], "Cost today": inr(p["cost_today"]),
            "Cost then (with inflation)": inr(p["future_cost"]), "Monthly SIP needed": inr(p["monthly_sip"]),
            "Mix (Equity/Hybrid/Debt)": " / ".join(f"{int(v * 100)}%" for v in p["mix"].values()),
            "Expected return": f"{p['expected_return_%']}%"} for p in plans])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"Monthly SIP (₹)": {p["goal"]: p["monthly_sip"] for p in plans}}))
        st.markdown("#### In simple words")
        for p in plans:
            where = ("mostly safe debt funds, because you need it soon" if p["mix"]["Equity"] == 0 else
                     "a balanced mix, since it is a few years away" if p["mix"]["Equity"] < 0.5 else
                     "mostly equity funds for growth, since you have time to ride out ups and downs")
            if p["monthly_sip"] == 0:
                st.markdown(f"- **{p['goal']}**: your existing savings should already cover this. 🎉")
            else:
                st.markdown(f"- **{p['goal']}** costs {inr(p['cost_today'])} today but about **{inr(p['future_cost'])}** "
                            f"in {p['years']} years. Invest **{inr(p['monthly_sip'])}/month** in {where}.")
        st.caption("Assumed returns: equity 12%, hybrid 9.5%, debt 7% per year. Use the Get advice tab to pick funds for each goal.")

# ---------------- AI assistant ----------------
with tab_chat:
    st.subheader("🤖 Ask anything about mutual funds")
    key = secret_key()
    with st.expander("⚙️ AI settings", expanded=False):
        user_key = st.text_input("Gemini API key (optional)", type="password",
                                 help="Free key from aistudio.google.com. Stays only in this browser session.")
        if user_key:
            key = user_key
        st.caption(f"Knowledge base: {len(kb.chunks)} topics/passages. Put extra PDFs (e.g. RBI or AMFI guides) in "
                   "data/docs/ to add them.")
    st.caption("✨ AI mode: Gemini writes answers from our knowledge base and fund data." if key else
               "🔎 Search mode: answers come straight from our knowledge base. Add a Gemini key for full AI answers.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    examples = ["What is SIP?", "Direct vs Regular plan?", "Best small cap fund?", "How are mutual funds taxed?",
                "Should I stop SIP when market falls?"]
    cols = st.columns(len(examples))
    clicked = None
    for c, q in zip(cols, examples):
        if c.button(q, use_container_width=True):
            clicked = q

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("funds") is not None:
                st.dataframe(msg["funds"], hide_index=True, use_container_width=True)

    question = st.chat_input("Type your question (English or Hinglish)...") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = answer(question, kb, scored, st.session_state.chat, key,
                             st.session_state.get("profile_note"))
            st.markdown(res["text"])
            if res["funds"] is not None and not res["funds"].empty:
                st.dataframe(res["funds"], hide_index=True, use_container_width=True)
            if res["sources"]:
                st.caption("Sources: " + "; ".join(res["sources"][:3]))
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": res["text"], "funds": res["funds"]})
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

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
