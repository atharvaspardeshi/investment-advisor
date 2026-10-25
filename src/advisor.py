"""
Part C — THE ADVISOR: risk profiling -> asset allocation -> fund picks -> projection.

Risk profiling is a scored questionnaire (the same idea SEBI-registered advisors
and AMCs use). Why rules and not ML here? We have no labelled data of
"investor -> correct risk profile", so a transparent scoring rule is the
standard, explainable approach.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROFILES = [
    # name,                    min score, equity, hybrid, debt, max fund risk_level (1-6)
    ("Conservative",              0,      0.10,   0.20,  0.70, 2),
    ("Moderately Conservative",   5,      0.30,   0.25,  0.45, 3),
    ("Moderate",                  8,      0.50,   0.20,  0.30, 4),
    ("Moderately Aggressive",    11,      0.65,   0.15,  0.20, 5),
    ("Aggressive",               14,      0.80,   0.10,  0.10, 6),
]
RISK_NAMES = {1: "Low", 2: "Low to Moderate", 3: "Moderate",
              4: "Moderately High", 5: "High", 6: "Very High"}


@dataclass
class Investor:
    age: int
    horizon_years: int
    monthly_sip: float = 0            # ₹ per month
    lumpsum: float = 0                # ₹ one time
    income_stability: int = 2         # 1 = unstable, 2 = stable, 3 = very stable
    reaction_to_fall: int = 2         # 0 sell all, 1 sell some, 2 hold, 3 buy more
    experience: int = 1               # 0 none, 1 some, 2 experienced


def risk_score(inv: Investor) -> int:
    age_pts = 4 if inv.age < 30 else 3 if inv.age < 45 else 2 if inv.age < 60 else 1
    hz = inv.horizon_years
    hz_pts = 0 if hz < 1 else 1 if hz < 3 else 2 if hz < 5 else 3 if hz < 10 else 4
    return age_pts + hz_pts + inv.income_stability + inv.reaction_to_fall + inv.experience  # 1..16


def risk_profile(inv: Investor):
    score = risk_score(inv)
    profile = [p for p in PROFILES if score >= p[1]][-1]
    if inv.horizon_years < 3:            # short goals should never be equity-heavy
        profile = PROFILES[min(PROFILES.index(profile), 1)]
    name, _, eq, hy, de, max_risk = profile
    return {"score": score, "profile": name,
            "allocation": {"Equity": eq, "Hybrid": hy, "Debt": de},
            "max_risk_level": max_risk}


def adjust_for_market(allocation: dict, outlook: str) -> dict:
    """
    Small tactical tilt from the ML market outlook (max 10%).
    Why only a small tilt? Market-direction models are weak (accuracy ~55%);
    betting the whole portfolio on them is bad advice.
    """
    alloc = dict(allocation)
    shift = 0.10 if outlook == "Bearish" else -0.05 if outlook == "Bullish" else 0.0
    shift = min(shift, alloc["Equity"]) if shift > 0 else max(shift, -alloc["Debt"])
    alloc["Equity"] -= shift
    alloc["Debt"] += shift
    return {k: round(v, 2) for k, v in alloc.items()}


def _bucket(category: str) -> str:
    c = str(category).lower()
    if "equity" in c or "elss" in c:
        return "Equity"
    if "debt" in c or "liquid" in c or "gilt" in c:
        return "Debt"
    if "hybrid" in c or "balanced" in c:
        return "Hybrid"
    return "Other"


def recommend(scored_funds: pd.DataFrame, inv: Investor, outlook: str = "Neutral",
              funds_per_bucket: int = 2):
    prof = risk_profile(inv)
    alloc = adjust_for_market(prof["allocation"], outlook)

    df = scored_funds.copy()
    df["bucket"] = df["category"].map(_bucket) if "category" in df else "Equity"
    if "risk_level" in df:
        df = df[df["risk_level"].fillna(6) <= prof["max_risk_level"]]

    def select(pool, sip_amt, lump_amt):
        """Largest k (<= funds_per_bucket) such that each fund's minimum fits its share."""
        for k in range(funds_per_bucket, 0, -1):
            ok = pool
            if sip_amt > 0 and "min_sip" in pool:
                ok = ok[ok["min_sip"].fillna(0) <= sip_amt / k]
            if lump_amt > 0 and "min_lumpsum" in pool:
                ok = ok[ok["min_lumpsum"].fillna(0) <= lump_amt / k]
            if len(ok) >= k:
                return ok.head(k), k
        return pool.head(0), 0

    # If no eligible fund exists for a bucket (e.g. no low-risk equity fund for a
    # conservative investor), move its weight to the next safer bucket instead of
    # silently dropping that money.
    fallback = {"Equity": ["Hybrid", "Debt"], "Hybrid": ["Debt", "Equity"], "Debt": ["Hybrid", "Equity"]}
    fillable = {b: select(df[df["bucket"] == b], inv.monthly_sip * w, inv.lumpsum * w)[1] > 0
                for b, w in alloc.items() if w > 0}
    for b in list(alloc):
        if alloc[b] > 0 and not fillable.get(b):
            target = next((t for t in fallback[b] if fillable.get(t)), None)
            if target:
                alloc[target] = round(alloc[target] + alloc[b], 2)
                alloc[b] = 0.0

    picks = []
    for bucket, weight in alloc.items():
        if weight <= 0:
            continue
        sip_amt, lump_amt = inv.monthly_sip * weight, inv.lumpsum * weight
        top, k = select(df[df["bucket"] == bucket], sip_amt, lump_amt)
        for _, f in top.iterrows():
            picks.append({
                "bucket": bucket,
                "scheme_name": f.get("scheme_name", "-"),
                "sub_category": f.get("sub_category", "-"),
                "risk": RISK_NAMES.get(int(f["risk_level"]), "-") if pd.notna(f.get("risk_level")) else "-",
                "fund_score": f["fund_score"],
                "returns_3yr_%": f.get("returns_3yr"),
                "pred_return_3yr_%": round(f["pred_return_3yr"], 2),
                "expense_ratio_%": f.get("expense_ratio"),
                "monthly_sip_₹": round(sip_amt / k),
                "lumpsum_₹": round(lump_amt / k),
            })
    picks = pd.DataFrame(picks)
    return {"profile": prof, "allocation": alloc, "picks": picks}


def project_wealth(picks: pd.DataFrame, inv: Investor, cap: float = 15.0):
    """
    Future value with the portfolio's expected return.
    We cap the expected return at 15% because past 3-yr returns (esp. after a
    bull run) are usually higher than what is realistic long term.
    """
    if picks.empty:
        return None
    w = picks["monthly_sip_₹"] + picks["lumpsum_₹"]
    exp_ret = np.average(picks["pred_return_3yr_%"].clip(upper=cap), weights=w if w.sum() else None)
    exp_ret = float(np.clip(exp_ret, 4, cap))
    n_years = inv.horizon_years
    r_m = (1 + exp_ret / 100) ** (1 / 12) - 1
    months = n_years * 12

    fv_sip = inv.monthly_sip * (((1 + r_m) ** months - 1) / r_m) * (1 + r_m) if r_m else inv.monthly_sip * months
    fv_lump = inv.lumpsum * (1 + exp_ret / 100) ** n_years
    invested = inv.monthly_sip * months + inv.lumpsum

    yearly = []
    for y in range(n_years + 1):
        m = y * 12
        v = inv.monthly_sip * (((1 + r_m) ** m - 1) / r_m) * (1 + r_m) if m else 0
        v += inv.lumpsum * (1 + exp_ret / 100) ** y
        yearly.append({"year": y, "invested_₹": inv.monthly_sip * m + inv.lumpsum, "value_₹": round(v)})

    return {"expected_return_%": round(exp_ret, 2), "invested_₹": round(invested),
            "future_value_₹": round(fv_sip + fv_lump), "yearly": pd.DataFrame(yearly)}


# ----------------------------------------------------------------------------
# Plain-language advice (for people with no finance background)
# ----------------------------------------------------------------------------
def inr(x: float) -> str:
    """Indian style amounts: ₹4,500 / ₹3.6 lakh / ₹1.18 crore."""
    x = float(x)
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} crore"
    if x >= 1e5:
        return f"₹{x / 1e5:.1f} lakh"
    return f"₹{x:,.0f}"


PROFILE_TEXT = {
    "Conservative": "You prefer safety over high returns. Your plan focuses on protecting your money, so it stays mostly in low-risk options.",
    "Moderately Conservative": "You want steady growth but don't like big ups and downs. Most of your money goes into safer options, with some in the stock market for growth.",
    "Moderate": "You want a balance of growth and safety. Your money is split fairly evenly between the stock market and safer options.",
    "Moderately Aggressive": "You are okay with ups and downs for better long-term growth. Most of your money goes into the stock market.",
    "Aggressive": "You can handle big short-term falls to aim for the highest long-term growth. Almost all your money goes into the stock market.",
}
BUCKET_TEXT = {
    "Equity": "stock-market funds. They grow the most over 5+ years, but can fall 20-40% in a bad year",
    "Hybrid": "mixed funds (stocks + bonds). Smoother ride than pure equity, moderate growth",
    "Debt": "bond / money-market funds. Low risk and steady, similar to a fixed deposit (FD)",
}
OUTLOOK_TEXT = {
    "Bullish": "Our market model sees slightly positive signals for the next month, so we added a little (5%) more to stock-market funds.",
    "Bearish": "Our market model sees weak signals for the next month, so we moved 10% from stock-market funds to safer debt funds.",
    "Neutral": "Our market model sees mixed signals for the next month, so we kept your normal plan.",
}


def explain_advice(rec: dict, proj, inv: Investor, outlook: str = "Neutral") -> list:
    """Returns a list of (heading, markdown text) sections in simple language."""
    prof, alloc, picks = rec["profile"], rec["allocation"], rec["picks"]
    sections = [("Your investor type: " + prof["profile"], PROFILE_TEXT.get(prof["profile"], ""))]

    # where the money goes, in rupees
    lines = []
    for bucket, w in alloc.items():
        if w <= 0:
            continue
        parts = []
        if inv.monthly_sip:
            parts.append(f"{inr(inv.monthly_sip * w)}/month")
        if inv.lumpsum:
            parts.append(f"{inr(inv.lumpsum * w)} one-time")
        lines.append(f"- **{' + '.join(parts)} ({w:.0%}) in {bucket}**: {BUCKET_TEXT[bucket]}.")
    sections.append(("Where your money goes", "\n".join(lines)))

    signal = OUTLOOK_TEXT.get(outlook, OUTLOOK_TEXT["Neutral"])
    if outlook in ("Bullish", "Bearish"):
        signal += " This is only a small adjustment, because nobody can reliably predict the market month to month."
    else:
        signal += " Trying to time the market rarely works; investing regularly matters more."
    sections.append(("What the market signal means", signal))

    # why these funds
    if not picks.empty:
        lines = []
        for _, f in picks.iterrows():
            why = []
            if pd.notna(f.get("returns_3yr_%")):
                why.append(f"earned about {f['returns_3yr_%']:.0f}% per year over the last 3 years")
            if pd.notna(f.get("expense_ratio_%")):
                cost = "low" if f["expense_ratio_%"] <= 0.75 else "moderate" if f["expense_ratio_%"] <= 1.5 else "high"
                why.append(f"{cost} yearly fee ({f['expense_ratio_%']:.2f}%)")
            why.append(f"risk: {f['risk']}")
            lines.append(f"- **{f['scheme_name']}**: " + ", ".join(why) + ".")
        sections.append(("Why these funds",
                         "Each fund was picked for strong past returns, good returns for the risk taken, and low fees.\n\n" + "\n".join(lines)))

    # what to expect
    if proj:
        txt = (f"If you stay invested for **{inv.horizon_years} years** and returns average about "
               f"**{proj['expected_return_%']:.0f}% per year**, your {inr(proj['invested_₹'])} could grow to roughly "
               f"**{inr(proj['future_value_₹'])}**. ")
        if alloc.get("Equity", 0) >= 0.5:
            txt += ("Expect some years where your portfolio shows a loss; that's normal for stock-market funds "
                    "and usually recovers if you stay invested. ")
        txt += "This is an estimate based on past data, not a promise."
        sections.append(("What you can expect", txt))

    # practical tips
    tips = ["Keep 3-6 months of expenses in a savings account or FD as an **emergency fund** before investing.",
            "Choose the **Direct plan** of each fund (not Regular); it has lower fees.",
            "**Review your plan once a year**, or when your income or goals change."]
    if inv.monthly_sip:
        tips.insert(1, "**Don't stop your SIP when markets fall.** You buy more units cheaply, which helps long-term returns.")
    else:
        tips.insert(1, "**Don't panic-sell when markets fall.** Short-term drops usually recover if you stay invested.")
    if inv.horizon_years < 3:
        tips.insert(0, "Your goal is less than 3 years away, so your plan avoids putting much in stock-market funds. "
                       "Short-term money should stay safe.")
    if alloc.get("Equity", 0) > 0:
        tips.append("**ELSS** funds (a type of equity fund) can also save income tax under Section 80C, with a 3-year lock-in.")
    sections.append(("Simple tips", "\n".join(f"- {t}" for t in tips)))
    sections.append(("Important", "This is a college project using past data, not professional financial advice. "
                                  "Talk to a SEBI-registered advisor before investing real money."))
    return sections
