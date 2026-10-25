"""
Goal planner: "I need ₹X in Y years" -> how much SIP per month, and which mix of funds.

Steps for each goal:
1. Inflate today's cost to the future cost (a ₹10 lakh car today costs more in 5 years).
2. Pick an asset mix by time horizon: short goals stay safe, long goals can use equity.
3. Expected return of that mix -> required monthly SIP (calculator.required_sip).

Why the mix depends on horizon: equity can fall 30-40% in a bad year and may need
3-5 years to recover, so money needed soon should not be in equity.
"""
from dataclasses import dataclass

import pandas as pd

from .calculator import inflate, required_sip

# conservative long-run return assumptions (% per year) per asset class
EXPECTED_RETURN = {"Equity": 12.0, "Hybrid": 9.5, "Debt": 7.0}


def mix_for_horizon(years: int) -> dict:
    if years < 3:
        return {"Equity": 0.0, "Hybrid": 0.2, "Debt": 0.8}
    if years < 5:
        return {"Equity": 0.3, "Hybrid": 0.3, "Debt": 0.4}
    if years < 8:
        return {"Equity": 0.6, "Hybrid": 0.2, "Debt": 0.2}
    return {"Equity": 0.75, "Hybrid": 0.15, "Debt": 0.1}


@dataclass
class Goal:
    name: str
    cost_today: float
    years: int
    already_saved: float = 0.0
    inflation_pct: float = 6.0


def plan_goal(goal: Goal, step_up_pct: float = 0.0) -> dict:
    mix = mix_for_horizon(goal.years)
    exp_ret = sum(EXPECTED_RETURN[k] * w for k, w in mix.items())
    future_cost = inflate(goal.cost_today, goal.inflation_pct, goal.years)
    saved_grows_to = goal.already_saved * (1 + exp_ret / 100) ** goal.years
    gap = max(future_cost - saved_grows_to, 0)
    monthly = required_sip(gap, exp_ret, goal.years, step_up_pct) if gap > 0 else 0.0
    return {
        "goal": goal.name,
        "years": goal.years,
        "cost_today": round(goal.cost_today),
        "future_cost": round(future_cost),
        "expected_return_%": round(exp_ret, 1),
        "monthly_sip": round(monthly),
        "mix": mix,
    }


def plan_goals(goals: list, step_up_pct: float = 0.0) -> pd.DataFrame:
    rows = []
    for g in goals:
        p = plan_goal(g, step_up_pct)
        rows.append({**{k: v for k, v in p.items() if k != "mix"},
                     **{f"{k.lower()}_%": int(v * 100) for k, v in p["mix"].items()}})
    return pd.DataFrame(rows)
