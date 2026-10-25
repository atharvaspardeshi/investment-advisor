"""
Investment calculators (pure maths, no ML).

SIP future value uses the standard formula banks/AMCs use:
    FV = P * [((1 + r)^n - 1) / r] * (1 + r)      r = monthly rate, n = months
(payment at the START of each month, which is how SIPs work).
"""
import pandas as pd


def monthly_rate(annual_pct: float) -> float:
    """12% per year -> 0.949% per month (compounded), not 1% (simple division overstates returns)."""
    return (1 + annual_pct / 100) ** (1 / 12) - 1


def sip(monthly: float, annual_pct: float, years: int, step_up_pct: float = 0.0) -> pd.DataFrame:
    """
    Year-by-year SIP growth. step_up_pct = yearly increase in SIP amount
    (e.g. 10 -> SIP grows 10% every year with your salary).
    """
    r = monthly_rate(annual_pct)
    value = invested = 0.0
    amount = monthly
    rows = [{"year": 0, "invested": 0, "value": 0, "monthly_sip": monthly}]
    for y in range(1, years + 1):
        for _ in range(12):
            value = (value + amount) * (1 + r)
            invested += amount
        rows.append({"year": y, "invested": round(invested), "value": round(value), "monthly_sip": round(amount)})
        amount *= 1 + step_up_pct / 100
    return pd.DataFrame(rows)


def lumpsum(amount: float, annual_pct: float, years: int) -> pd.DataFrame:
    rows = [{"year": y, "invested": amount, "value": round(amount * (1 + annual_pct / 100) ** y)}
            for y in range(years + 1)]
    return pd.DataFrame(rows)


def swp(corpus: float, monthly_withdrawal: float, annual_pct: float, years: int) -> pd.DataFrame:
    """Systematic Withdrawal Plan: how long a corpus lasts if you withdraw every month (e.g. retirement)."""
    r = monthly_rate(annual_pct)
    value, withdrawn = corpus, 0.0
    rows = [{"year": 0, "withdrawn": 0, "balance": round(corpus)}]
    for y in range(1, years + 1):
        for _ in range(12):
            if value <= 0:
                break
            take = min(monthly_withdrawal, value)
            value = (value - take) * (1 + r)
            withdrawn += take
        rows.append({"year": y, "withdrawn": round(withdrawn), "balance": round(max(value, 0))})
    return pd.DataFrame(rows)


def required_sip(target: float, annual_pct: float, years: int, step_up_pct: float = 0.0) -> float:
    """Monthly SIP needed to reach `target`. SIP is linear in the starting amount, so scale a ₹1 SIP."""
    if years <= 0:
        return float(target)
    r = monthly_rate(annual_pct)
    v, amt = 0.0, 1.0
    for _ in range(years):
        for _ in range(12):
            v = (v + amt) * (1 + r)
        amt *= 1 + step_up_pct / 100
    return target / v if v else float(target)


def inflate(amount_today: float, inflation_pct: float, years: int) -> float:
    """What today's cost will be after `years` of inflation (India's long-run CPI ~5-6%)."""
    return amount_today * (1 + inflation_pct / 100) ** years
