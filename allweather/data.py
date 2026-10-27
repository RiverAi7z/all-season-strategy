"""Build the daily asset return panel on the SSE trading calendar from data/raw."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TARGET = ROOT / "data/target"


def _read(name):
    df = pd.read_parquet(RAW / f"{name}.parquet")
    df["trade_date"] = pd.to_datetime(df["trade_date"] if "trade_date" in df else df["date"])
    return df.sort_values("trade_date")


def sse_calendar(start="2005-01-01", end=None):
    cal = pd.read_parquet(RAW / "trade_cal.parquet")
    cal = cal[cal.is_open.astype(int) == 1]
    d = pd.DatetimeIndex(pd.to_datetime(cal.cal_date)).sort_values()
    last = _read("idx_000300_SH").trade_date.max()
    end = min(pd.Timestamp(end), last) if end else last
    return d[(d >= pd.Timestamp(start)) & (d <= end)]


def _asof_level(s: pd.Series, cal: pd.DatetimeIndex, lag_days=0):
    """Level of s as of each calendar date (using obs dated <= date - lag_days)."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    idx = cal - pd.Timedelta(days=lag_days)
    pos = s.index.searchsorted(idx, side="right") - 1
    vals = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=cal)


def _index_close(name):
    df = _read(name)
    return df.set_index("trade_date")["close"]


def _fut_ret(name):
    """Roll-adjusted daily return of the main contract: close / same-contract pre_close."""
    df = _read(name).dropna(subset=["close"])
    prev = df["pre_close"].where(df["pre_close"] > 0, df["pre_settle"])
    r = df["close"] / prev - 1
    return pd.Series(r.values, index=df.trade_date.values)


def _bond_from_curve(term=10.0):
    """Constant-maturity par-bond total return from the CGB yield curve (annual coupon)."""
    yc = _read("yc_cb")
    y = yc[yc.curve_term.astype(float) == term].set_index("trade_date")["yield"].astype(float) / 100
    y = y[~y.index.duplicated()].sort_index()
    y0 = y.shift(1)
    dt = y.index.to_series().diff().dt.days / 365
    # modified duration / convexity of a par bond with annual coupon y0
    n = term
    D = (1 - (1 + y0) ** -n) / y0
    C = (2 / y0 ** 2) * (1 - (1 + y0) ** -n) - (2 * n / (y0 * (1 + y0) ** (n + 1)))
    dy = y - y0
    r = y0 * dt - D * dy + 0.5 * C * dy ** 2
    return r.dropna()


def _cash_rate(cal, tenor="1w"):
    sh = pd.read_parquet(RAW / "shibor.parquet")
    sh["date"] = pd.to_datetime(sh["date"])
    s = sh.set_index("date")[tenor].astype(float).sort_index() / 100
    lvl = _asof_level(s, cal).bfill()
    days = cal.to_series().diff().dt.days.fillna(1).values
    return pd.Series(lvl.shift(1).bfill().values * days / 365, index=cal)


def _ret_on_cal(r: pd.Series, cal):
    """Compound a daily return series onto the SSE calendar (sum log returns between dates)."""
    r = r[~r.index.duplicated(keep="last")].sort_index()
    lvl = np.exp(np.log1p(r.fillna(0)).cumsum())
    first = r.index.min()
    a = _asof_level(lvl, cal)
    out = a / a.shift(1) - 1
    out[cal <= first] = np.nan
    return out


@lru_cache(maxsize=8)
def asset_returns(us_ccy="usd", bond="cdb10", oil_proxy=None, eq="price", cash_tenor="1w"):
    """Daily simple returns on SSE days. Columns: cn_eq us_eq cn_bond gold copper oil cash."""
    cal = sse_calendar()
    out = pd.DataFrame(index=cal)
    eq_code = "idx_H00300_CSI" if eq == "tr" else "idx_000300_SH"
    out["cn_eq"] = _index_close(eq_code).reindex(cal).pct_change()

    # SPX: US close of the previous calendar day is the latest print available at China close
    spx = _read("spx").set_index("trade_date")["close"]
    lvl = _asof_level(spx, cal, lag_days=1)
    if us_ccy == "cny":
        fx = _read("usdcnh").set_index("trade_date")["bid_close"]
        lvl = lvl * _asof_level(fx, cal, lag_days=1)
    out["us_eq"] = lvl.pct_change()

    if bond == "cdb10":
        out["cn_bond"] = _index_close("idx_930849_CSI").reindex(cal).pct_change()
    elif bond == "cgb10":
        out["cn_bond"] = _index_close("idx_930916_CSI").reindex(cal).pct_change()
    elif bond == "curve10":
        out["cn_bond"] = _ret_on_cal(_bond_from_curve(10.0), cal)
    elif bond == "tfut":
        out["cn_bond"] = _ret_on_cal(_fut_ret("fut_T_CFX"), cal)

    out["gold"] = _ret_on_cal(_fut_ret("fut_AU_SHF"), cal)
    out["copper"] = _ret_on_cal(_fut_ret("fut_CU_SHF"), cal)
    oil = _ret_on_cal(_fut_ret("fut_SC_INE"), cal)
    if oil_proxy:
        px = _ret_on_cal(_fut_ret(f"fut_{oil_proxy}"), cal)
        oil = oil.where(oil.notna(), px.where(cal < oil.first_valid_index()))
    out["oil"] = oil
    out["cash"] = _cash_rate(cal, cash_tenor)
    return out


def target_monthly():
    t = pd.read_csv(TARGET / "index_monthly.csv")
    t.index = pd.PeriodIndex(t.month, freq="M")
    return t["ret"]


def target_attr():
    a = pd.read_csv(TARGET / "attr_p1.csv")
    a.index = pd.PeriodIndex(a.month, freq="M")
    return a.drop(columns="month")
