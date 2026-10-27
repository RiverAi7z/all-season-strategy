"""Macro scenario scores (point-in-time) -> per-asset risk-budget tilts.

Every monthly series is shifted to the date it is actually published (conservatively) before it
is mapped onto the trading calendar, so a rebalance on day t only sees data released by t.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import RAW, _asof_level, sse_calendar

# (months after the reference month, day of month) when the figure is safely public
RELEASE = {"pmi": (1, 1), "ppi": (1, 15), "cpi": (1, 15), "m": (1, 20), "sf": (1, 20)}

# Prior exposures of each asset to the macro factors (not fitted)
#   growth: PMI momentum, infl: PPI momentum, liq: M1-M2 scissors momentum, real: US 10y real-yield change
EXPOSURE = pd.DataFrame(
    {"growth": [1, 0, -1, 0, 1, 0.5],
     "infl": [0, 0, -1, 0.5, 0.5, 1],
     "liq": [1, 0, 0, 0, 0, 0],
     "real": [0, -1, 0, -1, 0, 0]},
    index=["cn_eq", "us_eq", "cn_bond", "gold", "copper", "oil"], dtype=float)


def _monthly(name, cols):
    df = pd.read_parquet(RAW / f"{name}.parquet")
    df.index = pd.PeriodIndex(df["month"].astype(str), freq="M")
    return df[cols].apply(pd.to_numeric, errors="coerce").sort_index()


def _release_dates(idx: pd.PeriodIndex, key: str) -> pd.DatetimeIndex:
    months, day = RELEASE[key]
    return (idx + months).to_timestamp() + pd.Timedelta(days=day - 1)


def _pit(series: pd.Series, key: str, cal: pd.DatetimeIndex) -> pd.Series:
    s = series.dropna()
    s.index = _release_dates(s.index, key)
    return _asof_level(s, cal)


def _z(x: pd.Series, window: int) -> pd.Series:
    """Scale a (roughly zero-centred) momentum series by its own trailing dispersion, clipped to +-2."""
    sd = x.rolling(window, min_periods=window // 2).std()
    return (x / sd).clip(-2, 2)


@lru_cache(maxsize=4)
def factor_panel(us_window=60) -> pd.DataFrame:
    """Daily point-in-time macro factor z-scores on the SSE calendar."""
    cal = sse_calendar()
    pmi = _monthly("macro_pmi", ["pmi010000"])["pmi010000"]
    ppi = _monthly("macro_ppi", ["ppi_yoy"])["ppi_yoy"]
    m = _monthly("macro_m", ["m1_yoy", "m2_yoy"])
    growth = pmi.rolling(3).mean() - pmi.rolling(12).mean()
    infl = ppi - ppi.shift(3)
    scissors = m.m1_yoy - m.m2_yoy
    liq = scissors - scissors.shift(3)
    out = pd.DataFrame(index=cal)
    out["growth"] = _pit(_z(growth, 60), "pmi", cal)
    out["infl"] = _pit(_z(infl, 60), "ppi", cal)
    out["liq"] = _pit(_z(liq, 60), "m", cal)
    real = pd.read_parquet(RAW / "macro_us_real.parquet")
    real.index = pd.to_datetime(real["date"])
    r10 = real["y10"].astype(float).sort_index()
    r10 = r10[~r10.index.duplicated()]
    dr = r10 - r10.shift(us_window)
    out["real"] = _asof_level(_z(dr, 750), cal, lag_days=1)   # US data: previous calendar day
    return out


def asset_scores(us_window=60, factors=("growth", "infl", "liq", "real")) -> pd.DataFrame:
    """Daily per-asset macro score = exposure-weighted average of factor z-scores (row-normalised)."""
    f = factor_panel(us_window)[list(factors)].fillna(0.0)
    E = EXPOSURE[list(factors)]
    norm = E.abs().sum(axis=1).replace(0, np.nan)
    S = (f.values @ E.T.values) / norm.values
    return pd.DataFrame(np.nan_to_num(S), index=f.index, columns=E.index)


def zscore_daily(s: pd.Series, window=750) -> pd.Series:
    """Trailing (point-in-time) z-score of a daily series, clipped to +-2."""
    sd = s.rolling(window, min_periods=250).std()
    mu = s.rolling(window, min_periods=250).mean()
    return ((s - mu) / sd).clip(-2, 2)


def us_real_rate_z(window=120) -> pd.Series:
    """z-score of the change in the US 10y TIPS yield over `window` days, as known at each SSE close."""
    real = pd.read_parquet(RAW / "macro_us_real.parquet")
    real.index = pd.to_datetime(real["date"])
    r10 = real["y10"].astype(float).sort_index()
    r10 = r10[~r10.index.duplicated()]
    return _asof_level(zscore_daily(r10 - r10.shift(window)), sse_calendar(), lag_days=1).fillna(0.0)


def us_real_scores(window=120) -> pd.DataFrame:
    """Selected overlay: rising US real yields score gold and US equities down (and falling ones up)."""
    z = us_real_rate_z(window)
    zero = 0.0 * z
    return pd.DataFrame({"cn_eq": zero, "us_eq": -z, "cn_bond": zero, "gold": -z, "copper": zero, "oil": zero})
