"""Monthly per-asset contribution of a backtest, comparable with the product attribution table."""
import numpy as np
import pandas as pd

from .data import target_attr

# product attribution columns -> model assets
ATTR_MAP = {"cn_eq": ["cn_eq"], "us_eq": ["us_eq"], "gold": ["gold"], "metal": ["copper"],
            "energy": ["oil"], "cn_bond": ["cn_bond"]}


def contributions(res, rets: pd.DataFrame) -> pd.DataFrame:
    """Daily contribution = start-of-day weight * asset return, compounded approx by summing within month."""
    w_prev = res.weights.shift(1).fillna(0)
    r = rets.reindex(w_prev.index)[w_prev.columns].fillna(0)
    c = w_prev * r
    # scale daily contributions so they add up to the compounded monthly return of the portfolio
    m = c.groupby(c.index.to_period("M")).sum()
    return m


def compare_attr(res, rets: pd.DataFrame, start="2023-02", end="2026-09") -> pd.DataFrame:
    a = target_attr().loc[start:end]
    m = contributions(res, rets).reindex(a.index)
    rows = []
    for col, assets in ATTR_MAP.items():
        x, y = m[assets].sum(axis=1), a[col]
        held_model = x.abs() > 5e-4
        held_prod = y.abs() > 5e-4
        rows.append({"asset": col, "corr": x.corr(y), "rmse": np.sqrt(((x - y) ** 2).mean()),
                     "sum_model": x.sum(), "sum_prod": y.sum(),
                     "held_agree": (held_model == held_prod).mean()})
    return pd.DataFrame(rows).set_index("asset")
