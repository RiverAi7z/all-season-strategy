"""Compare a backtest against the CI011001.WI monthly target."""
import numpy as np
import pandas as pd

from .data import target_monthly


def monthly(daily_ret: pd.Series) -> pd.Series:
    m = (1 + daily_ret).groupby(daily_ret.index.to_period("M")).prod() - 1
    return m


def stats(m: pd.Series) -> dict:
    nav = (1 + m).cumprod()
    dd = nav / nav.cummax() - 1
    ann = nav.iloc[-1] ** (12 / len(m)) - 1
    vol = m.std() * np.sqrt(12)
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.02) / vol if vol else np.nan, "mdd": dd.min()}


def compare(daily_ret: pd.Series, start="2009-01", end="2026-09") -> dict:
    tgt = target_monthly().loc[start:end]
    m = monthly(daily_ret).reindex(tgt.index)
    ok = m.notna()
    m, t = m[ok], tgt[ok]
    diff = m - t
    ym = (1 + m).groupby(m.index.year).prod() - 1
    yt = (1 + t).groupby(t.index.year).prod() - 1
    s, st = stats(m), stats(t)
    return {
        "n": int(ok.sum()),
        "corr": m.corr(t),
        "te": diff.std() * np.sqrt(12),
        "rmse": np.sqrt((diff ** 2).mean()),
        "bias": diff.mean() * 12,
        "corr_y": ym.corr(yt),
        "ann": s["ann"], "vol": s["vol"], "mdd": s["mdd"], "sharpe": s["sharpe"],
        "t_ann": st["ann"], "t_vol": st["vol"], "t_mdd": st["mdd"],
    }


def fmt(d: dict) -> str:
    return (f"corr={d['corr']:.3f} corr_y={d['corr_y']:.3f} te={d['te']:.2%} bias={d['bias']:+.2%} | "
            f"ann={d['ann']:.2%} vol={d['vol']:.2%} mdd={d['mdd']:.2%} (tgt {d['t_ann']:.2%}/{d['t_vol']:.2%}/{d['t_mdd']:.2%})")
