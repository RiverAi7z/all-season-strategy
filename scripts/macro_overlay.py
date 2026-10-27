"""Backtest macro overlays on top of the FINAL replication; report in-sample (2009-2017) vs out-of-sample (2018+).

Overlays (all point-in-time):
  prior    fixed economic exposures to growth / inflation / liquidity / US real-rate factors
  credit   credit-impulse composite (social financing, M1, M1-M2) -> + cn_eq/copper/oil, - cn_bond
  usreal   US 10y real-yield change -> - gold / us_eq
  adaptive rolling 5y Spearman IC of each candidate signal per asset, IC-weighted score
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.config import DATA, FINAL  # noqa: E402
from allweather.data import asset_returns, target_monthly  # noqa: E402
from allweather.engine import rebalance_mask, run  # noqa: E402
from allweather.macro import asset_scores, zscore_daily  # noqa: E402
from allweather.metrics import compare, monthly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["cn_eq", "us_eq", "cn_bond", "gold", "copper", "oil"]
CANDS = ["pmi_mom", "neword_mom", "inputpx_lvl", "ppi_d3", "cpi_d3", "m1_d3", "m2_d3", "scissor_d3", "sf_d3",
         "shibor3m_d60", "us_real_d60", "us_nom_d60", "us_2s10s"]


@lru_cache(maxsize=1)
def cand_z() -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "data/macro_candidates.parquet")
    return c[CANDS].apply(zscore_daily)


def scores_credit() -> pd.DataFrame:
    z = cand_z()
    credit = z[["sf_d3", "m1_d3", "scissor_d3"]].mean(axis=1).fillna(0)
    expo = {"cn_eq": 1, "us_eq": 0, "cn_bond": -1, "gold": 0, "copper": 1, "oil": 0.5}
    return pd.DataFrame({a: credit * e for a, e in expo.items()})


def scores_usreal() -> pd.DataFrame:
    r = cand_z()["us_real_d60"].fillna(0)
    expo = {"cn_eq": 0, "us_eq": -1, "cn_bond": 0, "gold": -1, "copper": 0, "oil": 0}
    return pd.DataFrame({a: r * e for a, e in expo.items()})


@lru_cache(maxsize=4)
def scores_adaptive(lookback=60, min_obs=36) -> pd.DataFrame:
    """At each rebalance t: IC of every signal vs realised next-period returns over the last `lookback`
    rebalances whose forward window has already closed by t; score = sum(IC * z) / sum(|IC|)."""
    rets = asset_returns(**DATA)
    z = cand_z().reindex(rets.index)
    reb_idx = np.flatnonzero(rebalance_mask(rets.index, FINAL.reb_period, FINAL.anchor))
    lvl = (1 + rets[NAMES].fillna(0)).cumprod().to_numpy()
    vol = (rets[NAMES].rolling(250).std() * np.sqrt(20)).to_numpy()
    zz = z.to_numpy()
    out = pd.DataFrame(0.0, index=rets.index, columns=NAMES)
    rows = []
    for k, t in enumerate(reb_idx):
        past = [j for j in reb_idx[:k] if j + 20 <= t][-lookback:]
        sc = np.zeros(len(NAMES))
        if len(past) >= min_obs:
            P = np.array(past)
            fwd = (lvl[P + 20] / lvl[P] - 1) / vol[P]
            X = zz[P]
            for a in range(len(NAMES)):
                y = fwd[:, a]
                ics = []
                for s in range(X.shape[1]):
                    ok = ~np.isnan(X[:, s]) & ~np.isnan(y)
                    ics.append(pd.Series(X[ok, s]).corr(pd.Series(y[ok]), method="spearman") if ok.sum() >= min_obs else 0.0)
                ics = np.nan_to_num(np.array(ics))
                cur = np.nan_to_num(zz[t])
                sc[a] = (ics * cur).sum() / (np.abs(ics).sum() + 1e-9)
        rows.append((rets.index[t], *sc))
    reb_scores = pd.DataFrame(rows, columns=["date", *NAMES]).set_index("date")
    return reb_scores.reindex(rets.index).ffill().fillna(0.0)


def period_stats(ret: pd.Series, lo, hi) -> dict:
    r = ret.loc[lo:hi]
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.02) / vol, "mdd": mdd, "calmar": ann / -mdd}


def evaluate(job):
    name, k, veto = job
    rets = asset_returns(**DATA)
    S = {"none": None, "prior": asset_scores(), "credit": scores_credit(), "usreal": scores_usreal(),
         "adaptive": scores_adaptive()}[name]
    p = replace(FINAL, macro_k=k, macro_veto=veto)
    res = run(p, rets, scores=S)
    r = res.daily.ret
    row = {"overlay": name, "k": k, "veto": veto}
    for lab, lo, hi in [("is", "2009", "2017"), ("oos", "2018", "2026-09")]:
        row.update({f"{m}_{lab}": v for m, v in period_stats(r, lo, hi).items()})
    row["corr_full"] = compare(r)["corr"]
    row["corr_live"] = compare(r, start="2023-02")["corr"]
    return row


def main():
    jobs = [("none", 0.0, 0.0)]
    for name in ("prior", "credit", "usreal", "adaptive"):
        jobs += [(name, k, 0.0) for k in (0.1, 0.2, 0.3, 0.5, 0.75)]
        jobs += [(name, 0.0, v) for v in (1.0, 1.5)]
    scores_adaptive()  # warm cache once before forking
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(evaluate, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/macro_overlay.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["overlay", "k", "veto", "sharpe_is", "mdd_is", "ann_is", "sharpe_oos", "mdd_oos", "ann_oos", "calmar_oos",
            "corr_full", "corr_live"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
