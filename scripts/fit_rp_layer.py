"""Fit the risk-parity layer (vol window, budgets, target vol, cap) to the chart's raw weights.

In months where bonds are held, chart total = raw (pre-filter) gross and chart bond = raw bond weight,
both independent of the momentum filter on other assets.
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.data import asset_returns  # noqa: E402
from allweather.engine import Params, run  # noqa: E402
from chart_match import label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    rets = asset_returns(us_ccy="usd", bond="cdb10", eq="tr", cash_tenor="3m")
    chart = pd.read_csv(ROOT / "data/target/chart_weights.csv")
    chart.index = pd.PeriodIndex(chart.month, freq="M")
    cb = chart[chart.bond > 0.05]
    rows = []
    for vw, cov, bud, tv, cap, lw in itertools.product([200, 250, 300], [True], ["asset"],
                                                       [0.052, 0.054, 0.055, 0.056, 0.058], [1.95, 2.0, 2.1],
                                                       [800, 900, 950, 1000, 1050, 1100, 1250]):
        sc = "port"
        p = Params(vol_window=vw, cov=cov, budget=bud, target_vol=tv, max_gross=cap, scale=sc, lev_window=lw)
        res = run(p, rets, start="2022-06-01")
        raw = res.reb_raw
        raw = raw.groupby(label(raw.index, rets.index, "mid")).last().reindex(cb.index)
        tot, bond = raw.sum(axis=1), raw.cn_bond
        nonbond_share = 1 - bond / tot
        rows.append({"vol_window": vw, "cov": cov, "budget": bud, "target_vol": tv, "cap": cap, "lev_window": lw,
                     "mae_total": (tot - cb.total).abs().mean(), "mae_bond": (bond - cb.bond).abs().mean(),
                     "corr_total": tot.corr(cb.total), "corr_bond": bond.corr(cb.bond),
                     "mae_share": (nonbond_share - (1 - cb.bond / cb.total)).abs().mean(),
                     "mean_total": tot.mean()})
    df = pd.DataFrame(rows)
    df["score"] = df.mae_total + df.mae_bond + df.mae_share
    pd.set_option("display.width", 200)
    print("chart mean total", round(cb.total.mean(), 3), "mean bond", round(cb.bond.mean(), 3))
    print(df.sort_values("score").head(25).round(4).to_string(index=False))
    df.to_csv(ROOT / "results/fit_rp_layer.csv", index=False)


if __name__ == "__main__":
    main()
