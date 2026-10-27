"""Detailed diagnostics for one parameter set: yearly table, attribution fit, recent rebalances.

usage: uv run python scripts/diagnose.py '{"budget":"class","cov":true,...}' ['{"us_ccy":"cny"}']
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.attribution import compare_attr  # noqa: E402
from allweather.data import asset_returns, target_monthly  # noqa: E402
from allweather.engine import Params, run  # noqa: E402
from allweather.metrics import compare, fmt, monthly  # noqa: E402


def to_params(d):
    d = dict(d)
    for k in ("assets", "budget_override", "stop_assets"):
        if k in d:
            d[k] = tuple(tuple(x) if isinstance(x, list) else x for x in d[k])
    return Params(**d)


def main():
    p = to_params(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
    dk = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rets = asset_returns(**dk)
    res = run(p, rets)
    pd.set_option("display.width", 220)
    print(p)
    print("full   ", fmt(compare(res.daily.ret)))
    print("2019+  ", fmt(compare(res.daily.ret, start="2019-01")))
    print("2023-02+", fmt(compare(res.daily.ret, start="2023-02")))
    m = monthly(res.daily.ret)
    t = target_monthly()
    y = pd.DataFrame({"model": (1 + m).groupby(m.index.year).prod() - 1,
                      "target": (1 + t).groupby(t.index.year).prod() - 1}).dropna()
    y["mdd_model"] = [((1 + res.daily.ret[str(yr)]).cumprod() / (1 + res.daily.ret[str(yr)]).cumprod().cummax() - 1).min()
                      for yr in y.index]
    tgt_y = pd.read_csv(Path(__file__).resolve().parents[1] / "data/target/index_yearly.csv").set_index("year")
    y["mdd_target"] = tgt_y.mdd
    print(y.round(4).to_string())
    print(compare_attr(res, rets).round(4).to_string())
    print("avg rebalance weights:\n", res.reb_weights.loc["2023":].mean().round(3).to_string())
    print(res.reb_weights.loc["2025-06":].round(3).to_string())
    if len(res.stops):
        print("stops per year:\n", res.stops.groupby([res.stops.date.dt.year, "asset"]).size().unstack(fill_value=0).tail(8))


if __name__ == "__main__":
    main()
