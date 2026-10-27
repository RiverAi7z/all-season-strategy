"""Grid-search model variants against CI011001.WI monthly returns (and product attribution).

usage: uv run python scripts/calibrate.py <stage> [--top 20]
"""
import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.data import asset_returns  # noqa: E402
from allweather.engine import Params, run  # noqa: E402
from allweather.metrics import compare  # noqa: E402
from allweather.attribution import compare_attr  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def evaluate(job):
    data_kw, p = job
    rets = asset_returns(**data_kw)
    res = run(p, rets)
    c = compare(res.daily.ret)
    c2 = compare(res.daily.ret, start="2019-01")
    c3 = compare(res.daily.ret, start="2023-02")
    at = compare_attr(res, rets)
    return {**data_kw, **{k: v for k, v in asdict(p).items() if k not in ("assets", "anchor")},
            **c, "corr_19": c2["corr"], "te_19": c2["te"], "corr_23": c3["corr"], "te_23": c3["te"],
            "attr": at["corr"].mean()}


def grid(base: Params, **axes):
    keys = list(axes)
    for vals in itertools.product(*axes.values()):
        yield replace(base, **dict(zip(keys, vals)))


def stage_jobs(stage):
    d0 = dict(us_ccy="usd", bond="cdb10", oil_proxy=None, eq="price", cash_tenor="1w")
    base = Params()
    if stage == "base":
        # pure risk parity: no filter, no stop
        for dk in [dict(d0), {**d0, "us_ccy": "cny"}]:
            for p in grid(base, budget=["asset", "class"], cov=[False, True], vol_window=[20, 60, 120, 250],
                          target_vol=[0.03, 0.04, 0.05, 0.06], cash_mode=["residual", "free"],
                          max_gross=[2.0, 3.0]):
                yield dk, p
    elif stage == "mom":
        for p in grid(base, budget=["asset", "class"], cov=[False, True], vol_window=[60, 120, 250],
                      target_vol=[0.03, 0.04, 0.05], cash_mode=["residual", "free"],
                      mom_window=[20, 40, 60, 120, 250]):
            yield d0, p
    elif stage == "stop":
        b = replace(base, budget="asset", cov=False, vol_window=250, target_vol=0.05, cash_mode="residual")
        for st, ks, ws in [("none", [0], [0]),
                           ("cum", [0.25, 0.5, 0.75, 1.0, 1.5], [20]), ("dd", [0.25, 0.5, 0.75, 1.0, 1.5], [20]),
                           ("loss", [0.05, 0.1, 0.2, 0.3, 0.5], [20]), ("mom", [0], [5, 10, 15, 20, 30, 40]),
                           ("ma", [0], [5, 10, 20, 40, 60])]:
            for p in grid(b, stop=[st], stop_k=ks, stop_window=ws, mom_window=[0, 20, 40],
                          target_vol=[0.04, 0.05, 0.06]):
                yield d0, p
    elif stage == "data":
        best = replace(base, budget="asset", vol_window=250, target_vol=0.06, mom_window=40, stop="ma", stop_window=60)
        for us, oil, eq, bond, cash in itertools.product(["usd", "cny"], [None, "FU_SHF"], ["price", "tr"],
                                                         ["cdb10", "cgb10", "tfut"], ["1w", "3m"]):
            dk = dict(us_ccy=us, bond=bond, oil_proxy=oil, eq=eq, cash_tenor=cash)
            for p in grid(best, cash_mode=["residual", "free_plus"]):
                yield dk, p
    elif stage == "refine":
        d1 = {**d0, **DATA_BEST}
        for p in grid(base, budget=["asset", "class"], cov=[False, True], vol_window=[120, 250],
                      target_vol=[0.05, 0.06, 0.07], cash_mode=["residual", "free_plus"],
                      mom_type=["ret", "ma"], mom_window=[20, 40, 60],
                      stop=["ma", "mom"], stop_window=[20, 40, 60, 90, 120]):
            yield d1, p
    elif stage == "final":
        # risk-parity layer fixed from the weight chart fit (scripts/fit_rp_layer.py)
        d1 = {**d0, **DATA_BEST}
        rp = replace(base, cov=True, vol_window=250, lev_window=1000, max_gross=2.0)
        stops = [("none", 0, 0)] + [(s, 0, w) for s in ("ma", "mom") for w in (20, 40, 60, 90)] + \
                [(s, k, 20) for s in ("cum", "dd") for k in (0.5, 1.0, 1.5)]
        for st, k, w in stops:
            for p in grid(rp, stop=[st], stop_k=[k], stop_window=[w], target_vol=[0.05, 0.055, 0.06],
                          mom_type=["ret", "ma"], mom_window=[20, 40, 60, 90], cash_mode=["residual", "free_plus"]):
                yield d1, p
    elif stage == "fine":
        d1 = {**d0, **DATA_BEST}
        rp = replace(base, cov=True, vol_window=250, lev_window=1000, max_gross=2.0, mom_type="ma")
        stops = [(s, 0, w) for s in ("ma",) for w in (45, 50, 55, 60, 65, 70, 80)] + \
                [(s, k, 20) for s in ("cum", "dd") for k in (0.75, 1.0, 1.25)]
        nonbond = ("cn_eq", "us_eq", "gold", "copper", "oil")
        for st, k, w in stops:
            for p in grid(rp, stop=[st], stop_k=[k], stop_window=[w], target_vol=[0.055, 0.06],
                          mom_window=[30, 35, 40, 45, 50, 60], stop_assets=[(), nonbond]):
                yield d1, p
    else:
        raise SystemExit(f"unknown stage {stage}")


DATA_BEST = dict(us_ccy="usd", bond="cdb10", oil_proxy=None, eq="tr", cash_tenor="3m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sort", default="te")
    a = ap.parse_args()
    jobs = list(stage_jobs(a.stage))
    print(f"{len(jobs)} jobs")
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(evaluate, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"grid_{a.stage}.csv", index=False)
    show = df.sort_values(a.sort, ascending=a.sort not in ("corr", "corr_y", "corr_19", "corr_23", "attr"))
    cols = [c for c in df.columns if df[c].nunique() > 1]
    pd.set_option("display.width", 250)
    print(show[cols].head(a.top).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
