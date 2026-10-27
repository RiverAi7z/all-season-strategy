"""Match model rebalance weights to the digitized weight chart under different schedules / label mappings.

usage: uv run python scripts/chart_match.py '<params json>' '<data json>'
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.data import asset_returns  # noqa: E402
from allweather.engine import run  # noqa: E402
from diagnose import to_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def class_weights(rw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rw.index)
    out["eq"] = rw[["cn_eq", "us_eq"]].sum(axis=1)
    out["bond"] = rw["cn_bond"]
    out["cmd"] = rw[["gold", "copper", "oil"]].sum(axis=1)
    return out


def label(dates: pd.DatetimeIndex, cal: pd.DatetimeIndex, how: str) -> pd.PeriodIndex:
    if how == "same":         # rebalance month itself
        return dates.to_period("M")
    pos = cal.searchsorted(dates)
    if how == "mid":          # month containing the middle of the holding period
        return cal[np.minimum(pos + 10, len(cal) - 1)].to_period("M")
    if how == "next1":        # month of the next trading day
        return cal[np.minimum(pos + 1, len(cal) - 1)].to_period("M")
    raise ValueError(how)


def score(model: pd.DataFrame, chart: pd.DataFrame) -> dict:
    m = model.reindex(chart.index)
    ok = m.notna().all(axis=1)
    m, c = m[ok], chart[ok]
    bond_agree = ((m.bond > 0.05) == (c.bond > 0.05)).mean()
    eq_agree = ((m["eq"] > 0.02) == (c["eq"] > 0.02)).mean()
    cmd_agree = ((m.cmd > 0.02) == (c.cmd > 0.02)).mean()
    mae = (m - c[["eq", "bond", "cmd"]]).abs().mean()
    return {"n": int(ok.sum()), "bond_agree": bond_agree, "eq_agree": eq_agree, "cmd_agree": cmd_agree,
            "mae_eq": mae["eq"], "mae_bond": mae.bond, "mae_cmd": mae.cmd}


def main():
    p = to_params(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
    dk = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rets = asset_returns(**dk)
    cal = rets.index
    chart = pd.read_csv(ROOT / "data/target/chart_weights.csv")
    chart.index = pd.PeriodIndex(chart.month, freq="M")
    rows = []
    for period in (19, 20, 21, 22):
        for shift in range(-3, 4):
            anchor = cal[cal.get_loc(pd.Timestamp(p.anchor)) + shift]
            res = run(replace(p, reb_period=period, anchor=str(anchor.date())), rets, start="2022-06-01")
            cw = class_weights(res.reb_weights)
            for how in ("same", "mid", "next1"):
                lab = label(cw.index, cal, how)
                m = cw.groupby(lab).last()
                rows.append({"period": period, "anchor": str(anchor.date()), "label": how, **score(m, chart)})
    df = pd.DataFrame(rows)
    df["agree"] = df[["bond_agree", "eq_agree", "cmd_agree"]].mean(axis=1)
    pd.set_option("display.width", 200)
    print(df.sort_values(["bond_agree", "agree"], ascending=False).head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
