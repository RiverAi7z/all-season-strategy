"""Point-in-time macro candidate signals vs next-period (20d, vol-normalised) asset returns.

Selection is done on 2009-2017 only; 2018-2026 is reported as out-of-sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.config import DATA  # noqa: E402
from allweather.data import RAW, _asof_level, asset_returns  # noqa: E402
from allweather.engine import rebalance_mask  # noqa: E402
from allweather.macro import _monthly, _pit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def daily_series(name, col, date_col="date"):
    df = pd.read_parquet(RAW / f"{name}.parquet")
    df.index = pd.to_datetime(df[date_col])
    s = df[col].astype(float).sort_index()
    return s[~s.index.duplicated()]


def candidates(cal) -> pd.DataFrame:
    pmi = _monthly("macro_pmi", ["pmi010000", "pmi010500", "pmi011200"])
    ppi = _monthly("macro_ppi", ["ppi_yoy"]).ppi_yoy
    cpi = _monthly("macro_cpi", ["nt_yoy"]).nt_yoy
    m = _monthly("macro_m", ["m1_yoy", "m2_yoy"])
    sf = _monthly("macro_sf", ["inc_month"]).inc_month
    sf12 = sf.rolling(12).sum()
    sf_yoy = sf12 / sf12.shift(12) - 1
    out = pd.DataFrame(index=cal)
    out["pmi_mom"] = _pit(pmi.pmi010000.rolling(3).mean() - pmi.pmi010000.rolling(12).mean(), "pmi", cal)
    out["pmi_lvl"] = _pit(pmi.pmi010000 - 50, "pmi", cal)
    out["neword_mom"] = _pit(pmi.pmi010500.rolling(3).mean() - pmi.pmi010500.rolling(12).mean(), "pmi", cal)
    out["inputpx_lvl"] = _pit(pmi.pmi011200 - 50, "pmi", cal)
    out["ppi_d3"] = _pit(ppi - ppi.shift(3), "ppi", cal)
    out["cpi_d3"] = _pit(cpi - cpi.shift(3), "cpi", cal)
    out["m1_d3"] = _pit(m.m1_yoy - m.m1_yoy.shift(3), "m", cal)
    out["m2_d3"] = _pit(m.m2_yoy - m.m2_yoy.shift(3), "m", cal)
    out["scissor_d3"] = _pit((m.m1_yoy - m.m2_yoy).diff(3), "m", cal)
    out["sf_d3"] = _pit((sf_yoy - sf_yoy.shift(3)) * 100, "sf", cal)
    sh = pd.read_parquet(RAW / "shibor.parquet")
    sh.index = pd.to_datetime(sh["date"])
    s3m = sh["3m"].astype(float).sort_index()
    out["shibor3m_d60"] = _asof_level(s3m - s3m.shift(60), cal)
    yc = pd.read_parquet(RAW / "yc_cb.parquet")
    yc["trade_date"] = pd.to_datetime(yc["trade_date"])
    piv = yc.pivot_table(index="trade_date", columns="curve_term", values="yield", aggfunc="last").sort_index()
    piv.columns = piv.columns.astype(float)
    out["cn_term"] = _asof_level(piv[10.0] - piv[1.0], cal)
    out["cn10_d60"] = _asof_level(piv[10.0] - piv[10.0].shift(60), cal)
    r10 = daily_series("macro_us_real", "y10")
    n10 = daily_series("macro_us_nominal", "y10")
    n2 = daily_series("macro_us_nominal", "y2")
    out["us_real_d60"] = _asof_level(r10 - r10.shift(60), cal, lag_days=1)
    out["us_nom_d60"] = _asof_level(n10 - n10.shift(60), cal, lag_days=1)
    out["us_2s10s"] = _asof_level(n10 - n2, cal, lag_days=1)
    return out


def main():
    rets = asset_returns(**DATA)
    names = ["cn_eq", "us_eq", "cn_bond", "gold", "copper", "oil"]
    cand = candidates(rets.index)
    cand.to_parquet(ROOT / "data/macro_candidates.parquet")
    reb = rebalance_mask(rets.index, 20, "2026-09-24")
    idx = rets.index[reb]
    lvl = (1 + rets[names].fillna(0)).cumprod()
    fz = (lvl.shift(-20) / lvl - 1).loc[idx] / (rets[names].rolling(250).std().loc[idx] * np.sqrt(20))
    rows = []
    for c in cand.columns:
        x = cand.loc[idx, c]
        for a in names:
            r = {"signal": c, "asset": a}
            for lab, lo, hi in [("is", "2009", "2017"), ("oos", "2018", "2026-09")]:
                xx, yy = x.loc[lo:hi], fz.loc[lo:hi, a]
                ok = xx.notna() & yy.notna()
                r[f"ic_{lab}"] = xx[ok].corr(yy[ok], method="spearman") if ok.sum() > 30 else np.nan
                r[f"n_{lab}"] = int(ok.sum())
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/macro_ic.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot(index="signal", columns="asset", values="ic_is").round(2).to_string())
    print("\nOOS:")
    print(df.pivot(index="signal", columns="asset", values="ic_oos").round(2).to_string())
    se = 1 / np.sqrt(df.n_is.median())
    print(f"\napprox SE of an IC: IS {se:.3f}, OOS {1 / np.sqrt(df.n_oos.median()):.3f}")
    sel = df[(df.ic_is.abs() > 2 * se)].copy()
    sel["same_sign_oos"] = np.sign(sel.ic_is) == np.sign(sel.ic_oos)
    print("\nIS-significant (|IC| > 2 SE) and their OOS:\n", sel.round(3).to_string(index=False))
    print(f"\nshare keeping sign OOS: {sel.same_sign_oos.mean():.2f} of {len(sel)}")


if __name__ == "__main__":
    main()
