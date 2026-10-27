"""Run the final replication, write results/ and print current signals + upcoming rebalance dates."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.attribution import compare_attr  # noqa: E402
from allweather.config import DATA, FINAL  # noqa: E402
from allweather.data import RAW, asset_returns, target_monthly  # noqa: E402
from allweather.engine import run, target_weights  # noqa: E402
from allweather.metrics import compare, fmt, monthly  # noqa: E402
from chart_match import label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_TGT, C_MODEL = "#2a78d6", "#eb6834"


def upcoming_rebalances(n=6):
    cal = pd.read_parquet(RAW / "trade_cal.parquet")
    d = pd.DatetimeIndex(pd.to_datetime(cal[cal.is_open.astype(int) == 1].cal_date)).sort_values()
    a = d.get_loc(pd.Timestamp(FINAL.anchor))
    reb = d[((np.arange(len(d)) - a) % FINAL.reb_period) == 0]
    return reb[reb > pd.Timestamp.today().normalize() - pd.Timedelta(days=40)][:n]


def current_signals(rets, res):
    names = list(FINAL.assets)
    lvl = (1 + rets[names].fillna(0)).cumprod()
    last = rets.index[-1]
    t = len(rets) - 1
    hist = rets[names].to_numpy()[t - FINAL.vol_window + 1: t + 1]
    hist_lev = rets[names].to_numpy()[t - FINAL.lev_window + 1: t + 1]
    raw = target_weights(FINAL, hist, names, hist_lev)
    above = lvl.iloc[-1] > lvl.iloc[-FINAL.mom_window:].mean()
    df = pd.DataFrame({
        "held_now": res.weights.iloc[-1],
        f"px/MA{FINAL.mom_window}-1": lvl.iloc[-1] / lvl.iloc[-FINAL.mom_window:].mean() - 1,
        f"px/MA{FINAL.stop_window}-1": lvl.iloc[-1] / lvl.iloc[-FINAL.stop_window:].mean() - 1,
        "raw_if_rebal_today": raw,
        "target_if_rebal_today": np.where(above, raw, 0.0),
    })
    return last, df


def plot(res, chart_cmp, path):
    m = monthly(res.daily.ret)
    t = target_monthly()
    idx = t.index.intersection(m.index)
    nav_t, nav_m = (1 + t[idx]).cumprod(), (1 + m[idx]).cumprod()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    x = idx.to_timestamp(how="end")
    ax.plot(x, nav_t.values, color=C_TGT, lw=2, label="CI011001.WI (target)")
    ax.plot(x, nav_m.values, color=C_MODEL, lw=2, label="Replication")
    ax.text(x[-1], nav_t.iloc[-1], f"  target {nav_t.iloc[-1]:.2f}", color=INK, va="center", fontsize=9)
    ax.text(x[-1], nav_m.iloc[-1], f"  model {nav_m.iloc[-1]:.2f}", color=INK, va="center", fontsize=9)
    c_all, c23 = compare(res.daily.ret), compare(res.daily.ret, start="2023-02")
    ax.set_title(f"Cumulative NAV, 2009-01 to 2026-09   (monthly corr {c_all['corr']:.2f} full / "
                 f"{c23['corr']:.2f} since 2023-02)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax2 = axes[1]
    yr = pd.DataFrame({"target": (1 + t[idx]).groupby(idx.year).prod() - 1,
                       "model": (1 + m[idx]).groupby(idx.year).prod() - 1})
    xs = np.arange(len(yr))
    ax2.bar(xs - 0.2, yr.target * 100, width=0.38, color=C_TGT, label="target")
    ax2.bar(xs + 0.2, yr.model * 100, width=0.38, color=C_MODEL, label="replication")
    ax2.set_xticks(xs, [str(y) if y < 2026 else "2026\nYTD" for y in yr.index], fontsize=8)
    ax2.set_ylabel("calendar-year return, %")
    ax2.set_title("Calendar-year returns", loc="left")
    ax2.legend(frameon=False, loc="upper left", ncols=2)
    ax3 = axes[2]
    xm = chart_cmp.index.to_timestamp()
    ax3.plot(xm, chart_cmp.chart_total, color=C_TGT, lw=2, label="report chart: gross")
    ax3.plot(xm, chart_cmp.model_total, color=C_TGT, lw=1.5, ls="--", label="model: gross")
    ax3.plot(xm, chart_cmp.chart_bond, color=C_MODEL, lw=2, label="report chart: bond")
    ax3.plot(xm, chart_cmp.model_bond, color=C_MODEL, lw=1.5, ls="--", label="model: bond")
    ax3.set_ylabel("weight (x NAV)")
    ax3.set_title("Rebalance weights vs the report's chart (months where bonds are held)", loc="left")
    ax3.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    rets = asset_returns(**DATA)
    res = run(FINAL, rets)
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    print(FINAL)
    for s in ("2009-01", "2019-01", "2023-02"):
        print(f"{s}+  ", fmt(compare(res.daily.ret, start=s)))

    daily = res.daily.join(res.weights.add_prefix("w_"))
    daily.to_csv(OUT / "final_daily.csv")
    m, t = monthly(res.daily.ret), target_monthly()
    mon = pd.DataFrame({"model": m, "target": t}).dropna()
    mon["diff"] = mon.model - mon.target
    mon.to_csv(OUT / "final_monthly.csv")
    reb = res.reb_raw.add_prefix("raw_").join(res.reb_weights.add_prefix("w_"))
    reb.to_csv(OUT / "rebalance_weights.csv")
    res.stops.to_csv(OUT / "stops.csv", index=False)
    print(compare_attr(res, rets).round(3).to_string())

    chart = pd.read_csv(ROOT / "data/target/chart_weights.csv")
    chart.index = pd.PeriodIndex(chart.month, freq="M")
    raw = res.reb_raw.groupby(label(res.reb_raw.index, rets.index, "mid")).last()
    cmp_ = pd.DataFrame({"chart_total": chart.total, "chart_bond": chart.bond,
                         "model_total": raw.sum(axis=1), "model_bond": raw.cn_bond}).dropna()
    cmp_ = cmp_[cmp_.chart_bond > 0.05]
    print(f"chart fit (bond-held months): corr gross={cmp_.chart_total.corr(cmp_.model_total):.3f} "
          f"bond={cmp_.chart_bond.corr(cmp_.model_bond):.3f}")
    plot(res, cmp_, OUT / "final_vs_target.png")

    last, sig = current_signals(rets, res)
    print(f"\nsignals as of {last.date()}:\n{sig.round(4).to_string()}")
    print("\nrebalance dates (20 SSE days, phase 2026-09-24):", ", ".join(str(d.date()) for d in upcoming_rebalances()))
    print("\nlast rebalances:\n", res.reb_weights.tail(4).round(3).to_string())
    if len(res.stops):
        print("\nstops since 2026-08:\n", res.stops[res.stops.date >= "2026-08-01"].to_string(index=False))


if __name__ == "__main__":
    main()
