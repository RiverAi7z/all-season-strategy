"""Compare FINAL (replication) with MACRO (US real-rate overlay); print current macro readings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allweather.config import DATA, FINAL, MACRO, MACRO_WINDOW  # noqa: E402
from allweather.data import asset_returns  # noqa: E402
from allweather.engine import run  # noqa: E402
from allweather.macro import factor_panel, us_real_rate_z, us_real_scores  # noqa: E402
from allweather.metrics import compare, monthly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def stats(r: pd.Series, lo, hi) -> dict:
    r = r.loc[lo:hi]
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": (ann - 0.02) / vol, "mdd": mdd, "calmar": ann / -mdd}


def main():
    rets = asset_returns(**DATA)
    base = run(FINAL, rets)
    mac = run(MACRO, rets, scores=us_real_scores(MACRO_WINDOW))
    rows = []
    for lab, lo, hi in [("2009-2017 样本内", "2009", "2017"), ("2018-2026 样本外", "2018", "2026-09"), ("全样本", "2009", "2026-09")]:
        for name, r in [("复刻（无宏观）", base.daily.ret), ("宏观叠加", mac.daily.ret)]:
            rows.append({"period": lab, "model": name, **stats(r, lo, hi)})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    print("fit vs index (2023-02+): base", round(compare(base.daily.ret, start="2023-02")["corr"], 3),
          "macro", round(compare(mac.daily.ret, start="2023-02")["corr"], 3))

    z = us_real_rate_z(MACRO_WINDOW)
    m_base, m_mac = monthly(base.daily.ret), monthly(mac.daily.ret)
    out = pd.DataFrame({"base_ret": m_base, "macro_ret": m_mac,
                        "us_real_z": z.groupby(z.index.to_period("M")).last()}).dropna(subset=["base_ret"])
    out.index = out.index.astype(str)
    out.index.name = "month"
    out.loc[:"2026-09"].round(6).to_csv(OUT / "macro_monthly.csv")
    tab.to_csv(OUT / "macro_vs_base.csv", index=False)
    mac.reb_weights.to_csv(OUT / "macro_rebalance_weights.csv")

    last = rets.index[-1]
    print(f"\nmacro readings as of {last.date()}:")
    print(factor_panel().iloc[-1].round(2).to_string())
    print(f"US 10y real-yield {MACRO_WINDOW}d change z = {z.iloc[-1]:.2f}  "
          f"(> {MACRO.macro_veto} excludes gold and US equities at the next rebalance; budget x{np.exp(-MACRO.macro_k * z.iloc[-1]):.2f})")
    print("\nlast macro-overlay rebalances:\n", mac.reb_weights.tail(4).round(3).to_string())


if __name__ == "__main__":
    main()
