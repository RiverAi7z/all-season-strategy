"""Risk-parity / risk-budget backtest with rebalance-day momentum filter and intra-period stops.

Timing convention: everything is decided on the close of day t using data up to t and earns
returns from t+1. A stop triggered at the close of t exits at that close and the asset stays
in cash until the next scheduled rebalance.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ASSETS = ("cn_eq", "us_eq", "cn_bond", "gold", "copper", "oil")
CLASS = {"cn_eq": "eq", "us_eq": "eq", "cn_bond": "bond", "gold": "cmd", "copper": "cmd", "oil": "cmd"}


@dataclass(frozen=True)
class Params:
    assets: tuple = ASSETS
    reb_period: int = 20
    anchor: str = "2026-09-24"
    vol_window: int = 60
    cov: bool = False             # True: ERC on full covariance; False: inverse vol (diagonal)
    budget: str = "asset"         # "asset": equal per asset, "class": equal per class then split
    budget_override: tuple = ()   # ((asset, budget), ...) applied after the base scheme
    scale: str = "port"           # "port": full-universe ex-ante vol = target_vol; "asset": standalone
    target_vol: float = 0.05
    lev_window: int = 0           # window for the leverage (vol-target) covariance; 0 -> vol_window
    max_gross: float = 2.0
    max_w_nonbond: float = 1.0
    mom_window: int = 0           # rebalance filter: trailing return over window <= mom_thr -> cash
    mom_thr: float = 0.0
    mom_type: str = "ret"         # "ret": trailing return; "ma": price vs MA(mom_window)
    soft_floor: float = 0.0       # filtered assets keep this fraction of their weight instead of 0
    stop: str = "none"            # none | cum | dd | loss | mom | ma
    stop_k: float = 1.0
    stop_window: int = 20
    stop_assets: tuple = ()       # empty -> all assets
    # residual: unused/levered capital at rf; free: leverage costless;
    # free_plus: leverage costless and filtered/stopped weight parked in cash (capped by capital)
    cash_mode: str = "residual"
    cost_bps: float = 0.0
    drift: bool = True
    # macro overlay (scores passed to run()): budget *= exp(macro_k * score); score < -macro_veto -> excluded
    macro_k: float = 0.0
    macro_veto: float = 0.0


def rebalance_mask(index: pd.DatetimeIndex, period: int, anchor: str) -> np.ndarray:
    a = index.searchsorted(pd.Timestamp(anchor))
    if a >= len(index) or index[a] != pd.Timestamp(anchor):
        raise ValueError(f"anchor {anchor} is not a trading day in the return index")
    pos = np.arange(len(index))
    return ((pos - a) % period) == 0


def _budgets(p: Params, avail: list, mult: dict | None = None) -> np.ndarray:
    if p.budget == "class":
        classes = {CLASS[a] for a in avail}
        b = np.array([1 / len(classes) / sum(CLASS[x] == CLASS[a] for x in avail) for a in avail])
    else:
        b = np.full(len(avail), 1 / len(avail))
    ov = dict(p.budget_override)
    b = np.array([ov.get(a, bi) for a, bi in zip(avail, b)])
    if mult:
        b = b * np.array([mult.get(a, 1.0) for a in avail])
    return b / b.sum()


def _erc(cov: np.ndarray, b: np.ndarray, iters=200) -> np.ndarray:
    """Risk budgeting via cyclical coordinate descent: w_i (Cov w)_i proportional to b_i."""
    n = len(b)
    w = b / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        w_old = w.copy()
        for i in range(n):
            c = cov[i] @ w - cov[i, i] * w[i]
            w[i] = (-c + np.sqrt(c * c + 4 * cov[i, i] * b[i])) / (2 * cov[i, i])
        if np.max(np.abs(w - w_old)) < 1e-10:
            break
    return w / w.sum()


def target_weights(p: Params, hist: np.ndarray, names: list, hist_lev: np.ndarray | None = None,
                   mult: dict | None = None) -> np.ndarray:
    """hist: (window, n) daily returns ending today. Returns weights for all `names` (0 if n/a).
    hist_lev: optional longer history used only for the vol-target leverage."""
    ok = ~np.isnan(hist).any(axis=0)
    w = np.zeros(len(names))
    if ok.sum() == 0:
        return w
    avail = [a for a, k in zip(names, ok) if k]
    h = hist[:, ok]
    cov = np.cov(h, rowvar=False) * 252
    cov = np.atleast_2d(cov)
    vol = np.sqrt(np.diag(cov))
    b = _budgets(p, avail, mult)
    if p.scale == "asset":
        wa = b * len(b) * p.target_vol / vol      # each asset's standalone vol = budget share
    else:
        wa = _erc(cov, b) if p.cov else (b / vol) / (b / vol).sum()
        cov_lev = cov
        if hist_lev is not None:
            hl = hist_lev[:, ok]
            hl = hl[~np.isnan(hl).any(axis=1)]
            if len(hl) >= 60:
                cov_lev = np.atleast_2d(np.cov(hl, rowvar=False) * 252)
        wa = wa * p.target_vol / np.sqrt(wa @ cov_lev @ wa)
    nb = np.array([a != "cn_bond" for a in avail])
    wa[nb] = np.minimum(wa[nb], p.max_w_nonbond)
    if wa.sum() > p.max_gross:
        wa *= p.max_gross / wa.sum()
    w[ok] = wa
    return w


@dataclass
class Result:
    daily: pd.DataFrame
    weights: pd.DataFrame          # end-of-day risky weights
    reb_weights: pd.DataFrame      # target weights at each rebalance (after filter)
    reb_raw: pd.DataFrame = field(default_factory=pd.DataFrame)  # before filter
    stops: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(p: Params, rets: pd.DataFrame, start="2008-12-01", end=None, scores: pd.DataFrame | None = None) -> Result:
    names = list(p.assets)
    R = rets[names].to_numpy(float)
    rf = rets["cash"].fillna(0).to_numpy(float)
    idx = rets.index
    T, N = R.shape
    Rz = np.nan_to_num(R)
    L = np.vstack([np.zeros(N), np.cumsum(np.log1p(Rz), axis=0)])   # L[t+1] = log level at t
    reb = rebalance_mask(idx, p.reb_period, p.anchor)
    S = None
    if scores is not None and (p.macro_k or p.macro_veto):
        S = scores.reindex(idx).reindex(columns=names).fillna(0.0).to_numpy(float)
    t0 = idx.searchsorted(pd.Timestamp(start))
    t_end = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end), side="right")
    stop_set = set(p.stop_assets) if p.stop_assets else set(names)
    stoppable = np.array([a in stop_set for a in names])
    is_bond = np.array([a == "cn_bond" for a in names])

    w = np.zeros(N)
    raw_sum = 0.0
    nav = 1.0
    started = False
    entry_L = np.zeros(N)
    peak_L = np.zeros(N)
    w_entry = np.zeros(N)
    out_ret = np.zeros(T)
    out_w = np.zeros((T, N))
    reb_rows, stop_rows, raw_rows = [], [], []

    for t in range(t0, t_end):
        r = Rz[t]
        if started:
            gross = w @ r
            if p.cash_mode == "residual":
                lev_cash = 1 - w.sum()
            elif p.cash_mode == "free":
                lev_cash = max(0.0, 1 - w.sum())
            else:
                lev_cash = min(max(raw_sum - w.sum(), 0.0), max(0.0, 1 - w[~is_bond].sum()))
            pr = gross + lev_cash * rf[t]
            nav *= 1 + pr
            out_ret[t] = pr
            if p.drift:
                w = w * (1 + r) / (1 + pr)
            # intra-period stops, evaluated on the close of t
            if p.stop != "none":
                lt = L[t + 1]
                peak_L = np.maximum(peak_L, lt)
                held = (w > 0) & stoppable
                if held.any():
                    trig = np.zeros(N, bool)
                    if p.stop in ("cum", "dd", "loss"):
                        sig = np.nanstd(R[max(0, t - p.vol_window + 1): t + 1], axis=0) * np.sqrt(p.reb_period)
                        cum = np.exp(lt - entry_L) - 1
                        if p.stop == "cum":
                            trig = cum < -p.stop_k * sig
                        elif p.stop == "dd":
                            trig = np.exp(lt - peak_L) - 1 < -p.stop_k * sig
                        else:  # loss in portfolio terms, stop_k in percent of NAV
                            trig = w_entry * cum < -p.stop_k / 100
                    elif p.stop == "mom":
                        k = p.stop_window
                        trig = (lt - L[max(0, t + 1 - k)]) < 0
                    elif p.stop == "ma":
                        k = p.stop_window
                        lv = np.exp(L[max(0, t + 1 - k + 1): t + 2])
                        trig = np.exp(lt) < lv.mean(axis=0)
                    trig &= held
                    if trig.any():
                        nav_cost = np.abs(w[trig]).sum() * p.cost_bps / 1e4
                        nav *= 1 - nav_cost
                        out_ret[t] = (1 + out_ret[t]) * (1 - nav_cost) - 1
                        for i in np.where(trig)[0]:
                            stop_rows.append((idx[t], names[i], w[i]))
                        w[trig] = 0.0
        if reb[t]:
            lo = max(0, t - p.vol_window + 1)
            hist = R[lo: t + 1]
            if len(hist) >= max(20, p.vol_window // 2):
                hist_lev = R[max(0, t - p.lev_window + 1): t + 1] if p.lev_window else None
                mult = dict(zip(names, np.exp(p.macro_k * S[t]))) if S is not None and p.macro_k else None
                tw = target_weights(p, hist, names, hist_lev, mult)
                raw_sum = tw.sum()
                raw_rows.append((idx[t], *tw))
                if p.mom_window:
                    lo_m = max(0, t + 1 - p.mom_window)
                    if p.mom_type == "ma":
                        lv = np.exp(L[max(0, t + 2 - p.mom_window): t + 2])
                        bad = np.exp(L[t + 1]) <= lv.mean(axis=0) * (1 + p.mom_thr)
                    else:
                        bad = np.exp(L[t + 1] - L[lo_m]) - 1 <= p.mom_thr
                    if lo_m == 0:
                        bad[:] = False
                    tw[bad] *= p.soft_floor
                if S is not None and p.macro_veto:
                    tw[S[t] < -p.macro_veto] = 0.0
                cost = np.abs(tw - w).sum() * p.cost_bps / 1e4
                if started:
                    nav *= 1 - cost
                    out_ret[t] = (1 + out_ret[t]) * (1 - cost) - 1
                w = tw
                w_entry = tw.copy()
                entry_L = L[t + 1].copy()
                peak_L = L[t + 1].copy()
                started = True
                reb_rows.append((idx[t], *tw))
        out_w[t] = w

    sl = slice(t0, t_end)
    daily = pd.DataFrame({"ret": out_ret[sl]}, index=idx[sl])
    first = daily.index.get_loc(reb_rows[0][0]) if reb_rows else 0
    daily = daily.iloc[first:]
    daily["nav"] = (1 + daily.ret).cumprod()
    weights = pd.DataFrame(out_w[sl], index=idx[sl], columns=names).iloc[first:]
    rw = pd.DataFrame(reb_rows, columns=["date", *names]).set_index("date")
    st = pd.DataFrame(stop_rows, columns=["date", "asset", "w"])
    raw = pd.DataFrame(raw_rows, columns=["date", *names]).set_index("date")
    return Result(daily, weights, rw, reb_raw=raw, stops=st)
