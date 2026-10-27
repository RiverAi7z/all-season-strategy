"""Final replication spec (see README for how each piece was identified)."""
from .engine import Params

DATA = dict(us_ccy="usd", bond="cdb10", oil_proxy=None, eq="tr", cash_tenor="3m")

FINAL = Params(
    reb_period=20, anchor="2026-09-24",          # every 20 SSE trading days, phase from 2026-09-24
    cov=True, vol_window=250, budget="asset",     # ERC, equal risk per asset, 1y covariance
    lev_window=1000, target_vol=0.055, max_gross=2.0,  # leverage from ~4y covariance, cap 2x
    mom_window=45, mom_type="ma",                 # rebalance filter: price above MA45
    stop="ma", stop_window=60,                    # intra-period stop: close below MA60, out until next rebalance
    cash_mode="residual",
)

# Alternative stop that fits the full 2009+ history slightly better (worse after 2023)
ALT_CUM = Params(**{**FINAL.__dict__, "stop": "cum", "stop_k": 1.0, "stop_window": 20})

# Candidate macro overlay, NOT the default (README "宏观叠加"): chosen after seeing 2018+ results, and the
# year-by-year walk-forward test shows no Sharpe gain (1.817 vs 1.809) with a deeper drawdown.
# US 10y real-yield change over 120 days, z-scored; gold / US-equity budgets *= exp(-0.5 z), excluded when z > 1.
MACRO = Params(**{**FINAL.__dict__, "macro_k": 0.5, "macro_veto": 1.0})
MACRO_WINDOW = 120
