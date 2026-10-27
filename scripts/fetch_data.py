"""Download raw data from tushare into data/raw/*.parquet (cached; --force to refetch).

Token is read from env TUSHARE_TOKEN.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
START, END = "20040101", pd.Timestamp.today().strftime("%Y%m%d")

INDEX_DAILY = ["000300.SH", "H00300.CSI", "930849.CSI", "930916.CSI", "931310.CSI", "H11077.CSI"]
FUT_MAIN = ["AU.SHF", "CU.SHF", "SC.INE", "FU.SHF", "T.CFX", "TF.CFX"]
YC_TERMS = [1, 5, 7, 10]

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])


def call(fn, retries=5, **kw):
    for i in range(retries):
        try:
            return getattr(pro, fn)(**kw)
        except Exception as e:  # rate limit / transient network
            wait = 3 * (i + 1)
            print(f"  {fn} {kw.get('ts_code', '')} err: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{fn} failed: {kw}")


def chunked(fn, years=4, start=START, end=END, **kw):
    """Call a date-ranged API in year chunks to stay under per-call row limits."""
    out = []
    s = pd.Timestamp(start)
    e_all = pd.Timestamp(end)
    while s <= e_all:
        e = min(s + pd.DateOffset(years=years) - pd.Timedelta(days=1), e_all)
        df = call(fn, start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"), **kw)
        if df is not None and len(df):
            out.append(df)
        s = e + pd.Timedelta(days=1)
    return pd.concat(out, ignore_index=True).drop_duplicates() if out else pd.DataFrame()


def save(df, name):
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW / f"{name}.parquet", index=False)
    print(f"saved {name}: {len(df)} rows")


def cached(name, force):
    return (RAW / f"{name}.parquet").exists() and not force


def fetch_futures(code, force):
    name = f"fut_{code.replace('.', '_')}"
    if cached(name, force):
        return
    mp = chunked("fut_mapping", years=8, ts_code=code)
    mp = mp.sort_values("trade_date")
    rows = []
    for c, g in mp.groupby("mapping_ts_code"):
        df = call("fut_daily", ts_code=c, start_date=g.trade_date.min(), end_date=g.trade_date.max())
        if df is not None and len(df):
            rows.append(df)
        time.sleep(0.12)
    daily = pd.concat(rows, ignore_index=True)
    out = mp.merge(daily, left_on=["mapping_ts_code", "trade_date"], right_on=["ts_code", "trade_date"],
                   how="left", suffixes=("_main", ""))
    out = out.drop(columns=["ts_code"]).rename(columns={"ts_code_main": "ts_code"})
    save(out, name)


def main(force=False):
    if not cached("trade_cal", force):
        save(call("trade_cal", exchange="SSE", start_date=START, end_date="20271231"), "trade_cal")
    for c in INDEX_DAILY:
        name = f"idx_{c.replace('.', '_')}"
        if not cached(name, force):
            save(chunked("index_daily", years=10, ts_code=c), name)
    if not cached("spx", force):
        save(chunked("index_global", years=10, ts_code="SPX"), "spx")
    if not cached("usdcnh", force):
        save(chunked("fx_daily", years=10, ts_code="USDCNH.FXCM"), "usdcnh")
    if not cached("shibor", force):
        save(chunked("shibor", years=5), "shibor")
    if not cached("yc_cb", force):
        parts = [chunked("yc_cb", years=6, ts_code="1001.CB", curve_type="0", curve_term=t) for t in YC_TERMS]
        save(pd.concat(parts, ignore_index=True), "yc_cb")
    for code in FUT_MAIN:
        fetch_futures(code, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
