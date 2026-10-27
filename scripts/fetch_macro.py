"""Download macro series from tushare into data/raw/macro_*.parquet (cached; --force to refetch)."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import RAW, cached, call, chunked, save  # noqa: E402

START_M, END_M = "200401", pd.Timestamp.today().strftime("%Y%m")

MONTHLY = {
    "macro_pmi": ("cn_pmi", ["month", "pmi010000", "pmi010500", "pmi010700", "pmi011200", "pmi011300", "pmi011400", "pmi020100"]),
    "macro_cpi": ("cn_cpi", ["month", "nt_yoy", "nt_mom"]),
    "macro_ppi": ("cn_ppi", ["month", "ppi_yoy", "ppi_mom"]),
    "macro_m": ("cn_m", ["month", "m1_yoy", "m2_yoy"]),
    "macro_sf": ("sf_month", ["month", "inc_month", "stk_endval"]),
}


def main(force=False):
    for name, (fn, fields) in MONTHLY.items():
        if cached(name, force):
            continue
        df = call(fn, start_m=START_M, end_m=END_M, fields=",".join(fields))
        save(df.sort_values("month"), name)
    if not cached("macro_us_real", force):
        save(chunked("us_trycr", years=4, start="20040101", fields="date,y5,y10"), "macro_us_real")
    if not cached("macro_us_nominal", force):
        save(chunked("us_tycr", years=4, start="20040101", fields="date,m3,y2,y10"), "macro_us_nominal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
