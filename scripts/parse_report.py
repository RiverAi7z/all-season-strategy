"""Parse targets out of the monthly report text (pdftotext -layout output).

Outputs:
  data/target/index_monthly.csv   CI011001.WI monthly returns 2009-01 .. 2026-09
  data/target/index_yearly.csv    yearly stats (ret, mdd, vol, sharpe) as printed
  data/target/attr_p1.csv         product No.1 monthly attribution by asset class
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/target/report_2026_09.txt"
PCT = r"-?\d+(?:\.\d+)?%"


def pct(s: str) -> float:
    return float(s.rstrip("%")) / 100


def parse_index(lines):
    monthly, yearly = [], []
    for ln in lines:
        m = re.match(r"^(20\d{2})\s+(.*)$", ln)
        if not m:
            continue
        year = int(m.group(1))
        vals = re.findall(PCT + r"|\b\d+\.\d+\b", m.group(2))
        if len(vals) < 5:
            continue
        # last 4 tokens: annual ret, mdd, vol, sharpe
        *months, ann, mdd, vol, sharpe = vals
        for i, v in enumerate(months, start=1):
            monthly.append({"month": f"{year}-{i:02d}", "ret": pct(v)})
        yearly.append({"year": year, "ret": pct(ann), "mdd": pct(mdd), "vol": pct(vol), "sharpe": float(sharpe)})
        if year == 2026:
            break
    return pd.DataFrame(monthly), pd.DataFrame(yearly)


def parse_attr(lines, start_marker):
    cols = ["cn_eq", "us_eq", "gold", "metal", "energy", "cash", "cn_bond", "fee", "total"]
    out, on = [], False
    for ln in lines:
        if start_marker in ln:
            on = True
            continue
        if not on:
            continue
        m = re.match(r"^\s*(20\d{2})年(\d{1,2})月\s+(.*)$", ln)
        if m:
            vals = re.findall(PCT, m.group(3))
            if len(vals) == len(cols):
                out.append({"month": f"{m.group(1)}-{int(m.group(2)):02d}", **dict(zip(cols, map(pct, vals)))})
        elif "成立以来" in ln:
            break
    return pd.DataFrame(out)


def main():
    lines = TXT.read_text().splitlines()
    monthly, yearly = parse_index(lines)
    monthly.to_csv(ROOT / "data/target/index_monthly.csv", index=False)
    yearly.to_csv(ROOT / "data/target/index_yearly.csv", index=False)
    attr = parse_attr(lines, "大类资产全天候1号资产管理计划")
    attr.to_csv(ROOT / "data/target/attr_p1.csv", index=False)
    print(f"index months={len(monthly)} ({monthly.month.iloc[0]}..{monthly.month.iloc[-1]}), years={len(yearly)}")
    print(f"attr_p1 months={len(attr)} ({attr.month.iloc[0]}..{attr.month.iloc[-1]})")
    # sanity: compounded monthly vs printed annual
    monthly["year"] = monthly.month.str[:4].astype(int)
    chk = monthly.groupby("year").ret.apply(lambda r: (1 + r).prod() - 1).rename("compounded")
    chk = yearly.set_index("year").join(chk)
    bad = chk[(chk.ret - chk.compounded).abs() > 0.003]
    if len(bad):
        print("WARN annual mismatch:\n", bad, file=sys.stderr)


if __name__ == "__main__":
    main()
