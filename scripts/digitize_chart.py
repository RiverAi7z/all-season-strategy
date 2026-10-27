"""Digitize the stacked weight bar chart (report p.3) into data/target/chart_weights.csv."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"eq": (91, 155, 213), "bond": (112, 173, 71), "cmd": (165, 165, 165), "cash": (255, 192, 0)}


def classify(px):
    d = {k: np.abs(px[..., :3].astype(int) - np.array(c)).sum(-1) for k, c in COLORS.items()}
    stack = np.stack(list(d.values()), -1)
    lab = stack.argmin(-1)
    lab[stack.min(-1) > 60] = -1
    return lab


def main():
    img = np.array(Image.open(ROOT / "data/target/weights_chart.png").convert("RGB"))
    lab = classify(img)
    colored = lab >= 0
    # baseline: lowest row with many colored pixels
    rows = colored.sum(1)
    base = int(np.where(rows > 200)[0].max())
    cols = colored[base - 3]
    edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
    bars = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a > 10]
    months = pd.period_range("2023-02", "2026-09", freq="M")
    assert len(bars) == len(months), (len(bars), len(months))
    # scale from the bars that top out at exactly 100% (bond == 0 bars): find their tops
    heights = []
    out = []
    for (a, b), m in zip(bars, months):
        c = (a + b) // 2
        col = lab[: base + 1, c - 3: c + 4]
        counts = {k: int(np.median((col == i).sum(0))) for i, k in enumerate(COLORS)}
        heights.append(sum(counts.values()))
        out.append({"month": str(m), **counts})
    df = pd.DataFrame(out)
    tot = df[list(COLORS)].sum(1)
    no_bond = df.bond < 5
    px_per_100 = np.median(tot[no_bond])
    df[list(COLORS)] = df[list(COLORS)] / px_per_100
    df["total"] = df[list(COLORS)].sum(1)
    df.round(4).to_csv(ROOT / "data/target/chart_weights.csv", index=False)
    print(f"baseline y={base}, px per 100% = {px_per_100}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
