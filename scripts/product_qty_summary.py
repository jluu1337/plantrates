#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TIDY = Path("data/tidy/rates_tidy.csv")
DEFAULT_OUT = Path("out/salescogs/product_qty_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize production quantity by plant/product/period without element double-counting."
    )
    parser.add_argument("--tidy", type=Path, default=DEFAULT_TIDY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.tidy)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0) * 1000.0

    # Collapse element duplication: one quantity per plant/product/period
    summary = (
        df.groupby(["plant", "product", "period_text"], dropna=False)["qty"]
        .max(min_count=1)
        .reset_index(name="prod_qty")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {len(summary)} rows to {args.out}")


if __name__ == "__main__":
    main()
