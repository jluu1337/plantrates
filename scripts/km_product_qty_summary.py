#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MERGED = Path("out/salescogs/merged_kmforecast_rates.csv")
DEFAULT_OUT = Path("out/salescogs/km_product_qty_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize KM forecast quantity by plant/product/period without element double-counting."
    )
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.merged)
    df["Forecasted_Qty"] = pd.to_numeric(df["Forecasted_Qty"], errors="coerce").fillna(0.0)

    summary = (
        df.groupby(["Plant", "PlantProductMesh", "PeriodKey"], dropna=False)["Forecasted_Qty"]
        .max(min_count=1)
        .reset_index(name="prod_qty")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {len(summary)} rows to {args.out}")


if __name__ == "__main__":
    main()
