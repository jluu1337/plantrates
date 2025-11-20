#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_MEM_MERGED = Path("out/salescogs/merged_forecast_rates.csv")
DEFAULT_KM_MERGED = Path("out/salescogs/merged_kmforecast_rates.csv")
DEFAULT_OUT = Path("out/salescogs/qty_summary.csv")


def summarize(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Forecasted_Qty"] = pd.to_numeric(df["Forecasted_Qty"], errors="coerce").fillna(0.0)
    grouped = (
        df.groupby(["Plant", "PlantProductMesh", "PeriodKey"], dropna=False)["Forecasted_Qty"]
        .sum(min_count=1)
        .reset_index()
    )
    grouped.insert(0, "source", source)
    grouped = grouped.rename(columns={"Forecasted_Qty": "qty"})
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize forecast quantities by plant/product/period for Membrain and KM."
    )
    parser.add_argument("--mem-merged", type=Path, default=DEFAULT_MEM_MERGED)
    parser.add_argument("--km-merged", type=Path, default=DEFAULT_KM_MERGED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summaries: List[pd.DataFrame] = []
    if Path(args.mem_merged).exists():
        summaries.append(summarize(Path(args.mem_merged), "membrain"))
    if Path(args.km_merged).exists():
        summaries.append(summarize(Path(args.km_merged), "km"))

    if not summaries:
        raise FileNotFoundError("No merged forecast files found to summarize.")

    result = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
