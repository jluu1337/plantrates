#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_MEMBRAIN = Path("out/salescogs/merged_forecast_rates.csv")
DEFAULT_KM = Path("out/salescogs/merged_kmforecast_rates.csv")
DEFAULT_OUT = Path("out/salescogs/product_context_summary.csv")


def summarize(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Forecasted_Qty"] = pd.to_numeric(df["Forecasted_Qty"], errors="coerce").fillna(0.0)
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(0.0)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["Plant", "PlantProductMesh", "PeriodKey"], dropna=False)
        .agg(
            qty_lb=("Forecasted_Qty", "sum"),
            revenue=("Cost", "sum"),
            avg_rate=("Rate", "mean"),
        )
        .reset_index()
    )
    grouped["price_lb"] = grouped.apply(
        lambda row: row["revenue"] / row["qty_lb"] if row["qty_lb"] else row["avg_rate"],
        axis=1,
    )
    grouped["cost_lb"] = grouped["price_lb"]  # alias for readability
    grouped = grouped.drop(columns=["avg_rate"])
    grouped.insert(0, "source", source)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize product context for Membrain and KM Membrain outputs.")
    parser.add_argument("--membrain", type=Path, default=DEFAULT_MEMBRAIN)
    parser.add_argument("--km-membrain", type=Path, default=DEFAULT_KM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summaries: List[pd.DataFrame] = []
    if Path(args.membrain).exists():
        summaries.append(summarize(Path(args.membrain), "membrain"))
    if Path(args.km_membrain).exists():
        summaries.append(summarize(Path(args.km_membrain), "km"))

    if not summaries:
        raise FileNotFoundError("No input files found for summarization.")

    result = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
"""  """