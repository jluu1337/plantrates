#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_MEM_QTY = Path("data/salescogs/MembrainProductbyQTY&Period.csv")
DEFAULT_KM_QTY = Path("data/salescogs/MaxPullFromKhurramMembrainProductbyQTY&Period.csv")
DEFAULT_MEM_MAP = Path("data/membrainmap/membrain_mastermap.csv")
DEFAULT_KM_MAP = Path("data/membrainmap/kmmembrain_mastermap.csv")
DEFAULT_MEM_COST = Path("out/salescogs/merged_forecast_rates.csv")
DEFAULT_KM_COST = Path("out/salescogs/merged_kmforecast_rates.csv")
DEFAULT_OUT = Path("out/salescogs/product_cost_summary.csv")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def parse_period_to_key(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    cleaned = series.fillna("").astype(str).str.strip()
    dt = pd.to_datetime(cleaned.str.replace("/", "-", regex=False), format="%Y-%m", errors="coerce")
    key = dt.dt.strftime("%Y-%m")
    return dt, key


def load_qty(path: Path, master_map: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    lower_cols = {col.lower(): col for col in df.columns}

    def find_col(substr: str) -> str:
        for key, orig in lower_cols.items():
            if substr in key:
                return orig
        raise KeyError(f"Expected column containing '{substr}' in {path}")

    period_col = find_col("period")
    product_col = find_col("product")
    qty_col = find_col("qty")

    df = df.rename(columns={period_col: "Period", product_col: "MembrainProductMesh", qty_col: "QTY"})
    qty_clean = df["QTY"].astype("string").str.replace(",", "", regex=False).str.strip()
    df["QTY"] = pd.to_numeric(qty_clean, errors="coerce").fillna(0.0)
    _, period_key = parse_period_to_key(df["Period"])
    df["PeriodKey"] = period_key
    df["MembrainProductMesh_norm"] = normalize_key(df["MembrainProductMesh"])

    map_df = pd.read_csv(master_map, dtype="string", usecols=["MembrainProductMesh", "ProductMesh"])
    map_df["MembrainProductMesh_norm"] = normalize_key(map_df["MembrainProductMesh"])
    joined = df.merge(
        map_df[["MembrainProductMesh_norm", "ProductMesh"]],
        on="MembrainProductMesh_norm",
        how="left",
    )
    joined["PlantProductMesh"] = joined["ProductMesh"].fillna(joined["MembrainProductMesh"])
    joined["PlantProductMesh_norm"] = normalize_key(joined["PlantProductMesh"])
    qty_agg = (
        joined.groupby(["PlantProductMesh_norm", "PlantProductMesh", "PeriodKey"], dropna=False)["QTY"]
        .sum(min_count=1)
        .reset_index()
    )
    return qty_agg


def load_costs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["PlantProductMesh_norm"] = normalize_key(df["PlantProductMesh"])
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").fillna(0.0)
    cost_agg = (
        df.groupby(["Plant", "PlantProductMesh", "PlantProductMesh_norm", "PeriodKey"], dropna=False)["Cost"]
        .sum(min_count=1)
        .reset_index()
    )
    return cost_agg


def build_summary(qty_path: Path, map_path: Path, cost_path: Path, source: str) -> pd.DataFrame:
    qty = load_qty(qty_path, map_path)
    cost = load_costs(cost_path)
    merged = cost.merge(
        qty[["PlantProductMesh_norm", "PeriodKey", "QTY"]],
        on=["PlantProductMesh_norm", "PeriodKey"],
        how="left",
    )
    merged["qty_lb"] = merged["QTY"].fillna(0.0)
    merged["cost"] = merged["Cost"]
    merged["cost_lb"] = merged.apply(
        lambda row: row["cost"] / row["qty_lb"] if row["qty_lb"] else None,
        axis=1,
    )
    merged = merged.drop(columns=["QTY", "Cost"])
    merged.insert(0, "source", source)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize qty and cost by plant product and period for Membrain and KM sources.")
    parser.add_argument("--mem-qty", type=Path, default=DEFAULT_MEM_QTY)
    parser.add_argument("--km-qty", type=Path, default=DEFAULT_KM_QTY)
    parser.add_argument("--mem-map", type=Path, default=DEFAULT_MEM_MAP)
    parser.add_argument("--km-map", type=Path, default=DEFAULT_KM_MAP)
    parser.add_argument("--mem-cost", type=Path, default=DEFAULT_MEM_COST)
    parser.add_argument("--km-cost", type=Path, default=DEFAULT_KM_COST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summaries = [
        build_summary(args.mem_qty, args.mem_map, args.mem_cost, "membrain"),
        build_summary(args.km_qty, args.km_map, args.km_cost, "km"),
    ]

    result = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
