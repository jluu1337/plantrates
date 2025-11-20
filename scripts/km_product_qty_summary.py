#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MEMBRAIN_QTY = Path("data/salescogs/MaxPullFromKhurramMembrainProductbyQTY&Period.csv")
DEFAULT_MASTER_MAP = Path("data/membrainmap/kmmembrain_mastermap.csv")
DEFAULT_OUT = Path("out/salescogs/km_product_qty_summary.csv")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def parse_period(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    dt = pd.to_datetime(cleaned.str.replace("/", "-", regex=False), format="%Y-%m", errors="coerce")
    return dt.dt.strftime("%Y-%m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize KM forecast quantities by PlantProductMesh and Period from the source MaxPull file."
    )
    parser.add_argument("--membrain-qty", type=Path, default=DEFAULT_MEMBRAIN_QTY)
    parser.add_argument("--master-map", type=Path, default=DEFAULT_MASTER_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    qty_df = pd.read_csv(args.membrain_qty, dtype="string")
    map_df = pd.read_csv(args.master_map, dtype="string", usecols=["MembrainProductMesh", "ProductMesh"])

    lower_cols = {col.lower(): col for col in qty_df.columns}

    def find_col(substr: str) -> str:
        for key, orig in lower_cols.items():
            if substr in key:
                return orig
        raise KeyError(f"Expected column containing '{substr}' in {args.membrain_qty}")

    period_col = find_col("period")
    product_col = find_col("productmesh")
    qty_col = find_col("qty")

    qty_df = qty_df.rename(columns={period_col: "Period", product_col: "MembrainProductMesh", qty_col: "QTY"})
    qty_df["PeriodKey"] = parse_period(qty_df["Period"])
    qty_df["MembrainProductMesh_norm"] = normalize_key(qty_df["MembrainProductMesh"])
    qty_clean = qty_df["QTY"].astype("string").str.replace(",", "", regex=False).str.strip()
    qty_df["QTY"] = pd.to_numeric(qty_clean, errors="coerce").fillna(0.0)

    map_df["MembrainProductMesh_norm"] = normalize_key(map_df["MembrainProductMesh"])
    joined = qty_df.merge(
        map_df[["MembrainProductMesh_norm", "ProductMesh"]],
        on="MembrainProductMesh_norm",
        how="left",
    )
    joined["PlantProductMesh"] = joined["ProductMesh"].fillna(joined["MembrainProductMesh"])

    summary = (
        joined.groupby(["PlantProductMesh", "PeriodKey"], dropna=False)["QTY"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"QTY": "prod_qty"})
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {len(summary)} rows to {args.out}")


if __name__ == "__main__":
    main()
