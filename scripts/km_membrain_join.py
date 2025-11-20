#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_MEMBRAIN_QTY = Path("data/salescogs/MaxPullFromKhurramMembrainProductbyQTY&Period.csv")
DEFAULT_MASTER_MAP = Path("data/membrainmap/kmmembrain_mastermap.csv")
DEFAULT_RATES_TIDY = Path("data/tidy/rates_tidy.csv")
OUT_MEMBRAIN = Path("out/salescogs/kmmembrainjoined.csv")
OUT_GROUPED = Path("out/groupedplantrates.csv")
OUT_MERGED = Path("out/salescogs/merged_kmforecast_rates.csv")
OUT_LOG = Path("out/logs/run_summary_km.json")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def parse_period_to_key(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    cleaned = series.fillna("").astype(str).str.strip()
    dt = pd.to_datetime(cleaned.str.replace("/", "-", regex=False), format="%Y-%m", errors="coerce")
    key = dt.dt.strftime("%Y-%m")
    return dt, key


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Metrics:
    membrain_qty_rows: int = 0
    master_map_rows: int = 0
    rates_rows: int = 0
    membrain_joined_rows: int = 0
    grouped_rates_rows: int = 0
    merged_rows: int = 0
    zero_groups_adjusted: int = 0
    rows_rate_changed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def load_membrain_qty(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    lower_cols = {col.lower(): col for col in df.columns}

    def find_col(substr: str) -> str:
        for key, orig in lower_cols.items():
            if substr in key:
                return orig
        raise KeyError(f"Expected column containing '{substr}' in {path}")

    period_col = find_col("period")
    qty_col = find_col("qty")
    product_col = find_col("productmesh")

    df = df.rename(
        columns={
            period_col: "Period",
            qty_col: "QTY",
            product_col: "MembrainProductMesh",
        }
    )

    df["MembrainProductMesh"] = df["MembrainProductMesh"].astype("string").fillna("")
    qty_clean = df["QTY"].astype("string").str.replace(",", "", regex=False).str.strip()
    df["QTY"] = pd.to_numeric(qty_clean, errors="coerce").fillna(0.0)
    _, period_key = parse_period_to_key(df["Period"])
    df["PeriodKey"] = period_key
    df["MembrainProductMesh_norm"] = normalize_key(df["MembrainProductMesh"])
    return df


def load_master_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"MembrainProductMesh": "string", "ProductMesh": "string"},
        usecols=["MembrainProductMesh", "ProductMesh"],
    )
    df["MembrainProductMesh_norm"] = normalize_key(df["MembrainProductMesh"])
    df["ProductMesh_norm"] = normalize_key(df["ProductMesh"])
    return df


def build_membrain_join(mem_qty: pd.DataFrame, master_map: pd.DataFrame) -> pd.DataFrame:
    joined = mem_qty.merge(
        master_map[["MembrainProductMesh_norm", "ProductMesh"]],
        on="MembrainProductMesh_norm",
        how="left",
        suffixes=("", "_map"),
    )
    joined = joined.rename(columns={"ProductMesh": "PlantProductMesh"})
    joined["PlantProductMesh"] = joined["PlantProductMesh"].fillna(joined["MembrainProductMesh"])
    result = joined[["PlantProductMesh", "MembrainProductMesh", "PeriodKey", "QTY"]].copy()
    result["PlantProductMesh_norm"] = normalize_key(result["PlantProductMesh"])
    return result


def load_and_group_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "product": "string",
            "plant": "string",
            "element_code": "string",
            "period_text": "string",
            "rate": "float64",
        },
        usecols=["product", "plant", "element_code", "period_text", "rate"],
    )
    _, period_key = parse_period_to_key(df["period_text"])
    df["PeriodKey"] = period_key
    grouped = (
        df.groupby(["product", "plant", "element_code", "PeriodKey"], dropna=False)["rate"]
        .sum(min_count=1)
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"product": "PlantProductMesh", "plant": "Plant", "rate": "Rate"}
    )
    grouped["PlantProductMesh_norm"] = normalize_key(grouped["PlantProductMesh"])
    if grouped[["PlantProductMesh", "element_code", "PeriodKey"]].isna().any().any():
        raise ValueError("Nulls detected in key columns after grouping rates.")
    return grouped


def apply_carry_forward(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    # For KM variant: keep original rates, no carry-forward/back adjustments.
    df = df.copy()
    df["rate_fill_flag"] = "original"
    zero_groups_adjusted = 0
    rows_changed = 0
    df = df.sort_values(["PlantProductMesh", "element_code", "PeriodKey"])
    return df, zero_groups_adjusted, rows_changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Join MaxPull Membrain forecasts with plant rates.")
    parser.add_argument("--membrain-qty", default=str(DEFAULT_MEMBRAIN_QTY))
    parser.add_argument("--master-map", default=str(DEFAULT_MASTER_MAP))
    parser.add_argument("--rates", default=str(DEFAULT_RATES_TIDY))
    parser.add_argument("--out-membrain", default=str(OUT_MEMBRAIN))
    parser.add_argument("--out-grouped", default=str(OUT_GROUPED))
    parser.add_argument("--out-merged", default=str(OUT_MERGED))
    parser.add_argument("--log", default=str(OUT_LOG))
    args = parser.parse_args()

    start_ts = datetime.utcnow().isoformat()
    metrics = Metrics()

    membrain_qty = load_membrain_qty(Path(args.membrain_qty))
    metrics.membrain_qty_rows = len(membrain_qty)

    master_map = load_master_map(Path(args.master_map))
    metrics.master_map_rows = len(master_map)

    rates_grouped = load_and_group_rates(Path(args.rates))
    metrics.rates_rows = len(rates_grouped)

    membrain_joined = build_membrain_join(membrain_qty, master_map)
    metrics.membrain_joined_rows = len(membrain_joined)

    ensure_parent(Path(args.out_membrain))
    membrain_joined[["PlantProductMesh", "MembrainProductMesh", "PeriodKey", "QTY"]].to_csv(args.out_membrain, index=False)

    grouped = rates_grouped.copy()
    metrics.grouped_rates_rows = len(grouped)
    grouped[["Plant", "PlantProductMesh", "element_code", "PeriodKey", "Rate"]].to_csv(args.out_grouped, index=False)

    merged = grouped.merge(
        membrain_joined[["PlantProductMesh_norm", "PeriodKey", "QTY", "MembrainProductMesh"]],
        left_on=["PlantProductMesh_norm", "PeriodKey"],
        right_on=["PlantProductMesh_norm", "PeriodKey"],
        how="left",
    )
    merged = merged.rename(columns={"QTY": "Forecasted_Qty"})
    merged["Forecasted_Qty"] = pd.to_numeric(merged["Forecasted_Qty"], errors="coerce")

    merged, zero_groups, rows_changed = apply_carry_forward(merged)
    metrics.zero_groups_adjusted = zero_groups
    metrics.rows_rate_changed = rows_changed
    metrics.merged_rows = len(merged)

    merged["Cost"] = merged["Rate"] * merged["Forecasted_Qty"]

    merged = merged[
        [
            "Plant",
            "PlantProductMesh",
            "element_code",
            "PeriodKey",
            "Rate",
            "Cost",
            "rate_fill_flag",
            "Forecasted_Qty",
            "MembrainProductMesh",
        ]
    ]

    ensure_parent(Path(args.out_merged))
    merged.to_csv(args.out_merged, index=False)

    ensure_parent(Path(args.log))
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "start_ts": start_ts,
                "end_ts": datetime.utcnow().isoformat(),
                "metrics": metrics.to_dict(),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
