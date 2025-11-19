#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MEMBRAIN_QTY = Path("data/salescogs/MembrainProductbyQTY&Period.csv")
DEFAULT_FACAD_MAP = Path("data/membrainmap/membrain_facadmap.csv")
DEFAULT_FACAD_COST = Path("src/FACAD/Product Cost File  Oct 2025-2026 final-V3.xlsx")
DEFAULT_OUTPUT = Path("out/salescogs/membrainfacadrates.csv")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def find_column(columns: pd.Index, needle: str) -> str:
    needle = needle.lower()
    for col in columns:
        if needle in str(col).lower():
            return col
    raise KeyError(f"Unable to locate column containing '{needle}' in {list(columns)}")


def load_membrain_qty(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    period_col = find_column(df.columns, "period")
    product_col = find_column(df.columns, "product")
    qty_col = find_column(df.columns, "qty")

    df = df.rename(
        columns={
            period_col: "Period",
            product_col: "MembrainProduct",
            qty_col: "QTY",
        }
    )
    df["QTY"] = pd.to_numeric(df["QTY"], errors="coerce").fillna(0.0)
    df["MembrainProduct_norm"] = normalize_key(df["MembrainProduct"])
    return df[["Period", "MembrainProduct", "MembrainProduct_norm", "QTY"]]


def load_facad_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    df = df.rename(columns={"MembrainProductMesh": "MembrainProductMap", "ProductMap": "Product"})
    df["MembrainProduct_norm"] = normalize_key(df["MembrainProductMap"])
    df["Product_norm"] = normalize_key(df["Product"])
    return df[["MembrainProduct_norm", "Product", "Product_norm"]]


def load_facad_cost(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype="object")
    df = df.rename(columns=lambda col: str(col).strip())
    required = ["Product", "Facad Adder", "Plt"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns {missing} in {path}")
    df = df[required].copy()
    df["Product_norm"] = normalize_key(df["Product"])
    df["Facad Adder"] = pd.to_numeric(df["Facad Adder"], errors="coerce")
    return df


def build_facad_rates(membrain_qty: pd.DataFrame, facad_map: pd.DataFrame, facad_cost: pd.DataFrame) -> pd.DataFrame:
    mapped = membrain_qty.merge(facad_map, on="MembrainProduct_norm", how="left")
    mapped["Product"] = mapped["Product"].fillna(mapped["MembrainProduct"])
    mapped["Product_norm"] = normalize_key(mapped["Product"])

    joined = mapped.merge(facad_cost, on="Product_norm", how="left", suffixes=("", "_facad"))
    joined["rate"] = pd.to_numeric(joined["Facad Adder"], errors="coerce").fillna(0.0)
    joined["plant"] = joined["Plt"].astype("string").fillna("")
    joined["cost"] = joined["rate"] * joined["QTY"]

    result = joined.rename(
        columns={
            "MembrainProduct": "membrain_product",
            "QTY": "qty",
            "Period": "period",
            "Product": "product",
        }
    )
    return result[["membrain_product", "qty", "period", "product", "plant", "rate", "cost"]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce FACAD rate joins for Membrain quantities.")
    parser.add_argument("--membrain-qty", default=str(DEFAULT_MEMBRAIN_QTY))
    parser.add_argument("--facad-map", default=str(DEFAULT_FACAD_MAP))
    parser.add_argument("--facad-cost", default=str(DEFAULT_FACAD_COST))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    membrain_qty = load_membrain_qty(Path(args.membrain_qty))
    facad_map = load_facad_map(Path(args.facad_map))
    facad_cost = load_facad_cost(Path(args.facad_cost))

    facad_rates = build_facad_rates(membrain_qty, facad_map, facad_cost)
    ensure_parent(Path(args.out))
    facad_rates.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
