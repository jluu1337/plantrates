#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TIDY = Path("data/tidy/rates_tidy.csv")
DEFAULT_PRODUCT_MAP = Path("configs/product_map.csv")
DEFAULT_OUT = Path("out/logs/orphan_products.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="List products in rates_tidy that are not present in product_map.")
    parser.add_argument("--tidy", type=Path, default=DEFAULT_TIDY)
    parser.add_argument("--product-map", type=Path, default=DEFAULT_PRODUCT_MAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tidy = pd.read_csv(args.tidy, dtype="string")
    product_map = pd.read_csv(args.product_map, dtype="string")

    mapped_keys = set(product_map["product"].str.upper().str.strip())
    tidy["product_key"] = tidy["product"].str.upper().str.strip()

    orphan_df = tidy[~tidy["product_key"].isin(mapped_keys)].copy()
    summary = (
        orphan_df.groupby("product", dropna=False)
        .agg(
            rows=("product", "size"),
            plants=("plant", lambda s: ",".join(sorted(set(s.dropna())))),
            periods=("period", lambda s: ",".join(sorted(set(s.dropna())))),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {len(summary)} orphan products to {args.out}")


if __name__ == "__main__":
    main()
