#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "2025 Rates"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "2025rates"

MONTH_SHEETS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
MONTH_PERIOD_MAP = {sheet: f"2025/{idx:02d}" for idx, sheet in enumerate(MONTH_SHEETS, start=1)}


@dataclass(frozen=True)
class BlockConfig:
    product_row: int
    start_column: str
    stride: int
    rate_row: int
    qty_row: int


@dataclass(frozen=True)
class PlantConfig:
    plant: str
    filename: str
    sheet_periods: Dict[str, str]
    blocks: List[BlockConfig]


PLANT_CONFIGS: List[PlantConfig] = [
    PlantConfig(
        plant="EUF",
        filename="2025 EUF Mfg Std P85.xlsx",
        sheet_periods=dict(MONTH_PERIOD_MAP),
        blocks=[
            BlockConfig(product_row=3, start_column="B", stride=3, rate_row=22, qty_row=26),
        ],
    ),
    PlantConfig(
        plant="MCI",
        filename="2025 MCI Mfg Std P85.xlsx",
        sheet_periods=dict(MONTH_PERIOD_MAP),
        blocks=[
            BlockConfig(product_row=3, start_column="B", stride=3, rate_row=22, qty_row=24),
            BlockConfig(product_row=26, start_column="B", stride=3, rate_row=45, qty_row=47),
        ],
    ),
    PlantConfig(
        plant="NI",
        filename="2025 New Iberia RC Mfg Std P85.xlsx",
        sheet_periods={"NI RC 2025 Std": "2025FY"},
        blocks=[
            BlockConfig(product_row=3, start_column="B", stride=3, rate_row=27, qty_row=29),
            BlockConfig(product_row=34, start_column="B", stride=3, rate_row=58, qty_row=60),
        ],
    ),
    PlantConfig(
        plant="TMB",
        filename="2025 TMB Plant Std P85.xlsx",
        sheet_periods=dict(MONTH_PERIOD_MAP),
        blocks=[
            BlockConfig(product_row=3, start_column="B", stride=3, rate_row=22, qty_row=24),
            BlockConfig(product_row=26, start_column="B", stride=3, rate_row=45, qty_row=47),
        ],
    ),
]


def column_letter_to_index(letter: str) -> int:
    idx = 0
    for char in letter.upper():
        if not char.isalpha():
            raise ValueError(f"Invalid column letter '{letter}'")
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Number) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_number(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, numbers.Number):
        if pd.isna(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned in {".", "-"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def get_cell(df: pd.DataFrame, row_idx: int, col_idx: int):
    if row_idx < 0 or col_idx < 0:
        return None
    if row_idx >= df.shape[0] or col_idx >= df.shape[1]:
        return None
    return df.iat[row_idx, col_idx]


def extract_block(df: pd.DataFrame, block: BlockConfig) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    product_row = block.product_row - 1
    rate_row = block.rate_row - 1
    qty_row = block.qty_row - 1
    start_col = column_letter_to_index(block.start_column)
    stride = block.stride
    col_idx = start_col
    while col_idx < df.shape[1]:
        product_name = normalize_text(get_cell(df, product_row, col_idx))
        if not product_name:
            break
        record = {
            "product": product_name,
            "cost_lb": to_number(get_cell(df, rate_row, col_idx)),
            "qty": to_number(get_cell(df, qty_row, col_idx)),
            "column": col_idx + 1,  # store column number for diagnostics
        }
        records.append(record)
        col_idx += stride
    return records


def build_records(source_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for plant_cfg in PLANT_CONFIGS:
        workbook_path = source_dir / plant_cfg.filename
        if not workbook_path.exists():
            raise FileNotFoundError(f"Workbook not found: {workbook_path}")
        with pd.ExcelFile(workbook_path, engine="openpyxl") as xls:
            available_sheets = set(xls.sheet_names)
            for sheet_name, period in plant_cfg.sheet_periods.items():
                if sheet_name not in available_sheets:
                    print(
                        f"Warning: sheet '{sheet_name}' missing in {workbook_path.name}",
                        file=sys.stderr,
                    )
                    continue
                df = xls.parse(sheet_name=sheet_name, header=None, dtype=object)
                sheet_records = 0
                for block in plant_cfg.blocks:
                    block_records = extract_block(df, block)
                    sheet_records += len(block_records)
                    for record in block_records:
                        record.update(
                            {
                                "plant": plant_cfg.plant,
                                "period": period,
                                "sheet": sheet_name,
                                "source": str(workbook_path),
                            }
                        )
                        records.append(record)
                if sheet_records == 0:
                    print(
                        f"Warning: no records extracted for {plant_cfg.plant} sheet '{sheet_name}'",
                        file=sys.stderr,
                    )
    return records


def build_weighted_rates(df: pd.DataFrame) -> pd.DataFrame:
    weighted = (
        df.dropna(subset=["product"])
        .groupby(["plant", "product"], dropna=False, as_index=False)
        .agg(
            total_qty=("qty", lambda s: s.sum(min_count=1)),
            total_cost=("cost", lambda s: s.sum(min_count=1)),
        )
    )
    weighted["weighted_cost_lb"] = weighted["total_cost"] / weighted["total_qty"]
    # Avoid divide-by-zero noise
    weighted.loc[weighted["total_qty"] == 0, "weighted_cost_lb"] = pd.NA
    cols = ["plant", "product", "weighted_cost_lb", "total_qty", "total_cost"]
    return weighted[cols].sort_values(["plant", "product"]).reset_index(drop=True)


def write_outputs(detail_df: pd.DataFrame, weighted_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "plant_rates_2025.csv"
    parquet_path = out_dir / "plant_rates_2025.parquet"
    detail_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Wrote {len(detail_df)} rows to {csv_path}")
    try:
        detail_df.to_parquet(parquet_path, index=False)
        print(f"Wrote {len(detail_df)} rows to {parquet_path}")
    except Exception as exc:
        print(f"Warning: failed to write parquet output ({exc})", file=sys.stderr)

    weighted_csv = out_dir / "plant_rates_2025_weighted.csv"
    weighted_parquet = out_dir / "plant_rates_2025_weighted.parquet"
    weighted_df.to_csv(weighted_csv, index=False, encoding="utf-8")
    print(f"Wrote {len(weighted_df)} rows to {weighted_csv}")
    try:
        weighted_df.to_parquet(weighted_parquet, index=False)
        print(f"Wrote {len(weighted_df)} rows to {weighted_parquet}")
    except Exception as exc:
        print(f"Warning: failed to write weighted parquet output ({exc})", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tidy 2025 plant rate extracts from standardized workbooks."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing the 2025 rate workbooks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory that will receive the consolidated outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args.source_dir)
    if not records:
        print("No records extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    df = df[["plant", "period", "product", "cost_lb", "qty"]]
    df["cost"] = df["cost_lb"] * df["qty"]
    df = df.sort_values(["plant", "period", "product"]).reset_index(drop=True)

    weighted_df = build_weighted_rates(df)

    write_outputs(df, weighted_df, args.out_dir)

    counts = df.groupby("plant").size().to_dict()
    for plant, count in sorted(counts.items()):
        print(f"{plant}: {count} rows")


if __name__ == "__main__":
    main()
