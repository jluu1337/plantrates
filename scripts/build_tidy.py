#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "plants.yml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "tidy"
def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_row_keys(row: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    normalized: Dict[str, Optional[str]] = {}
    for key, value in row.items():
        if key is None:
            continue
        cleaned_key = key.replace("\ufeff", "").strip()
        normalized_key = cleaned_key.lower()
        normalized[normalized_key] = value.strip() if isinstance(value, str) else value
    return normalized


def load_mapping_csv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized = _normalize_row_keys(row)
            source_name = normalized.get("source_name")
            element_code = normalized.get("element_code")
            if source_name:
                mapping[source_name] = element_code
    return mapping


def load_product_map_csv(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized = _normalize_row_keys(row)
            product = normalized.get("product")
            if not product:
                continue
            key = normalize_product_key(product)
            mapping[key] = {
                "product": normalized.get("productmesh") or product,
                "mat_group": normalized.get("matgroup"),
            }
    return mapping


def list_excels(root_folder: Path) -> List[Path]:
    if not root_folder.exists():
        return []
    files = [
        path
        for path in root_folder.rglob("*.xlsx")
        if path.is_file() and not path.name.startswith("~$")
    ]
    return sorted(files)


def detect_plant(workbook_path: Path, plants_cfg: Sequence[Dict]) -> Optional[Tuple[str, Dict]]:
    filename = workbook_path.name
    for plant_entry in plants_cfg:
        pattern = plant_entry.get("match")
        if not pattern:
            continue
        if re.search(pattern, filename, flags=re.IGNORECASE):
            return plant_entry["plant_code"], plant_entry
    return None


def normalize_text(value, trim: bool, collapse: bool) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if trim:
        text = text.strip()
    if collapse:
        text = re.sub(r"\s+", " ", text)
    return text


def normalize_product_key(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def column_letter_to_index(letter: str) -> int:
    letter = letter.upper()
    idx = 0
    for char in letter:
        if not char.isalpha():
            raise ValueError(f"Invalid column letter '{letter}'")
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def get_cell(df: pd.DataFrame, row_idx: int, col_idx: int):
    if row_idx < 0 or col_idx < 0:
        return None
    if row_idx >= df.shape[0] or col_idx >= df.shape[1]:
        return None
    return df.iat[row_idx, col_idx]


def contains_ilike(text: str, pattern: str) -> bool:
    if not text:
        return False
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def label_matches(label: str, match_cfg: Dict) -> bool:
    if not label or not match_cfg:
        return False
    match_type = match_cfg.get("type")
    values = match_cfg.get("values", [])
    if match_type == "contains_ilike":
        label_lower = label.lower()
        return any(val.lower() in label_lower for val in values)
    if match_type == "equals_ilike":
        label_lower = label.lower()
        return any(label_lower == val.lower() for val in values)
    if match_type == "regex":
        return any(re.search(val, label, re.IGNORECASE) for val in values)
    return False


def apply_cost_rules(label: str, rules: Sequence[Dict]) -> str:
    for rule in rules or []:
        rule_type = rule.get("type")
        pattern = rule.get("pattern")
        replacement = rule.get("to")
        if not pattern or replacement is None:
            continue
        if rule_type == "contains" and re.search(pattern, label, re.IGNORECASE):
            return replacement
        if rule_type == "equals" and re.fullmatch(pattern, label, re.IGNORECASE):
            return replacement
        if rule_type == "startswith" and re.match(pattern, label, re.IGNORECASE):
            return replacement
    return label


def parse_numeric(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    text = text.replace(" ", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_products(
    df: pd.DataFrame,
    product_row_idx: int,
    start_col_idx: int,
    stride: int,
    norm_cfg: Dict,
) -> List[Tuple[int, str]]:
    products: List[Tuple[int, str]] = []
    col_idx = start_col_idx
    trim = norm_cfg.get("trim_whitespace", False)
    collapse = norm_cfg.get("collapse_internal_spaces", False)
    while col_idx < df.shape[1]:
        raw_value = get_cell(df, product_row_idx, col_idx)
        label = normalize_text(raw_value, trim, collapse)
        if not label:
            break
        products.append((col_idx, label))
        col_idx += stride
    return products


def find_production_row(
    df: pd.DataFrame,
    start_idx: int,
    label_col_idx: int,
    norm_cfg: Dict,
    match_cfg: Dict,
) -> Optional[int]:
    trim = norm_cfg.get("trim_whitespace", False)
    collapse = norm_cfg.get("collapse_internal_spaces", False)
    total_rows = df.shape[0]
    for row_idx in range(max(start_idx, 0), total_rows):
        raw_label = get_cell(df, row_idx, label_col_idx)
        label = normalize_text(raw_label, trim, collapse)
        if not label:
            continue
        if label_matches(label, match_cfg):
            return row_idx
    return None


def parse_band(
    df: pd.DataFrame,
    band_cfg: Dict,
    layout_cfg: Dict,
    production_cfg: Dict,
    cost_rules: Sequence[Dict],
    element_mapping: Optional[Dict[str, str]],
    product_map: Dict[str, Dict[str, Optional[str]]],
    plant_code: str,
    period_date: pd.Timestamp,
    source_path: str,
    norm_cfg: Dict,
) -> List[Dict]:
    trim = norm_cfg.get("trim_whitespace", False)
    collapse = norm_cfg.get("collapse_internal_spaces", False)
    product_row = band_cfg.get("product_row", layout_cfg.get("product_row"))
    start_column_letter = band_cfg.get(
        "product_start_column", layout_cfg.get("product_start_column", "A")
    )
    stride = band_cfg.get("product_stride", layout_cfg.get("product_stride", 1))
    if product_row is None or start_column_letter is None:
        return []
    product_row_idx = int(product_row) - 1
    start_col_idx = column_letter_to_index(start_column_letter)
    raw_products = extract_products(df, product_row_idx, start_col_idx, stride, norm_cfg)
    if not raw_products:
        return []

    products: List[Tuple[int, str, Optional[str]]] = []
    for col_idx, label in raw_products:
        # Use product name directly from the plant file (no product map normalization)
        products.append((col_idx, label, None))

    base_cost_cfg = layout_cfg.get("cost_elements", {})
    band_cost_cfg = band_cfg.get("cost_elements", {})
    label_column_letter = band_cost_cfg.get(
        "label_column", base_cost_cfg.get("label_column", "A")
    )
    label_col_idx = column_letter_to_index(label_column_letter)
    start_row = band_cost_cfg.get("start_row", base_cost_cfg.get("start_row"))
    if start_row is None:
        return []

    stop_text = band_cost_cfg.get(
        "stop_before_text_ilike", base_cost_cfg.get("stop_before_text_ilike")
    )
    exclude_rows = band_cost_cfg.get(
        "exclude_rows_ilike", base_cost_cfg.get("exclude_rows_ilike", [])
    )

    records: List[Dict] = []
    product_qty: Dict[str, Optional[float]] = {}
    row_idx = int(start_row) - 1
    total_rows = df.shape[0]
    while row_idx < total_rows:
        raw_label = get_cell(df, row_idx, label_col_idx)
        label = normalize_text(raw_label, trim, collapse)
        if not label:
            row_idx += 1
            continue
        if stop_text and contains_ilike(label, stop_text):
            break
        if any(contains_ilike(label, patt) for patt in exclude_rows):
            row_idx += 1
            continue
        if production_cfg and label_matches(label, production_cfg.get("match", {})):
            for col_idx, product_name, _ in products:
                value = parse_numeric(get_cell(df, row_idx, col_idx))
                if value is None:
                    continue
                product_qty[product_name] = value
            row_idx += 1
            continue
        normalized_label = apply_cost_rules(label, cost_rules)
        element_code = (
            element_mapping.get(normalized_label)
            if element_mapping is not None
            else normalized_label
        )
        if element_code is None:
            element_code = normalized_label
        for col_idx, product_name, mat_group in products:
            value = parse_numeric(get_cell(df, row_idx, col_idx))
            if value is None:
                continue
            records.append(
                {
                    "plant": plant_code,
                    "period": period_date,
                    "product": product_name,
                    "mat_group": mat_group,
                        "element_code": element_code,
                        "rate": value,
                        "qty": None,
                        "source_path": source_path,
                    }
                )
        row_idx += 1

    if production_cfg and not product_qty:
        match_cfg = production_cfg.get("match", {})
        prod_row_idx = find_production_row(
            df=df,
            start_idx=row_idx,
            label_col_idx=label_col_idx,
            norm_cfg=norm_cfg,
            match_cfg=match_cfg,
        )
        if prod_row_idx is not None:
            for col_idx, product_name, _ in products:
                value = parse_numeric(get_cell(df, prod_row_idx, col_idx))
                if value is None:
                    continue
                product_qty[product_name] = value

    if product_qty:
        for record in records:
            record["qty"] = product_qty.get(record["product"])

    return records


def parse_sheet(
    df: pd.DataFrame,
    ruleset: Dict,
    element_mapping: Optional[Dict[str, str]],
    product_map: Dict[str, Dict[str, Optional[str]]],
    plant_code: str,
    period_date: pd.Timestamp,
    source_path: str,
    norm_cfg: Dict,
) -> List[Dict]:
    layout = ruleset.get("layout", {})
    production_cfg = layout.get("production", {})
    cost_rules = ruleset.get("cost_element_normalization", [])
    records: List[Dict] = []

    product_bands = layout.get("product_bands")
    bands = product_bands or [layout]
    for band in bands:
        band_records = parse_band(
            df=df,
            band_cfg=band,
            layout_cfg=layout,
            production_cfg=production_cfg,
            cost_rules=cost_rules,
            element_mapping=element_mapping,
            product_map=product_map,
            plant_code=plant_code,
            period_date=period_date,
            source_path=source_path,
            norm_cfg=norm_cfg,
        )
        records.extend(band_records)
    return records


def write_outputs(tidy_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates_parquet = out_dir / "rates_tidy.parquet"
    rates_csv = out_dir / "rates_tidy.csv"
    cost_parquet = out_dir / "costs_by_period.parquet"
    cost_csv = out_dir / "costs_by_period.csv"

    tidy_df.to_parquet(rates_parquet, index=False)
    tidy_df.to_csv(rates_csv, index=False)

    cost_view = (
        tidy_df.groupby("period", dropna=False)
        .agg(cost=("cost", lambda s: s.sum(min_count=1)))
        .reset_index()
    )
    cost_view.to_parquet(cost_parquet, index=False)
    cost_view.to_csv(cost_csv, index=False)


def validate_outputs(tidy_df: pd.DataFrame, allowed_periods: Iterable[pd.Timestamp]) -> None:
    allowed_set = set(allowed_periods)
    invalid_periods = set()
    for period in pd.unique(tidy_df["period"]):
        if pd.isna(period):
            invalid_periods.add(period)
        elif period not in allowed_set:
            invalid_periods.add(period)
    if invalid_periods:
        print("Invalid period values detected:", invalid_periods, file=sys.stderr)
        sys.exit(3)

    key_cols = ["plant", "period", "product", "element_code"]
    if not tidy_df.empty:
        duplicates = tidy_df[tidy_df.duplicated(subset=key_cols, keep=False)]
        if not duplicates.empty:
            print("Duplicate rate keys detected:", file=sys.stderr)
            print(duplicates[key_cols], file=sys.stderr)
            sys.exit(2)


def period_str_to_timestamp(period_str: str) -> pd.Timestamp:
    try:
        dt = datetime.strptime(period_str, "%Y/%m")
    except ValueError as exc:
        raise ValueError(f"Invalid period format '{period_str}'") from exc
    return pd.Timestamp(dt.year, dt.month, 1)


def build_summary(df: pd.DataFrame, label: str) -> List[str]:
    if df.empty:
        return [f"{label}: 0 rows"]
    counts = df.groupby("plant").size()
    summary = [f"{label}: {len(df)} rows"]
    for plant, count in counts.items():
        summary.append(f"  - {plant}: {count}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build tidy plant rate datasets.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for tidy outputs (default: data/tidy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and print summary without writing outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    mapping_cfg = config.get("mapping") or {}
    element_mapping: Optional[Dict[str, str]] = None
    cost_map_path = mapping_cfg.get("cost_element_csv")
    if cost_map_path:
        resolved = (PROJECT_ROOT / cost_map_path).resolve()
        if resolved.exists():
            element_mapping = load_mapping_csv(resolved)
    product_map: Dict[str, Dict[str, Optional[str]]] = {}
    product_map_path = mapping_cfg.get("product_map_csv")
    if product_map_path:
        resolved = (PROJECT_ROOT / product_map_path).resolve()
        if resolved.exists():
            product_map = load_product_map_csv(resolved)

    root_folder = Path(config["root_folder"])
    excel_files = list_excels(root_folder)
    if not excel_files:
        print(f"No Excel files found under {root_folder}.")

    periods_cfg = config.get("periods", {})
    include_sheets = periods_cfg.get("include_sheets", [])
    period_mapping = periods_cfg.get("mapping", {})
    allowed_periods = {period_str_to_timestamp(v) for v in period_mapping.values()}
    norm_cfg = config.get("normalize", {})
    plants_cfg = config.get("plants", [])
    rulesets = config.get("rulesets", {})

    records: List[Dict] = []
    files_processed = 0

    for workbook_path in excel_files:
        plant_match = detect_plant(workbook_path, plants_cfg)
        if not plant_match:
            continue
        plant_code, plant_entry = plant_match
        ruleset_name = plant_entry.get("ruleset")
        ruleset = rulesets.get(ruleset_name)
        if not ruleset:
            print(f"Ruleset '{ruleset_name}' not found for {workbook_path.name}", file=sys.stderr)
            continue
        processed_sheet = False
        try:
            with pd.ExcelFile(workbook_path, engine="openpyxl") as xls:
                available_sheets = set(xls.sheet_names)
                for sheet_name in include_sheets:
                    if sheet_name not in available_sheets:
                        continue
                    period_str = period_mapping.get(sheet_name)
                    if not period_str:
                        print(f"No period mapping for sheet '{sheet_name}'.", file=sys.stderr)
                        sys.exit(3)
                    period_date = period_str_to_timestamp(period_str)
                    df = xls.parse(sheet_name=sheet_name, header=None, dtype=object)
                    sheet_records = parse_sheet(
                        df=df,
                        ruleset=ruleset,
                        element_mapping=element_mapping,
                        product_map=product_map,
                        plant_code=plant_code,
                        period_date=period_date,
                        source_path=str(workbook_path),
                        norm_cfg=norm_cfg,
                    )
                    qty_multipliers = plant_entry.get("qty_multipliers") or {}
                    if qty_multipliers:
                        for record in sheet_records:
                            element_code = record.get("element_code")
                            if element_code is None:
                                continue
                            factor = qty_multipliers.get(element_code)
                            if factor is None:
                                continue
                            qty_value = record.get("qty")
                            if qty_value is None:
                                continue
                            record["qty"] = qty_value * factor
                    if sheet_records:
                        processed_sheet = True
                        records.extend(sheet_records)
                        if args.verbose:
                            print(
                                f"{workbook_path.name} [{sheet_name}] -> "
                                f"{sum(1 for r in sheet_records if r.get('rate') is not None)} rates, "
                                f"{sum(1 for r in sheet_records if r.get('qty') is not None)} production rows."
                            )
        except Exception as exc:
            print(f"Failed to process {workbook_path}: {exc}", file=sys.stderr)
            continue
        if processed_sheet:
            files_processed += 1

    tidy_df = pd.DataFrame(records)
    if tidy_df.empty:
        tidy_df = pd.DataFrame(
            columns=[
                "plant",
                "period",
                "period_text",
                "product",
                "mat_group",
                "element_code",
                "rate",
                "qty",
                "cost",
                "source_path",
            ]
        )
    else:
        tidy_df = (
            tidy_df.groupby(["plant", "period", "product", "element_code"], dropna=False)
            .agg(
                rate=("rate", lambda s: s.sum(min_count=1)),
                qty=("qty", lambda s: s.sum(min_count=1)),
                mat_group=("mat_group", "first"),
                source_path=("source_path", lambda vals: "|".join(dict.fromkeys(v for v in vals if v))),
            )
            .reset_index()
        )
        tidy_df["rate"] = tidy_df["rate"].round(4)
        tidy_df["cost"] = tidy_df["rate"] * (tidy_df["qty"] * 1000.0)
        tidy_df["period_text"] = tidy_df["period"].dt.strftime("%Y/%m")

    desired_columns = config.get("output", {}).get(
        "columns",
        [
            "plant",
            "period",
            "period_text",
            "product",
            "mat_group",
            "element_code",
            "rate",
            "qty",
            "cost",
            "source_path",
        ],
    )
    for col in desired_columns:
        if col not in tidy_df.columns:
            tidy_df[col] = None
    tidy_df = tidy_df[desired_columns]

    validate_outputs(tidy_df, allowed_periods)

    out_dir = Path(args.out_dir).resolve()
    if not args.dry_run:
        write_outputs(tidy_df, out_dir)

    print(f"Workbooks processed: {files_processed}")
    for line in build_summary(tidy_df, "Tidy rows"):
        print(line)


if __name__ == "__main__":
    main()
