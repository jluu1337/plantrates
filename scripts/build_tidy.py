#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "plants.yml"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "tidy"


def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_mapping_csv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source_name = row.get("source_name")
            element_code = row.get("element_code")
            if source_name:
                mapping[source_name] = element_code
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
        return any(re.search(val, label, re.IGNORECASE) for val in values)
    if match_type == "equals_ilike":
        return any(re.fullmatch(val, label, re.IGNORECASE) for val in values)
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


def parse_band(
    df: pd.DataFrame,
    band_cfg: Dict,
    layout_cfg: Dict,
    production_cfg: Dict,
    cost_rules: Sequence[Dict],
    mapping: Dict[str, str],
    defaults: Dict[str, str],
    plant_code: str,
    period_date: pd.Timestamp,
    source_path: str,
    norm_cfg: Dict,
    unmapped_counter: Counter,
) -> Tuple[List[Dict], List[Dict]]:
    trim = norm_cfg.get("trim_whitespace", False)
    collapse = norm_cfg.get("collapse_internal_spaces", False)
    product_row = band_cfg.get("product_row", layout_cfg.get("product_row"))
    start_column_letter = band_cfg.get(
        "product_start_column", layout_cfg.get("product_start_column", "A")
    )
    stride = band_cfg.get("product_stride", layout_cfg.get("product_stride", 1))
    if product_row is None or start_column_letter is None:
        return [], []
    product_row_idx = int(product_row) - 1
    start_col_idx = column_letter_to_index(start_column_letter)
    products = extract_products(df, product_row_idx, start_col_idx, stride, norm_cfg)
    if not products:
        return [], []

    base_cost_cfg = layout_cfg.get("cost_elements", {})
    band_cost_cfg = band_cfg.get("cost_elements", {})
    label_column_letter = band_cost_cfg.get(
        "label_column", base_cost_cfg.get("label_column", "A")
    )
    label_col_idx = column_letter_to_index(label_column_letter)
    start_row = band_cost_cfg.get("start_row", base_cost_cfg.get("start_row"))
    if start_row is None:
        return [], []

    stop_text = band_cost_cfg.get(
        "stop_before_text_ilike", base_cost_cfg.get("stop_before_text_ilike")
    )
    exclude_rows = band_cost_cfg.get(
        "exclude_rows_ilike", base_cost_cfg.get("exclude_rows_ilike", [])
    )

    rates_rows: List[Dict] = []
    qty_rows: List[Dict] = []
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
            for col_idx, product_name in products:
                value = parse_numeric(get_cell(df, row_idx, col_idx))
                if value is None:
                    continue
                qty_rows.append(
                    {
                        "plant": plant_code,
                        "period": period_date,
                        "product": product_name,
                        "qty": value,
                        "source_path": source_path,
                    }
                )
            row_idx += 1
            continue
        normalized_label = apply_cost_rules(label, cost_rules)
        element_code = mapping.get(normalized_label)
        if normalized_label and element_code is None:
            unmapped_counter[normalized_label] += 1
        for col_idx, product_name in products:
            value = parse_numeric(get_cell(df, row_idx, col_idx))
            if value is None:
                continue
            rates_rows.append(
                {
                    "plant": plant_code,
                    "period": period_date,
                    "product": product_name,
                    "element_code": element_code,
                    "rate": value,
                    "currency": defaults.get("currency"),
                    "rate_uom": defaults.get("rate_uom"),
                    "source_path": source_path,
                }
            )
        row_idx += 1

    return rates_rows, qty_rows


def parse_sheet(
    df: pd.DataFrame,
    ruleset: Dict,
    defaults: Dict[str, str],
    mapping: Dict[str, str],
    plant_code: str,
    period_date: pd.Timestamp,
    source_path: str,
    norm_cfg: Dict,
    unmapped_counter: Counter,
) -> Tuple[List[Dict], List[Dict]]:
    layout = ruleset.get("layout", {})
    production_cfg = layout.get("production", {})
    cost_rules = ruleset.get("cost_element_normalization", [])
    rates_rows: List[Dict] = []
    qty_rows: List[Dict] = []

    product_bands = layout.get("product_bands")
    bands = product_bands or [layout]
    for band in bands:
        band_rates, band_qty = parse_band(
            df=df,
            band_cfg=band,
            layout_cfg=layout,
            production_cfg=production_cfg,
            cost_rules=cost_rules,
            mapping=mapping,
            defaults=defaults,
            plant_code=plant_code,
            period_date=period_date,
            source_path=source_path,
            norm_cfg=norm_cfg,
            unmapped_counter=unmapped_counter,
        )
        rates_rows.extend(band_rates)
        qty_rows.extend(band_qty)
    return rates_rows, qty_rows


def write_outputs(rates_df: pd.DataFrame, qty_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates_parquet = out_dir / "rates_tidy.parquet"
    rates_csv = out_dir / "rates_tidy.csv"
    prod_parquet = out_dir / "production_tidy.parquet"
    prod_csv = out_dir / "production_tidy.csv"

    rates_df.to_parquet(rates_parquet, index=False)
    rates_df.to_csv(rates_csv, index=False)
    qty_df.to_parquet(prod_parquet, index=False)
    qty_df.to_csv(prod_csv, index=False)


def write_unmapped_log(counter: Counter, dry_run: bool) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "unmapped_cost_elements.csv"
    rows = sorted(counter.items(), key=lambda kv: kv[0])
    df = pd.DataFrame(rows, columns=["cost_name_norm", "count"])
    if not dry_run:
        df.to_csv(log_path, index=False)
    return log_path


def validate_outputs(rates_df: pd.DataFrame, allowed_periods: Iterable[pd.Timestamp]) -> None:
    allowed_set = set(allowed_periods)
    invalid_periods = set()
    for period in pd.unique(rates_df["period"]):
        if pd.isna(period):
            invalid_periods.add(period)
        elif period not in allowed_set:
            invalid_periods.add(period)
    if invalid_periods:
        print("Invalid period values detected:", invalid_periods, file=sys.stderr)
        sys.exit(3)

    key_cols = ["plant", "period", "product", "element_code"]
    if not rates_df.empty:
        duplicates = rates_df[rates_df.duplicated(subset=key_cols, keep=False)]
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


def build_summary(df: pd.DataFrame, value_col: str, label: str) -> List[str]:
    if df.empty:
        return [f"{label}: 0 rows"]
    counts = df.groupby("plant")[value_col].count()
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
    mapping_rel_path = config.get("mapping", {}).get("cost_element_csv")
    if not mapping_rel_path:
        print("Missing mapping cost element CSV path in config.", file=sys.stderr)
        sys.exit(1)
    mapping_path = (PROJECT_ROOT / mapping_rel_path).resolve()
    mapping = load_mapping_csv(mapping_path)

    root_folder = Path(config["root_folder"])
    excel_files = list_excels(root_folder)
    if not excel_files:
        print(f"No Excel files found under {root_folder}.")

    periods_cfg = config.get("periods", {})
    include_sheets = periods_cfg.get("include_sheets", [])
    period_mapping = periods_cfg.get("mapping", {})
    allowed_periods = {period_str_to_timestamp(v) for v in period_mapping.values()}
    norm_cfg = config.get("normalize", {})
    defaults = config.get("output", {}).get("defaults", {})
    plants_cfg = config.get("plants", [])
    rulesets = config.get("rulesets", {})

    rates_records: List[Dict] = []
    qty_records: List[Dict] = []
    unmapped_counter: Counter = Counter()
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
                    sheet_rates, sheet_qty = parse_sheet(
                        df=df,
                        ruleset=ruleset,
                        defaults=defaults,
                        mapping=mapping,
                        plant_code=plant_code,
                        period_date=period_date,
                        source_path=str(workbook_path),
                        norm_cfg=norm_cfg,
                        unmapped_counter=unmapped_counter,
                    )
                    if sheet_rates or sheet_qty:
                        processed_sheet = True
                        rates_records.extend(sheet_rates)
                        qty_records.extend(sheet_qty)
                        if args.verbose:
                            print(
                                f"{workbook_path.name} [{sheet_name}] -> "
                                f"{len(sheet_rates)} rates, {len(sheet_qty)} production rows."
                            )
        except Exception as exc:
            print(f"Failed to process {workbook_path}: {exc}", file=sys.stderr)
            continue
        if processed_sheet:
            files_processed += 1

    rates_df = pd.DataFrame(rates_records)
    qty_df = pd.DataFrame(qty_records)
    if rates_df.empty:
        rates_df = pd.DataFrame(
            columns=[
                "plant",
                "period",
                "product",
                "element_code",
                "rate",
                "currency",
                "rate_uom",
                "source_path",
            ]
        )
    if qty_df.empty:
        qty_df = pd.DataFrame(columns=["plant", "period", "product", "qty", "source_path"])

    validate_outputs(rates_df, allowed_periods)

    out_dir = Path(args.out_dir).resolve()
    if not args.dry_run:
        write_outputs(rates_df, qty_df, out_dir)
        write_unmapped_log(unmapped_counter, dry_run=False)
    else:
        write_unmapped_log(unmapped_counter, dry_run=True)

    print(f"Workbooks processed: {files_processed}")
    for line in build_summary(rates_df, "rate", "Rates tidy"):
        print(line)
    for line in build_summary(qty_df, "qty", "Production tidy"):
        print(line)
    print(f"Unmapped cost elements: {sum(unmapped_counter.values())} (log: {LOG_DIR / 'unmapped_cost_elements.csv'})")


if __name__ == "__main__":
    main()
