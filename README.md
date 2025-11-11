# Plant Rates Tidy Builder

Utilities and configuration for converting plant rate/production cross-tab workbooks into tidy datasets consumable by notebooks or downstream pipelines.

## Repository Layout

- `configs/plants.yml` — parsing contract for workbook layouts, plant detection, sheet-to-period mapping, normalization settings, and output schema.
- `configs/map_cost_element.csv` — optional lookup that can be referenced by the YAML to map normalized cost-element labels to canonical codes.
- `scripts/build_tidy.py` — main entry point that reads the config, scans for Excel workbooks, parses sheets, applies normalization/mapping, validates uniqueness, and emits tidy parquet/csv outputs.
- `data/tidy/` — target folder for generated `rates_tidy` and `production_tidy` files.
- `data/logs/` — (optional) diagnostic logs; currently unused since cost-element mapping is embedded in `plants.yml`.
- `src/Components/` — local copy of the authoritative Excel workbooks (original source lives on the UNC path documented in the YAML).

## Environment Setup

1. Create a virtual environment (PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install pandas pyarrow pyyaml openpyxl
   ```
2. Keep the venv activated while running scripts so the required packages stay available.

## Configuration (`configs/plants.yml`)

Key sections you may need to edit:

- `root_folder` — where the Excel workbooks live. Point to the UNC share for production or a local directory for testing.
- `normalize_plant_from_filename` / `plants` — regex patterns that map workbook filenames to plant codes and rulesets.
- `periods` — whitelist of sheet names plus the mapping to `YYYY/MM` strings (converted to first-of-month dates).
- `rulesets` — layout instructions per plant style (EUF/MCI, NI, TMB). Defines product bands, cost-element row boundaries, production row matching, and cost-element normalization patterns.
- `output` — enforced column order and default metadata (currency and rate unit).

Adjust these entries when workbook formats change or new plants are introduced.

## Running the Builder

Full run (writes outputs):

```powershell
python scripts/build_tidy.py
```

Options:

- `--out-dir PATH` — override the default `data/tidy` output directory.
- `--dry-run` — parse/validate only; no files written.
- `--verbose` — echo per-sheet parsing summaries.

## Outputs

On success the script overwrites/creates a single tidy dataset (CSV + Parquet):

- `data/tidy/rates_tidy.*` — columns `[plant, period, product, element_code, rate, qty, currency, rate_uom, source_path]`.

Behavioral notes:

- Cost-element rows populate `rate`, leave `qty` null, and use the canonical `element_code`.
- Production rows populate `qty`, leave `rate` null, and use a sentinel element code (`__production_qty__`) so downstream logic can distinguish them.
- Duplicate `(plant, period, product, element_code)` rows (e.g., duplicate product columns in the workbook) are summed and their source paths concatenated (unique order preserved).

## Validation & Troubleshooting

- The script exits with non-zero status if:
  - Any `period` falls outside the whitelist mapping (exit code 3).
  - Duplicate `(plant, period, product, element_code)` keys remain after aggregation (exit code 2).
  - Unexpected errors occur during parsing.
- If production rows are missing, verify the `ruleset.layout.production.match` patterns against the actual sheet labels.
- When adding new cost-element names, update the normalization rules or mapping CSV so `element_code` stays populated.

Run with `--verbose` and inspect the generated data/logs to diagnose issues quickly.
