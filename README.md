# Plant Rates Tidy Builder

Utilities and configuration for converting plant rate/production cross-tab workbooks into tidy datasets consumable by notebooks or downstream pipelines.

## Repository Layout

- `configs/plants.yml` — parsing contract for workbook layouts, plant detection, sheet-to-period mapping, normalization settings, and output schema.
- `configs/map_cost_element.csv` — optional lookup that can be referenced by the YAML to map normalized cost-element labels to canonical codes.
- `configs/product_map.csv` — lookup that normalizes workbook product headers to `ProductMesh` names and supplies the associated `MatGroup`.
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

### Quick start

1. **Activate the virtual environment** described above so `pandas`, `pyarrow`, `pyyaml`, and `openpyxl` are available.
2. **Set `configs/plants.yml:root_folder`** to the directory that contains the Excel workbooks (UNC share in production or a local folder for testing). Confirm the `periods`, `plants`, and `rulesets` entries cover the files you plan to ingest.
3. **Dry-run the parser** to validate sheet layouts without touching output files:
   ```powershell
   python scripts/build_tidy.py --dry-run --verbose
   ```
   This prints workbook/sheet counts and any validation errors while leaving `data/tidy` untouched.
4. **Execute a full build** once the dry-run succeeds:
   ```powershell
   python scripts/build_tidy.py --out-dir data/tidy
   ```
   Omit `--out-dir` to use the default folder, or point it to another location/UNC share when needed.
5. **Inspect the outputs** (`data/tidy/rates_tidy.*`, `data/tidy/costs_by_period.*`) and the console summary. The script exits with non-zero status if validation fails, so a clean run always ends with the "Workbooks processed" summary.

### Additional flags

- `--out-dir PATH` – override the default `data/tidy` output directory.
- `--dry-run` – parse/validate only; no files written.
- `--verbose` – echo per-sheet parsing summaries.

## Outputs

On success the script overwrites/creates the following datasets (CSV + Parquet):

- `data/tidy/rates_tidy.*` — columns `[plant, period, product, mat_group, element_code, rate, qty, cost, source_path]`.
- `data/tidy/costs_by_period.*` — grouped view of total `cost` per `period`.

Behavioral notes:

- Workbook product headers are normalized via `configs/product_map.csv`: `product` stores the mapped `ProductMesh`, and `mat_group` stores the accompanying `MatGroup`. When a product is missing from the lookup, the original label is retained and `mat_group` is null.
- Cost-element rows populate `rate` and carry the period/product-level production value in `qty`; `cost` is simply `rate * qty`.
- Production row values are loaded once per period/product (from the `TOTAL PRODUCTION` row) and copied onto every tidy record sharing that `(plant, period, product)` key, so there is no separate "production element".
- Duplicate `(plant, period, product, element_code)` rows (e.g., duplicate product columns in the workbook) are summed and their source paths concatenated (unique order preserved).

## Validation & Troubleshooting

- The script exits with non-zero status if:
  - Any `period` falls outside the whitelist mapping (exit code 3).
  - Duplicate `(plant, period, product, element_code)` keys remain after aggregation (exit code 2).
  - Unexpected errors occur during parsing.
- If production rows are missing, verify the `ruleset.layout.production.match` patterns against the actual sheet labels.
- When adding new cost-element names, update the normalization rules or mapping CSV so `element_code` stays populated.

Run with `--verbose` and inspect the generated data/logs to diagnose issues quickly.
