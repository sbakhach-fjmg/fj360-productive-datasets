# salesforce_update

Builds the **Sales & Opportunity datasets for Tableau** from raw Salesforce and Advent (DMS) extracts. It runs as a plain Python script, launched by an Alteryx **Run Command** tool, and does all of its set logic in DuckDB.

| Script | Purpose |
|---|---|
| `create_sf_datasets.py` | Main pipeline: 7 parquet inputs → 3 parquet outputs, plus run logs and a QA log |
| `diagnose_unmatched_credit.py` | Diagnostic: lists sales credit that matched an opportunity but lost its sales rep |

> This repo holds code only. Inputs, outputs and diagnostic CSVs contain real employee and deal data, and `.gitignore` excludes them.

---

## Inputs and outputs

All paths and file names are set in the `CONFIG` dict at the top of `create_sf_datasets.py`.

**Inputs** (`CONFIG["input_dir"]`)

| File | Contents |
|---|---|
| `raw_opportunity.parquet` | Salesforce opportunities: lead and vehicle dims, stage, deal number, salesperson, BDR |
| `raw_sales_credit.parquet` | Advent sales credit: one row per credited rep (0.5/0.5 splits), front/back gross already split |
| `raw_users.parquet` | Salesforce users: store, department, profile, Advent id |
| `raw_events.parquet` | Appointments: created, scheduled, shows, BDC scheduling |
| `raw_tasks.parquet` | Calls, emails and texts (`activity_type`) |
| `raw_leads.parquet` | E-commerce leads |
| `raw_bdc_credit.parquet` | BDC sales credit per opportunity |

Column names are matched without regard to case or separators (`rs date` = `RS_Date` = `rsdate`). A missing column stops the run with an error at load time.

**Outputs** (`CONFIG["output_dir"]`)

| File | Grain |
|---|---|
| `output_1_salesperson.parquet` | dealercode × date × salesperson × vehicle × lead dims |
| `output_2_store.parquet` | Output 1 regrouped without the user columns, so the totals tie by construction |
| `output_3_bdc.parquet` | BDC population (`profile_name` contains `BDC Sales`) |

Measures: `opportunities_created`, `total_sales`, `front_gross`, `back_gross`, `appointments_created`, `appointments_scheduled`, `shows`, `sold_to_show`, `calls`, `texts`, `emails`, `leads_created`.

Each output is written to a `.tmp` file and then moved into place with `os.replace()`, which is atomic. A crashed run therefore never leaves a truncated file for a downstream Input Data tool to read.

**Logs** (`CONFIG["log_dir"]`)

- `dataset_<timestamp>.log`: full QA block, one file per run
- `last_run.log`: the same content at a fixed path, overwritten each run
- `qa_log.csv`: structured QA records (`seq, section, severity, label, value`) for Run Command's *Read Results from File*. Filter it on `WARNING`/`FAIL`.

---

## Design notes

**Facts are stacked, not joined.** Opportunities and sales credit share no date, user key or dimension source. Each source emits rows on its own date, with its own measure populated and the others set to zero, and one `GROUP BY` over all dimensions collapses them. A single `date` column therefore means the right thing for every measure, and Tableau date filters work correctly.

**Two store columns.** One rooftop is two Salesforce lots (LVMBN / L2MBN) that Advent treats as one store.
- `dealercode`: the Salesforce lot. All reporting uses it.
- `advent_dealer_id`: the Advent store. Only the sale → opportunity join uses it.

A sale matches on `advent_dealer_id` + deal number, then reports on the lot it inherits from the matched opportunity.

**Ranked sale → opportunity match.** `CONFIG["sale_match_rank"]` decides which opportunity stages can lend lead and vehicle dims to a sale. Its order sets the tie-break priority (Finalized non-Advent → Finalized Advent → Registered Sale). A sale with no eligible opportunity falls into the `Advent` lead-type bucket.

**Rep resolution.** The Advent `sp_id` maps to a Salesforce user through `(advent store, advent_id)`. Credit that doesn't resolve stays under the synthetic `unassigned` user (`-1`), so it still counts.

Further detail is in the module docstring and in the comments throughout `CONFIG`.

---

## Setup

Requires Python ≥ 3.12. Dependencies: `duckdb`, `pandas`, `numpy`, `pyarrow`.

With [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Or with pip:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "duckdb>=1.0,<2" "pandas>=2.0,<3" numpy pyarrow
```

## Configuration

Edit the `CONFIG` dict at the top of `create_sf_datasets.py`. That is the only block you should need to change. The keys you're most likely to touch:

| Key | Meaning |
|---|---|
| `input_dir`, `output_dir`, `log_dir` | Paths. The defaults are specific to the production Alteryx server and its mapped network drive. |
| `timezone` | Salesforce UTC timestamps are converted to Pacific calendar days (DST-aware) |
| `excluded_stages` | Opportunity stages dropped entirely (default `Closed Duplicate`) |
| `sale_match_rank` | Stages eligible to match a sale, in order of preference |
| `advent_only_stores` | Advent stores known to have no Salesforce opportunities (divested, or never on Salesforce) |
| `bdc_profile_match` | Profile substring that defines the Output 3 population |
| `activity_map` | Maps the leading word of `activity_type` to a measure (`call` → `calls`, etc.) |
| `label_overrides` | Forces a spelling to win during label canonicalization |
| `date_as_datetime` | `True` makes `date` a real datetime; `False` makes it an ISO string |
| `verbose_qa`, `strict_checks` | Echo QA to stdout; exit non-zero if reconciliation fails |

## Running

### From Alteryx (production)

Point a **Run Command** tool at the venv's interpreter, with the script as its argument:

```
Command:  <repo>\.venv\Scripts\python.exe
Args:     create_sf_datasets.py
```

- Use the venv's `python.exe`, **not** `py.exe`. The launcher would pick the system interpreter, which doesn't have duckdb.
- **The exit code is the contract:** `0` = all three outputs published, `1` = not published. Run Command throws away stdout, so the log files are the only record of a run.
- No `.bat` wrapper is needed. If you do use one, it must end with `exit /b %ERRORLEVEL%`. Otherwise it returns the exit code of its last command and can report success after a failed run.
- Afterwards, Input Data tools read the three parquet outputs back into the workflow.

### From a terminal

```powershell
.venv\Scripts\python.exe create_sf_datasets.py
```

## Diagnosing unmatched sales credit

`diagnose_unmatched_credit.py` reports on one specific gap that can be fixed. These are deals that matched a Finalized opportunity, but whose Advent `sp_id` didn't resolve to a Salesforce user. Their credit sits on `unassigned`, so a named rep is losing those numbers.

```powershell
.venv\Scripts\python.exe diagnose_unmatched_credit.py 2026-07-01 2026-07-31            # writes to current dir
.venv\Scripts\python.exe diagnose_unmatched_credit.py 2026-07-01 2026-07-31 C:\temp    # writes to C:\temp
.venv\Scripts\python.exe diagnose_unmatched_credit.py                                   # no date bounds
```

Both dates are inclusive. The script writes `unmatched_rep_<from>_<to>.csv` and prints the four-way match split plus a triage summary. Read each row like this:

| Pattern | Fix |
|---|---|
| Names agree, `sf_assigned_advent_id` blank | The rep has no Advent id. Populate it on the user record. |
| Names agree, Advent ids differ | The user record has the wrong Advent id |
| Names agree, ids agree | Store mismatch: the id is registered at a different store |
| Names differ | Different people. This isn't a data bug. |
| `sf_salesperson_id = -1` | The opportunity has no salesperson either |

The diagnostic imports `create_sf_datasets.py` from **its own folder** and reuses the real `load_inputs()` and `build()`. Its results therefore always agree with the published outputs. Importing the main script is safe because its entry point is guarded, so nothing gets written or published. **Keep the two files in sync.** If they're out of date with each other, the script stops with a `VERSION SKEW` message.
