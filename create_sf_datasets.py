"""
Sales & Opportunity dataset for Tableau  --  Alteryx Python tool

Builds two outputs from six parquet inputs:

    Output 1 : salesperson grain  (dealercode, date, user, vehicle, lead dims)
    Output 2 : store grain        (same dims minus the user attributes)

Design note -- why the facts are STACKED and not joined
-------------------------------------------------------
The two measures share no date, no user key, and no dimension source:

                        opportunities_created        total_sales
    grain               one opportunity              one sales-credit row (0.5/0.5 splits)
    date                opportunity_created_date     "rs date"
    user key            salesforce salesperson_id    advent sp_id -> advent_id -> user_id
    lead/vehicle dims   native to the record         borrowed from opportunity, else 'Advent'

One wide joined row would force the single `date` column to mean two different things, so any
date filter in Tableau would silently break one of the measures. Instead each source emits rows
on its own date with one measure populated and the other zero; a single GROUP BY over all
dimensions collapses them. Output 2 is Output 1 re-grouped without the user columns, which makes
"the measures tie" true by construction rather than by coincidence.

Design note -- two store columns, joined on one and reported on the other
------------------------------------------------------------------------
One rooftop is two lots in Salesforce (LVMBN / L2MBN) that Advent does not distinguish, so
sales credit arrives on a single store code covering both. The opportunity and task files
therefore carry both:

    dealercode        the SALESFORCE LOT.   Every dimension in both outputs reports on this.
    advent_dealer_id  the ADVENT STORE.     Coarser. Only the sale -> opportunity join uses it.

So a sale MATCHES on advent_dealer_id and is REPORTED on the lot it inherits from the matched
opportunity -- which is what keeps the two lots separate downstream. The columns are identical
for every store that is not split, making this the identity mapping almost everywhere.

Everything below the CONFIG block is set logic in DuckDB. pandas is used only at the edges:
reading parquet, the one timezone conversion, and writing the outputs.

How this runs
-------------
As a PLAIN SCRIPT, invoked by an Alteryx **Run Command** tool pointed at a venv's
`Scripts\\python.exe` -- not by the Alteryx Python tool, and NOT via `py.exe` (a venv has no
py.exe, and the global launcher would use the system interpreter and lose duckdb). It writes
three parquet files, which Input Data tools then read back into the workflow.

That means **stdout goes nowhere** -- Run Command discards it. Everything diagnostic is written
to CONFIG["log_dir"] instead: a per-run log, a fixed-path `last_run.log`, and `qa_log.csv`
structured for Run Command's "Read Results from File" so the QA block still lands on the canvas
and can be filtered to WARNING/FAIL.

The exit code is the contract: 0 published, 1 did not. Point Run Command straight at the venv's
python.exe with this file as the argument -- no .bat wrapper is required. If one is used it must
end with `exit /b %ERRORLEVEL%`, because a batch file returns its LAST command's code and any
stray echo would silently report success.
"""

import os
import re
import sys
from datetime import datetime

# Force UTF-8 on stdout/stderr. The QA block prints REAL DATA -- unmatched BDR_Name__c samples,
# unparseable dates, activity_type values -- and on Windows a redirected stream uses the locale
# encoding (cp1252) until Python 3.15. One accented character in a rep's name would otherwise
# raise UnicodeEncodeError and kill the run at the print, after all the work was done.
# Doing it here rather than via PYTHONUTF8 in a wrapper means the script is safe however it is
# launched. Guarded because stdout can be None under pythonw.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import pandas as pd

# duckdb is not bundled with Alteryx. Run this ONCE in a Python tool cell, then
# restart the Designer before running the workflow:
#
#     from ayx import Alteryx
#     Alteryx.installPackages("duckdb")
#
# Needs DuckDB >= 0.8 for GROUP BY ALL and try_strptime. Verified on 1.5.5.
import duckdb

# ---------------------------------------------------------------------------
# CONFIG  --  the only block you should need to edit
# ---------------------------------------------------------------------------
CONFIG = {
    # ---- inputs -----------------------------------------------------------
    "input_dir":          r"C:\Alteryx\Workflows\Salesforce Workflows\fj360_employee_productivity\parquets",
    "opportunity_file":   "raw_opportunity.parquet",
    "sales_credit_file":  "raw_sales_credit.parquet",
    "user_file":          "raw_users.parquet",
    "events_file":        "raw_events.parquet",
    "tasks_file":         "raw_tasks.parquet",
    "leads_file":         "raw_leads.parquet",
    "bdc_credit_file":    "raw_bdc_credit.parquet",

    # ---- business rules ---------------------------------------------------
    # Salesforce stores opportunity_created_date in UTC. DST-aware conversion,
    # so summer records land on the correct Pacific calendar day.
    "timezone":           "America/Los_Angeles",

    # Opportunities in these stages are excluded entirely: they do not count as
    # created, and they do not lend lead dims to a matching sale. Case- and
    # whitespace-insensitive.
    "excluded_stages":    ["Closed Duplicate"],

    # Which opportunities may lend lead/vehicle dims to a sale, AND how a deal carrying
    # several of them resolves. Most preferred FIRST -- the list position IS the rank.
    #
    # Pool membership is DERIVED from this list, which is why it replaced the old
    # sale_match_stages: a stage cannot be eligible without also having a rank, so the two
    # can never drift apart. A stage absent from here is not eligible at all.
    #
    #   advent_lead False -> the opportunity's own lead_type is NOT 'Advent'
    #                True -> it IS 'Advent' (a DMS-originated opportunity, not a real lead)
    #                None -> either; lead_type is not part of the test for that stage
    #
    # This gates the MATCH ONLY -- opportunities_created still counts every stage except
    # excluded_stages above. A sale whose deal has no eligible opportunity falls into the
    # Advent bucket.
    "sale_match_rank": [
        {"stage": "Finalized",       "advent_lead": False},   # tier 1
        {"stage": "Finalized",       "advent_lead": True},    # tier 2
        {"stage": "Registered Sale", "advent_lead": None},     # tier 3
    ],

    # Opportunities with no salesperson are kept under this synthetic user.
    "unassigned_user_id":   "-1",
    "unassigned_user_name": "unassigned",

    # Tableau filters NULL members out, so a row with a NULL user attribute silently vanishes
    # from every view that groups on it -- which is what was happening to unassigned credit.
    # These fill that gap in OUTPUT 1. They cover all three ways a NULL arises: the synthetic
    # '-1' user, a real user_id absent from raw_users, and a user whose profile_name is blank.
    "unassigned_user_role":   "Salesperson",
    "unassigned_department":  "Unassigned",
    # 'true' so the bucket survives the usual is_active filter. Deliberate, and worth knowing
    # it is a claim about a person who does not exist. A real TERMINATED rep is untouched and
    # still reads 'false' -- these only ever replace a NULL.
    "unassigned_is_active":   "true",

    # Label applied to sales that have no matching opportunity.
    "advent_lead_label":    "Advent",

    # Label applied to TASKS that have no matching opportunity. Unlike the Advent case,
    # only lead_type is labelled -- lead_source and lead_category stay NULL.
    "account_lead_label":   "Account",

    # raw_leads has no lead_type column, so this is a constant on every lead row.
    # lead_source and lead_category come from the lead file itself.
    "ecommerce_lead_label": "E-commerce",

    # A 'Showroom' opportunity whose sale landed LATER than the day the opportunity was created
    # is a come-back rather than a same-visit close, and reports under its own lead type.
    #
    # The comparison uses ADVENT's "rs date" (raw_sales_credit), not the opportunity's own
    # rs_date -- see the two-rs_date rule. Consequence, and it is deliberate: Advent's date lives
    # on a CREDIT row, so only MATCHED SALE rows can flip. The same opportunity's event, task and
    # lead rows keep the plain 'Showroom'. The QA block counts both sides so the split is visible.
    #
    # Strictly later: a same-day sale does not flip. Either date missing does not flip.
    "showroom_lead_label":     "Showroom",
    "showroom_ust_lead_label": "Showroom UST",

    # ---- text dimension canonicalization ----------------------------------
    # Tableau treats 'E-commerce' and 'E-Commerce' as two different members, splitting the
    # same population across two rows in every view that groups on lead type. xwalk_label
    # collapses spellings that differ only by CASE and INTERNAL WHITESPACE onto one form,
    # applied to these columns just before the GROUP BY in output_1 and output_3.
    #
    # 'year' and 'VIN' are absent deliberately: year has no casing, and VIN is passed through
    # untouched by a locked rule.
    "canonical_columns": ("lead_type", "lead_category", "lead_source",
                          "condition", "make", "model"),

    # Which spelling wins. The three label constants above are always preferred, so CONFIG
    # keeps describing the output; add a spelling here to force it to win over the most
    # common one. Anything not listed resolves by row count, then alphabetically.
    #
    # Punctuation is NOT part of the key, so 'Ecommerce' stays separate from 'E-commerce'.
    # The QA block lists those as near-misses rather than merging them -- widening the key
    # could collapse two labels that are deliberately distinct, so it stays a decision.
    "label_overrides":    [],

    # activity_type is matched on its LEADING WORD, lowercased -- 'Call - Outbound',
    # 'call__outbound' and a bare 'Call' all key as 'call'. Direction is deliberately
    # ignored; see the 'direction suffixes seen' QA line for why that is worth watching.
    # To pick up a new channel, add a key here -- the QA block lists any unmapped values.
    "activity_map": {"call": "calls", "email": "emails", "sms": "texts"},

    # Advent stores that legitimately have NO Salesforce opportunities, so their sales can
    # never match one and never inherit a lot. Their credit is still counted -- it publishes
    # under the raw Advent code, which is a real store code, just not a Salesforce one.
    #   CHAUD, AUDFR, FRMBN, FRPOR, CHMBN -- divested; history stays on the books
    #   CATOY                             -- never had Salesforce sales records
    # Listed rather than silently tolerated so the QA block can tell a KNOWN gap apart from
    # a new one. An unlisted store showing up here is worth investigating; add it once
    # confirmed to be another of these.
    "advent_only_stores": ["CHAUD", "AUDFR", "FRMBN", "FRPOR", "CHMBN", "CATOY"],

    # Output 3 population: users whose profile_name CONTAINS this, case-insensitively, so
    # 'BDC Sales Manager' is included alongside 'BDC Sales'. This is a FILTER on every fact
    # branch in Output 3, not just a dimension source -- activity by anyone else is excluded.
    # The QA block lists the profile_name values it actually matched, so a renamed profile
    # shows up as a shrinking population instead of a silently empty output.
    "bdc_profile_match":  "BDC Sales",

    # ---- output shaping ---------------------------------------------------
    # True  -> `date` is a real datetime (Alteryx types it DateTime). Recommended.
    # False -> `date` is an ISO 'YYYY-MM-DD' string (Alteryx types it V_String).
    "date_as_datetime":   True,

    # ---- outputs ----------------------------------------------------------
    # The three datasets are written as PARQUET here, and that is the delivery mechanism --
    # not Alteryx.write. An Alteryx Run Command tool invokes this script with a venv's
    # python.exe (NOT py.exe -- a venv has no py.exe, and the global launcher would use the
    # system interpreter and lose duckdb), then Input Data tools read these files back.
    #
    # Parquet rather than CSV because it preserves the int64/float64/datetime types exactly;
    # a 2.2M-row CSV round trip would hand Tableau strings and cost minutes.
    #
    # Written to a .tmp then os.replace()d, which is atomic on one filesystem -- so a crashed
    # run can never leave a truncated file for a downstream Input Data tool to read.
    "output_dir":         r"C:\Alteryx\Workflows\Salesforce Workflows\fj360_employee_productivity\parquets",
    "output_files": {
        "output_1": "output_1_salesperson.parquet",
        "output_2": "output_2_store.parquet",
        "output_3": "output_3_bdc.parquet",
    },

    # ---- diagnostics ------------------------------------------------------
    "verbose_qa":         True,   # echo the QA block to stdout as it is built
    "strict_checks":      True,   # exit non-zero if the reconciliation fails
    # Run Command discards stdout, so the QA block has to be written down to be seen at all.
    #   dataset_<timestamp>.log  full history, one file per run
    #   last_run.log             fixed path, overwritten -- what to tail or alert on
    #   qa_log.csv               structured, for Run Command's "Read Results from File" so the
    #                            QA block lands on the canvas and can be filtered for WARNING
    "log_dir":            r"N:\onsite_folder\claude_attempt\logs",
    "write_qa_csv":       True,
    # Vestigial. The script no longer depends on running inside Alteryx; leave False unless
    # someone drops it back into a Python tool and wants the old outputs too.
    "write_to_alteryx":   False,
}

# ---------------------------------------------------------------------------
# Alteryx detection. A plain interpreter is now the NORMAL path (Run Command); this only
# still exists so the script keeps working if anyone drops it back into a Python tool.
# ---------------------------------------------------------------------------
try:
    from ayx import Alteryx
    IN_ALTERYX = True
except ImportError:
    Alteryx = None
    IN_ALTERYX = False


# ---------------------------------------------------------------------------
# Expected input columns
# ---------------------------------------------------------------------------
OPP_COLUMNS = [
    "opportunity_id", "opportunity_created_date", "dealercode", "advent_dealer_id",
    "lead_category", "lead_source", "lead_type",
    "vin", "vehicle_year", "vehicle_make", "vehicle_model", "vehicle_condition",
    "StageName", "sale_type", "deal_number", "salesperson_id",
    # bdr_id credits opportunities_created in OUTPUT 3 ONLY. It is deliberately absent from
    # Output 1 -- crediting both people there would emit the measure once per
    # (opportunity, person) pair and make the Output 2 store rollup double count.
    "bdr_id",
    # Salesforce's own retail-sale date, which dates BDC total_sales. NOT the same field as
    # raw_sales_credit's "rs date" (Advent's), despite the identical name -- see sale_date in
    # stg_opp_all. Two systems stamping the same concept.
    "rs_date",
]
SALES_COLUMNS = [
    "dealercode", "dealnumber", "vin", "sp_id", "rs date", "sales_credit",
    # Advent's name for sp_id. Diagnostic only -- it never reaches an output, because a row
    # showing a name while user_id stays '-1' would look attributed without rolling up to that
    # person. diagnose_unmatched_credit.py is what reads it.
    "sp_name",
    # Gross dollars, ALREADY SPLIT to this employee's share upstream -- the source workflow
    # multiplies the deal-level figure by sales_credit before writing. This script therefore
    # does NO arithmetic on them: multiplying again would halve a split deal twice.
    # Required, not optional. load_parquet() fails the run if they are absent, which is the
    # intended behaviour -- the alternative is publishing a dataset of silent zeros.
    "front_gross", "back_gross",
]
USER_COLUMNS = [
    "user_id", "dealercode", "user_name", "department", "IsActive",
    "FirstName", "LastName", "advent_id", "profile_name",
    # tie-breaks the BDC dedupe when one person has several user records
    "SystemModStamp",
]
EVENT_COLUMNS = [
    "event_id", "cancelled", "CreatedById", "appt_created_date",
    "salesperson_id", "show", "appt_scheduled_date", "opportunity_id",
    # Output 3 only. BDR_Name__c is the bridge to the user table -- events carry no BDR id,
    # so this is the one join in the script with no id fallback.
    "BDC_Sales_Scheduled__c", "BDR_Name__c", "Show_Confirmed_by__c",
]
BDC_CREDIT_COLUMNS = ["credit_id", "bdc_id", "bdc_sales_credit", "opportunity_id"]
TASK_COLUMNS = [
    "task_id", "user_id", "opportunity_id", "activity_type",
    "dealercode", "advent_dealer_id", "CompletedDateTime",
]
LEAD_COLUMNS = [
    "lead_id", "opportunity_id", "CreatedDate", "dealercode",
    "VIN", "Year", "make", "model", "condition", "lead_source", "lead_category",
]

# Which raw columns feed each canonicalized output column. xwalk_label is built from exactly
# these, filtered by CONFIG["canonical_columns"], so an empty canonical_columns is a working
# kill switch -- the crosswalk then holds only the label constants, every LEFT JOIN misses, and
# COALESCE passes the raw value through.
#
# Only the two FILE origins appear here. raw_sales_credit, raw_events and raw_tasks supply none
# of these columns: sales inherit them through the opportunity match, events and tasks through
# opp_dim_by_id, leads through stg_leads. The three CONFIG label constants are injected
# separately, which is why the crosswalk is applied at the outputs and not at staging -- see
# section 0.
#
# These are IDENTIFIERS, interpolated into SQL as such. That is not the thing gotcha 1 forbids
# (config VALUES must arrive as registered frames, and do); an identifier cannot be a prepared
# parameter in any case. This is a module constant, never user input.
CANONICAL_SOURCES = {
    "lead_type":     (("raw_opportunity", "lead_type"),),
    "lead_category": (("raw_opportunity", "lead_category"), ("raw_leads", "lead_category")),
    "lead_source":   (("raw_opportunity", "lead_source"),   ("raw_leads", "lead_source")),
    "condition":     (("raw_opportunity", "vehicle_condition"), ("raw_leads", '"condition"')),
    "make":          (("raw_opportunity", "vehicle_make"),  ("raw_leads", "make")),
    "model":         (("raw_opportunity", "vehicle_model"), ("raw_leads", "model")),
}

# Float-valued measures. total_sales carries the 0.5 splits; the two gross columns are dollars.
# finalize() tests this tuple, so a float measure NOT listed here falls through to the
# object-dtype branch and reaches Alteryx as a STRING -- the same trap COUNT_MEASURES guards
# below, and just as invisible until a Tableau sum quietly fails.
MONEY_MEASURES = ("total_sales", "front_gross", "back_gross")

# Integer-valued measures. Any new count measure must be listed here or finalize() will let it
# fall through to the object-dtype branch and it reaches Alteryx as a string.
COUNT_MEASURES = (
    "opportunities_created", "appointments_created", "appointments_scheduled", "shows",
    "calls", "texts", "emails",
    "leads_created", "sold_to_show",
)

QA_LINES = []     # formatted, for the log file and stdout
QA_RECORDS = []   # structured, for qa_log.csv -> the Alteryx canvas
_QA_SECTION = ["(start)"]


def qa(label, value=""):
    """Collect a diagnostic line and echo it immediately."""
    line = f"  {label:<62} {value}"
    QA_LINES.append(line)
    # Kept structured as well as formatted: Run Command reads qa_log.csv back onto the canvas,
    # where a `severity` column is the difference between "browse 200 lines of text" and
    # "filter to WARNING and FAIL". Derived from the label rather than passed in, so every
    # existing qa() call gets it for free.
    stripped = label.strip()
    severity = ("FAIL"    if stripped.startswith("[FAIL]")
                else "PASS"    if stripped.startswith("[PASS]")
                else "WARNING" if stripped.startswith("WARNING")
                else "NOTE"    if stripped.startswith("NOTE")
                else "INFO")
    QA_RECORDS.append({"seq": len(QA_RECORDS) + 1, "section": _QA_SECTION[0],
                       "severity": severity, "label": stripped, "value": str(value)})
    if CONFIG["verbose_qa"]:
        print(line)


def qa_header(text):
    line = f"\n{text}\n{'-' * len(text)}"
    QA_LINES.append(line)
    _QA_SECTION[0] = text.strip()
    if CONFIG["verbose_qa"]:
        print(line)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _norm(name):
    """'rs date' / 'RS_Date' / 'rsdate' all collapse to the same key."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_parquet(filename, expected, label):
    """
    Read a parquet file and map its columns onto the expected names, tolerating
    case and separator differences. Missing columns fail loudly here rather than
    producing a quietly wrong dataset eight steps later.
    """
    path = os.path.join(CONFIG["input_dir"], filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label}: cannot find {path}\n"
            f"  Check CONFIG['input_dir'] and that the network share is mapped "
            f"for the account running this workflow."
        )

    df = pd.read_parquet(path)

    available = {_norm(c): c for c in df.columns}
    missing = [c for c in expected if _norm(c) not in available]
    if missing:
        raise KeyError(
            f"{label} ({filename}) is missing expected column(s): {missing}\n"
            f"  Columns present: {list(df.columns)}"
        )

    df = df[[available[_norm(c)] for c in expected]].copy()
    df.columns = expected

    qa(f"{label}: rows read", f"{len(df):,}")
    return df


def to_pacific_date(series):
    """
    Salesforce UTC timestamp -> naive midnight Pacific date.

    Done in pandas on purpose: it uses stdlib zoneinfo and so avoids any dependency on
    DuckDB's ICU extension, which can fail to load on a network-restricted Alteryx
    server. utc=True handles both '2026-08-05T14:23:00.000Z' and '2026-08-05 14:23:00'.
    Unparseable values become NaT and are counted in QA rather than dropped silently.
    If these files ever outgrow memory, swap to duckdb read_parquet() with
    "INSTALL icu; LOAD icu;" and AT TIME ZONE instead.

    format="mixed" only exists in pandas >= 2. Passing it to pandas 1.x would be treated
    as a literal strptime format and coerce EVERY row to NaT, so gate it on the version
    rather than trusting a try/except to notice.
    """
    mixed = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    parsed = pd.to_datetime(series, utc=True, errors="coerce", **mixed)
    return parsed.dt.tz_convert(CONFIG["timezone"]).dt.normalize().dt.tz_localize(None)


def to_plain_date(series):
    """
    Take the calendar date AS WRITTEN, with no timezone conversion.

    For a Salesforce DATE field (not DateTime) the value is midnight UTC, and putting it through
    to_pacific_date() would roll every row back one day -- which is exactly what happened to the
    opportunity's rs_date: 100% of opportunities carrying both Advent and BDC credit disagreed on
    the sale date by exactly one day. A retail sale date has no time of day to convert.

    tz_localize(None) after a utc=True parse keeps the UTC wall clock rather than shifting it,
    so '2026-07-27T03:30:00Z' stays 2026-07-27 instead of becoming the 26th.
    """
    mixed = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    parsed = pd.to_datetime(series, utc=True, errors="coerce", **mixed)
    return parsed.dt.tz_localize(None).dt.normalize()


def load_inputs():
    qa_header("INPUTS")
    opp = load_parquet(CONFIG["opportunity_file"], OPP_COLUMNS, "raw_opportunity")
    sales = load_parquet(CONFIG["sales_credit_file"], SALES_COLUMNS, "raw_sales_credit")
    users = load_parquet(CONFIG["user_file"], USER_COLUMNS, "raw_users")
    events = load_parquet(CONFIG["events_file"], EVENT_COLUMNS, "raw_events")
    tasks = load_parquet(CONFIG["tasks_file"], TASK_COLUMNS, "raw_tasks")
    leads = load_parquet(CONFIG["leads_file"], LEAD_COLUMNS, "raw_leads")
    bdc = load_parquet(CONFIG["bdc_credit_file"], BDC_CREDIT_COLUMNS, "raw_bdc_credit")

    opp["opp_date"] = to_pacific_date(opp["opportunity_created_date"])
    # NOT to_pacific_date: rs_date is a Salesforce DATE, already the calendar sale date. See
    # to_plain_date. Advent's "rs date" on sales credit is likewise never converted, which is
    # what lets the two be compared at all.
    opp["opp_sale_date"] = to_plain_date(opp["rs_date"])
    events["created_date"] = to_pacific_date(events["appt_created_date"])
    events["sched_date"] = to_pacific_date(events["appt_scheduled_date"])
    tasks["task_date"] = to_pacific_date(tasks["CompletedDateTime"])
    leads["lead_date"] = to_pacific_date(leads["CreatedDate"])

    # --- pin sales_credit to a real float ------------------------------------------
    # Alteryx FixedDecimal lands in parquet as DECIMAL, which pandas returns as
    # object-dtype decimal.Decimal. Registering that leaves DuckDB to INFER a type from
    # sampled values -- it picks a narrow DECIMAL whose precision depends on the data.
    # Coercing to float64 here pins it, so a 0.5 split can never be rounded to an
    # integer or overflow a narrow decimal. errors="coerce" turns anything genuinely
    # unparseable into NaN, which the QA block counts rather than hides.
    sales["sales_credit"] = pd.to_numeric(sales["sales_credit"], errors="coerce").astype("float64")
    # The gross columns are MONEY, so this is the FixedDecimal case in its purest form -- pinning
    # them is mandatory, not defensive. Left as NaN here rather than filled: the NaN is what the
    # QA null counter reads, and stg_sales_raw is where it becomes 0.
    for _g in ("front_gross", "back_gross"):
        sales[_g] = pd.to_numeric(sales[_g], errors="coerce").astype("float64")
    # Same pinning for BDC credit. It arrives as Alteryx Double rather than FixedDecimal, so
    # the decimal-inference trap does not apply -- but pinning costs nothing and means the two
    # credit columns cannot diverge in behaviour later.
    bdc["bdc_sales_credit"] = pd.to_numeric(
        bdc["bdc_sales_credit"], errors="coerce").astype("float64")

    return opp, sales, users, events, tasks, leads, bdc


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def build(con, opp, sales, users, events, tasks, leads, bdc):
    con.register("raw_opportunity", opp)
    con.register("raw_sales_credit", sales)
    con.register("raw_bdc_credit", bdc)
    con.register("raw_users", users)
    con.register("raw_events", events)
    con.register("raw_tasks", tasks)
    con.register("raw_leads", leads)

    # Config values are passed as DATA, never string-interpolated into SQL. DuckDB
    # rejects prepared parameters inside CREATE VIEW / CREATE TABLE, so the views
    # below read them back as scalar subqueries against these registered frames.
    con.register(
        "cfg_excluded_stages",
        pd.DataFrame({"stage": [str(s).strip().upper() for s in CONFIG["excluded_stages"]]}),
    )
    # (stage, advent flag) -> tier. Expanded HERE rather than in SQL: advent_lead=None becomes
    # both 'y' and 'n' rows sharing the tier, so the lookup is a plain equality join with
    # exactly one match per opportunity -- no aggregation, no correlated subquery (gotcha 4).
    # Keyed by dict so an earlier (more preferred) entry wins if a pair is listed twice.
    _tiers = {}
    for _i, _r in enumerate(CONFIG["sale_match_rank"], start=1):
        _flags = ("y", "n") if _r["advent_lead"] is None else ("y" if _r["advent_lead"] else "n",)
        for _f in _flags:
            _tiers.setdefault((str(_r["stage"]).strip().upper(), _f), _i)
    con.register("cfg_match_rank", pd.DataFrame(
        [{"stage": k[0], "advent": k[1], "tier": v} for k, v in _tiers.items()]
    ))
    con.register("cfg", pd.DataFrame([{
        "unassigned_user_id":   str(CONFIG["unassigned_user_id"]),
        "unassigned_user_name": str(CONFIG["unassigned_user_name"]),
        "unassigned_user_role":  str(CONFIG["unassigned_user_role"]),
        "unassigned_department": str(CONFIG["unassigned_department"]),
        "unassigned_is_active":  str(CONFIG["unassigned_is_active"]),
        "advent_lead_label":    str(CONFIG["advent_lead_label"]),
        "account_lead_label":   str(CONFIG["account_lead_label"]),
        "ecommerce_lead_label": str(CONFIG["ecommerce_lead_label"]),
        "showroom_lead_label":     str(CONFIG["showroom_lead_label"]),
        "showroom_ust_lead_label": str(CONFIG["showroom_ust_lead_label"]),
        # Matched case- and whitespace-insensitively, because canonicalization runs LATER (at
        # the output). A raw 'SHOWROOM' would otherwise slip past the flip test and only then
        # be tidied to 'Showroom', so the come-back would go uncounted.
        "showroom_key": re.sub(r"\s+", " ", str(CONFIG["showroom_lead_label"]).strip()).lower(),
        # Same normalization, for the sale-match ranking's Advent test. Needed for the same
        # reason: an opportunity whose raw lead_type is 'advent' must rank as Advent even
        # though canonicalization does not tidy it to 'Advent' until the output.
        "advent_key": re.sub(r"\s+", " ", str(CONFIG["advent_lead_label"]).strip()).lower(),
        "bdc_profile_match":    str(CONFIG["bdc_profile_match"]),
    }]))
    con.register("cfg_activity_map", pd.DataFrame(
        [{"channel": str(k).strip().lower(), "measure": str(v)}
         for k, v in CONFIG["activity_map"].items()]
    ))
    # Spellings that win the canonicalization tiebreak outright. The three label constants are
    # always included so CONFIG keeps describing the output: without them a Salesforce
    # 'E-Commerce' outnumbering the injected 'E-commerce' would silently rename the label.
    # dict.fromkeys dedupes while preserving order -- a duplicate here would fan out the
    # LEFT JOIN in xwalk_label and double every n_rows.
    con.register("cfg_labels", pd.DataFrame({"label": list(dict.fromkeys(
        [str(CONFIG["advent_lead_label"]), str(CONFIG["account_lead_label"]),
         str(CONFIG["ecommerce_lead_label"]),
         # Both Showroom labels, so the flip's output spelling is the canonical one. The UST
         # label is DERIVED, not read from a file, so it would otherwise be absent from the
         # crosswalk and a raw 'showroom ust' in the data would not converge onto it.
         str(CONFIG["showroom_lead_label"]), str(CONFIG["showroom_ust_lead_label"])]
        + [str(x).strip() for x in CONFIG["label_overrides"] if str(x).strip()]
    ))}))

    # -- 0. label canonicalization ------------------------------------------
    # Tableau splits 'E-commerce' and 'E-Commerce' into two members. This maps every observed
    # spelling of a text dimension onto one canonical form, keyed on case and internal
    # whitespace, and output_1 / output_3 apply it immediately before their GROUP BY.
    #
    # Built here because it reads the raw frames and nothing else depends on it.
    #
    # The key is GLOBAL, not per column: one string renders one way everywhere, so a value
    # appearing as both a lead_source and a condition cannot come out spelled two ways. The
    # consequence is that frequency in one column can decide the winner for another, which is
    # why CONFIG["label_overrides"] exists.
    #
    # TABLE, not VIEW: it contains window functions and is referenced twelve times (gotcha 3).
    observed = "\n            UNION ALL ".join(
        # Constants first, and unconditionally, so this is never an empty UNION when
        # canonical_columns is empty.
        ["SELECT label AS raw FROM cfg_labels"]
        + [f"SELECT NULLIF(TRIM(CAST({col} AS VARCHAR)), '') FROM {tbl}"
           for name in CONFIG["canonical_columns"]
           for tbl, col in CANONICAL_SOURCES.get(name, ())]
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE xwalk_label AS
        WITH observed AS (
            {observed}
        ),
        keyed AS (
            -- raw is already trimmed, so collapsing internal runs is all that is left.
            -- BACKSLASH IS DOUBLED: a bare \\s in a non-raw Python string raises
            -- SyntaxWarning, and the run is compiled under -W error::SyntaxWarning.
            SELECT o.raw,
                   LOWER(REGEXP_REPLACE(o.raw, '\\s+', ' ', 'g'))      AS label_key,
                   COUNT(*)                                            AS n_rows,
                   MAX(CASE WHEN p.label IS NULL THEN 0 ELSE 1 END)    AS is_preferred
            FROM observed o
            LEFT JOIN cfg_labels p ON p.label = o.raw
            WHERE o.raw IS NOT NULL
            GROUP BY ALL
        ),
        ranked AS (
            -- A preferred spelling wins; then the most common; then alphabetically, so the
            -- choice is deterministic across runs rather than depending on scan order.
            SELECT label_key, raw,
                   ROW_NUMBER() OVER (PARTITION BY label_key
                                      ORDER BY is_preferred DESC, n_rows DESC, raw) AS rn,
                   COUNT(*)   OVER (PARTITION BY label_key) AS n_spellings
            FROM keyed
        )
        SELECT k.raw           AS raw_label,
               c.raw           AS canonical_label,
               k.label_key     AS label_key,
               -- What the key WOULD be if punctuation and spaces were ignored too. Not used to
               -- merge anything -- QA reads it to report near-misses ('Ecommerce' beside
               -- 'E-commerce') so the next tier of drift is visible without acting on it.
               REGEXP_REPLACE(LOWER(k.raw), '[^a-z0-9]', '', 'g') AS loose_key,
               k.n_rows        AS n_rows,
               c.n_spellings   AS n_spellings
        FROM keyed k
        JOIN ranked c ON c.label_key = k.label_key AND c.rn = 1
    """)

    # -- 1. users -----------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW stg_users AS
        SELECT
            NULLIF(TRIM(CAST(user_id AS VARCHAR)), '')                AS user_id,
            UPPER(NULLIF(TRIM(CAST(dealercode AS VARCHAR)), ''))      AS user_dealercode,
            COALESCE(
                NULLIF(TRIM(CAST(user_name AS VARCHAR)), ''),
                NULLIF(TRIM(CONCAT_WS(' ',
                    NULLIF(TRIM(CAST(FirstName AS VARCHAR)), ''),
                    NULLIF(TRIM(CAST(LastName  AS VARCHAR)), '')
                )), '')
            )                                                          AS user_name,
            NULLIF(TRIM(CAST(profile_name AS VARCHAR)), '')            AS user_role,
            NULLIF(TRIM(CAST(department   AS VARCHAR)), '')            AS department,
            CASE
                WHEN LOWER(TRIM(CAST(IsActive AS VARCHAR))) IN ('true','1','t','yes','y')  THEN 'true'
                WHEN LOWER(TRIM(CAST(IsActive AS VARCHAR))) IN ('false','0','f','no','n') THEN 'false'
                ELSE NULLIF(TRIM(CAST(IsActive AS VARCHAR)), '')
            END                                                        AS is_active,
            UPPER(NULLIF(TRIM(CAST(advent_id AS VARCHAR)), ''))        AS advent_id,
            -- Only used to tie-break the Output 3 BDC dedupe. Kept as text and sorted as text:
            -- a Salesforce SystemModStamp is ISO-8601, so lexical order is chronological, and
            -- parsing it would add a failure mode for no benefit.
            NULLIF(TRIM(CAST(SystemModStamp AS VARCHAR)), '')          AS system_mod_stamp,
            -- normalized person key. Events carry no BDR id, so BDR_Name__c joins on this;
            -- collapsing whitespace and case is the most we can do without a real key.
            UPPER(TRIM(regexp_replace(
                COALESCE(
                    NULLIF(TRIM(CAST(user_name AS VARCHAR)), ''),
                    NULLIF(TRIM(CONCAT_WS(' ',
                        NULLIF(TRIM(CAST(FirstName AS VARCHAR)), ''),
                        NULLIF(TRIM(CAST(LastName  AS VARCHAR)), '')
                    )), '')
                -- Backslash is DOUBLED because this is a non-raw Python string: a single one
                -- before 's' is an invalid escape. It passes through today but Python already
                -- warns and will make it a SyntaxError. DuckDB receives one backslash either
                -- way. Same in stg_events.bdr_name_key -- both sides must normalize alike.
                ), '\\s+', ' ', 'g')))                                 AS name_key
        FROM raw_users
        WHERE NULLIF(TRIM(CAST(user_id AS VARCHAR)), '') IS NOT NULL
    """)

    # One row per salesforce user_id. If a user_id repeats (e.g. one row per store)
    # a naive join would multiply every measure, so collapse deterministically --
    # preferring the active record -- and report it below.
    con.execute("""
        CREATE OR REPLACE TABLE dim_user AS
        SELECT user_id, user_name, user_role, department, is_active
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY (is_active = 'true') DESC,
                         user_dealercode NULLS LAST,
                         user_name NULLS LAST
            ) AS rn
            FROM stg_users
        )
        WHERE rn = 1
    """)

    # NOTE: xwalk_advent is built in section 2b, not here -- it needs the store crosswalk,
    # which is derived from the opportunity file staged in section 2.

    # -- 2. opportunities ---------------------------------------------------
    # Normalized ONCE here with no stage filter. stg_opp (stage-filtered) drives
    # opportunities_created and the sale-match pool; opp_dim_for_events uses this
    # unfiltered version, because events may legitimately hang off an opportunity in a
    # stage we exclude from counting.
    #
    # A TABLE, not a view: this is scanned by the fact, both dim pools and the QA
    # block, and re-deriving it each time is what makes the whole script crawl.
    con.execute("""
        CREATE OR REPLACE TABLE stg_opp_all AS
        SELECT
            NULLIF(TRIM(CAST(opportunity_id AS VARCHAR)), '')          AS opportunity_id,
            -- TWO store columns, deliberately. dealercode is the SALESFORCE LOT and is what
            -- both outputs report on. advent_dealer_id is the coarser ADVENT store: one
            -- rooftop is two lots in Salesforce (LVMBN / L2MBN) that Advent does not
            -- distinguish, so sales credit arrives on a single code covering both. Sales
            -- therefore JOIN on advent_dealer_id but are REPORTED on dealercode, which is
            -- what keeps the lots separate. Identical for every other store.
            -- The COALESCE is insurance, and currently never fires: advent_dealer_id is fully
            -- populated on real data. Kept because a blank would otherwise drop that
            -- opportunity out of the sale-match pool entirely, silently costing credit that
            -- matched fine before this column existed. For an unsplit store the two codes are
            -- identical by definition, so falling back to the lot IS the correct Advent store;
            -- it is only a guess at a split rooftop, which the blank-coverage QA line reports.
            UPPER(NULLIF(TRIM(CAST(dealercode AS VARCHAR)), ''))       AS dealercode,
            COALESCE(UPPER(NULLIF(TRIM(CAST(advent_dealer_id AS VARCHAR)), '')),
                     UPPER(NULLIF(TRIM(CAST(dealercode       AS VARCHAR)), '')))
                                                                       AS advent_dealer_id,
            -- kept separately so QA can measure the coverage gap after the fallback hides it
            (NULLIF(TRIM(CAST(advent_dealer_id AS VARCHAR)), '') IS NULL) AS advent_id_missing,
            CAST(opp_date AS DATE)                                     AS activity_date,
            -- Salesforce's own retail-sale date, dating BDC total_sales in Output 3. Named
            -- sale_date rather than rs_date on purpose: raw_sales_credit has its OWN "rs date"
            -- (Advent's) which dates Output 1's total_sales. Same concept, two systems, and
            -- leaving both called rs_date is how the wrong one eventually gets joined.
            CAST(opp_sale_date AS DATE)                                AS sale_date,
            COALESCE(NULLIF(TRIM(CAST(salesperson_id AS VARCHAR)), ''),
                     (SELECT unassigned_user_id FROM cfg))         AS user_id,
            -- Output 3 only; see OPP_COLUMNS for why this never reaches Output 1.
            COALESCE(NULLIF(TRIM(CAST(bdr_id AS VARCHAR)), ''),
                     (SELECT unassigned_user_id FROM cfg))         AS bdr_id,
            NULLIF(TRIM(CAST(deal_number AS VARCHAR)), '')             AS deal_key,
            CAST(vin AS VARCHAR)                                       AS vin,
            NULLIF(TRIM(CAST(vehicle_condition AS VARCHAR)), '')       AS "condition",
            NULLIF(TRIM(CAST(vehicle_make  AS VARCHAR)), '')           AS make,
            NULLIF(TRIM(CAST(vehicle_model AS VARCHAR)), '')           AS model,
            NULLIF(TRIM(CAST(vehicle_year  AS VARCHAR)), '')           AS "year",
            NULLIF(TRIM(CAST(lead_type     AS VARCHAR)), '')           AS lead_type,
            NULLIF(TRIM(CAST(lead_category AS VARCHAR)), '')           AS lead_category,
            NULLIF(TRIM(CAST(lead_source   AS VARCHAR)), '')           AS lead_source,
            NULLIF(TRIM(CAST(StageName     AS VARCHAR)), '')           AS stage_name
        FROM raw_opportunity
    """)

    # Excluded stages are dropped here, so they neither count as created nor lend lead
    # dims to a SALE. COALESCE keeps a NULL StageName instead of letting three-valued
    # logic swallow the row.
    con.execute("""
        CREATE OR REPLACE TABLE stg_opp AS
        SELECT * FROM stg_opp_all
        WHERE COALESCE(UPPER(stage_name), '') NOT IN (SELECT stage FROM cfg_excluded_stages)
    """)

    # NOTE: opp_dim_by_id -- the one-row-per-opportunity dim source for events, tasks and leads
    # -- used to be built HERE. It now lives in section 4a', after the sale match, because the
    # Showroom UST flip is decided by Advent's sale date and that is not known until the match
    # has run. Nothing between here and section 4b reads it.

    # -- 2b. store crosswalk ------------------------------------------------
    # The opportunity file carries BOTH store columns, so it is its own crosswalk. Note the
    # asymmetry, which is the whole reason this section exists: lot -> Advent store is a
    # function (many lots to one store), Advent store -> lot is not.
    con.execute("""
        CREATE OR REPLACE TABLE store_pairs AS
        SELECT DISTINCT dealercode, advent_dealer_id
        FROM stg_opp_all
        WHERE dealercode IS NOT NULL AND advent_dealer_id IS NOT NULL
    """)

    # lot -> Advent store. One row per dealercode, because that direction is a function.
    # Deduped defensively: a lot mapping to two Advent stores would break the premise, and
    # reconcile() asserts it never happens.
    con.execute("""
        CREATE OR REPLACE TABLE xwalk_lot_to_advent AS
        SELECT dealercode, advent_dealer_id FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY dealercode
                                         ORDER BY advent_dealer_id) AS rn
            FROM store_pairs
        )
        WHERE rn = 1
    """)

    # Advent store -> PRIMARY lot, used only when a sale matched no opportunity and so has
    # no lot to inherit. Prefers the lot whose own code equals the Advent id, which makes
    # this the identity mapping for every store that is not split -- and for a split store
    # reproduces the pre-change behaviour rather than inventing a new store code.
    #
    # The UNION adds self-pairs for lots that are not themselves anyone's advent_dealer_id.
    # Real data has sales arriving on codes that ARE Salesforce lot codes but whose
    # opportunities carry a different advent_dealer_id; without this those sales fall through
    # to the raw Advent code. A self-pair can only ever add a resolution, never redirect an
    # existing one, because it is excluded wherever the code is already an Advent key.
    con.execute("""
        CREATE OR REPLACE TABLE xwalk_advent_to_lot AS
        WITH pairs AS (
            SELECT dealercode, advent_dealer_id FROM store_pairs
            UNION
            SELECT dealercode, dealercode FROM store_pairs
            WHERE dealercode NOT IN (SELECT advent_dealer_id FROM store_pairs)
        )
        SELECT advent_dealer_id, primary_dealercode, n_lots
        FROM (
            SELECT advent_dealer_id,
                   dealercode AS primary_dealercode,
                   COUNT(*) OVER (PARTITION BY advent_dealer_id) AS n_lots,
                   ROW_NUMBER() OVER (
                       PARTITION BY advent_dealer_id
                       ORDER BY (dealercode = advent_dealer_id) DESC, dealercode
                   ) AS rn
            FROM pairs
        )
        WHERE rn = 1
    """)

    # -- 2c. the BDC population (Output 3) ----------------------------------
    # One employee sometimes has SEVERAL user records -- a retired login plus a current one.
    # The person, not the login, is the reporting grain, so rank within name_key and keep the
    # active/most-recently-modified record.
    con.execute("""
        CREATE OR REPLACE TABLE bdc_users AS
        SELECT *, ROW_NUMBER() OVER (
                   PARTITION BY name_key
                   ORDER BY (is_active = 'true') DESC,
                            system_mod_stamp DESC NULLS LAST,
                            user_id
               ) AS rn
        FROM (
            SELECT DISTINCT user_id, user_name, user_role, department, is_active,
                            system_mod_stamp, name_key
            FROM stg_users
            WHERE name_key IS NOT NULL
              AND user_role ILIKE '%' || (SELECT bdc_profile_match FROM cfg) || '%'
        )
    """)

    con.execute("""
        CREATE OR REPLACE TABLE dim_bdc AS
        SELECT user_id, user_name, user_role, department, is_active, name_key
        FROM bdc_users WHERE rn = 1
    """)

    # The crosswalk is what makes that dedupe apply to FACTS, not just to the dimension.
    # Without it a task or credit row on a RETIRED login finds no dim row and -- because the
    # BDC list is a filter, not a fallback -- is dropped outright, silently undercounting that
    # person. Mapping every one of their ids onto the winner is many-to-one, so it can only
    # ever merge rows, never multiply them: THE INVARIANT is safe.
    con.execute("""
        CREATE OR REPLACE TABLE xwalk_bdc_user AS
        SELECT DISTINCT b.user_id, d.user_id AS bdc_user_id
        FROM bdc_users b
        JOIN dim_bdc d ON d.name_key = b.name_key
    """)

    # (Advent store, advent_id) -> salesforce user_id crosswalk for the sales-credit join.
    # Advent salesperson ids are only unique WITHIN a store, so advent_id alone can bind
    # a sale to a different person who happens to share that id at another rooftop.
    #
    # The store is translated into ADVENT space first. This was thought unnecessary because
    # reps are given the primary lot -- but real data has 10 users sitting at a non-primary
    # lot, and for them an untranslated join silently sends their credit to 'unassigned'
    # rather than erroring. COALESCE keeps a user whose store appears on no opportunity
    # behaving exactly as before. Identity for every unsplit store.
    con.execute("""
        CREATE OR REPLACE TABLE xwalk_advent AS
        SELECT advent_store, advent_id, user_id
        FROM (
            SELECT COALESCE(w.advent_dealer_id, u.user_dealercode) AS advent_store,
                   u.advent_id, u.user_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(w.advent_dealer_id, u.user_dealercode),
                                    u.advent_id
                       ORDER BY (u.is_active = 'true') DESC, u.user_id
                   ) AS rn
            FROM stg_users u
            LEFT JOIN xwalk_lot_to_advent w ON w.dealercode = u.user_dealercode
            WHERE u.advent_id IS NOT NULL AND u.user_dealercode IS NOT NULL
        )
        WHERE rn = 1
    """)

    # -- 3. sales credit ----------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW stg_sales_raw AS
        WITH src AS (
            SELECT
                UPPER(NULLIF(TRIM(CAST(dealercode AS VARCHAR)), '')) AS dealercode,
                NULLIF(TRIM(CAST(dealnumber AS VARCHAR)), '')        AS deal_key,
                CAST(vin AS VARCHAR)                                 AS vin,
                UPPER(NULLIF(TRIM(CAST(sp_id AS VARCHAR)), ''))      AS sp_id,
                -- Trimmed but NOT upper-cased: it is a display name and the point is that it
                -- reads properly in the diagnostic CSV.
                NULLIF(TRIM(CAST(sp_name AS VARCHAR)), '')           AS sp_name,
                TRIM(CAST("rs date" AS VARCHAR))                     AS rs_raw,
                TRY_CAST(sales_credit AS DOUBLE)                     AS sales_credit,
                -- Already this employee's share; see SALES_COLUMNS. Nothing here multiplies.
                -- COALESCE to 0, and note what is NOT happening: a null gross must never gate
                -- the row the way a null sales_credit does below. Dropping a sale because its
                -- gross is missing would destroy total_sales. QA counts the nulls instead.
                COALESCE(TRY_CAST(front_gross AS DOUBLE), 0)         AS front_gross,
                COALESCE(TRY_CAST(back_gross  AS DOUBLE), 0)         AS back_gross,
                -- Pre-COALESCE, purely so QA can measure upstream coverage of the new columns
                -- once the 0s are indistinguishable from real zeros.
                (TRY_CAST(front_gross AS DOUBLE) IS NULL)            AS front_gross_missing,
                (TRY_CAST(back_gross  AS DOUBLE) IS NULL)            AS back_gross_missing
            FROM raw_sales_credit
        )
        SELECT
            dealercode, deal_key, vin, sp_id, sp_name, rs_raw, sales_credit,
            front_gross, back_gross, front_gross_missing, back_gross_missing,
            CAST(COALESCE(
                TRY_STRPTIME(rs_raw, '%Y-%m-%d'),
                TRY_STRPTIME(rs_raw, '%m/%d/%Y'),
                TRY_STRPTIME(rs_raw, '%m/%d/%y'),
                TRY_STRPTIME(rs_raw, '%Y-%m-%d %H:%M:%S'),
                TRY_STRPTIME(rs_raw, '%m/%d/%Y %H:%M:%S'),
                TRY_CAST(rs_raw AS TIMESTAMP)
            ) AS DATE) AS activity_date
        FROM src
    """)

    # A credit row with no parseable date cannot be placed on a timeline, and one
    # with no credit value contributes nothing. Both are excluded and counted --
    # never dropped silently.
    # sales_row_id gives every physical credit row its own identity. The opportunity
    # pick below partitions on it, so two legitimately identical 0.5 rows stay two rows.
    con.execute("""
        CREATE OR REPLACE TABLE stg_sales AS
        SELECT ROW_NUMBER() OVER () AS sales_row_id, *
        FROM stg_sales_raw
        WHERE activity_date IS NOT NULL AND sales_credit IS NOT NULL
    """)

    # -- 4. sales borrow lead/vehicle dims from the opportunity -------------
    # Resolve the credited Advent salesperson to a Salesforce user first, because the
    # opportunity pick below prefers the opportunity owned by that same person.
    # deal_has_any_opp lets the QA block separate the two causes of an Advent tag
    # (no opportunity at all vs. one exists but is not in an eligible stage) with a
    # plain flag instead of a correlated subquery per row.
    con.execute("""
        CREATE OR REPLACE TABLE sales_resolved AS
        SELECT
            s.sales_row_id, s.dealercode, s.deal_key, s.vin,
            s.activity_date, s.sales_credit,
            s.front_gross, s.back_gross,
            COALESCE(x.user_id, (SELECT unassigned_user_id FROM cfg)) AS user_id,
            (d.deal_key IS NOT NULL)                                  AS deal_has_any_opp,
            -- true when the sp_id is a known Advent id but not at THIS store, i.e. the
            -- row would have matched on advent_id alone. Quantifies what adding
            -- dealercode to the join costs.
            (x.user_id IS NULL AND a.advent_id IS NOT NULL)            AS sp_id_other_store
        FROM stg_sales s
        LEFT JOIN xwalk_advent x
               ON x.advent_id = s.sp_id AND x.advent_store = s.dealercode
        LEFT JOIN (SELECT DISTINCT advent_id FROM xwalk_advent) a
               ON a.advent_id = s.sp_id
        LEFT JOIN (SELECT DISTINCT advent_dealer_id, deal_key FROM stg_opp
                   WHERE deal_key IS NOT NULL) d
               ON d.advent_dealer_id = s.dealercode AND d.deal_key = s.deal_key
    """)

    # Only opportunities ranked in sale_match_rank are eligible to lend dims to a sale.
    # This is a match-side filter only -- these same rows still count toward
    # opportunities_created via stg_opp, whatever their stage.
    con.execute("""
        CREATE OR REPLACE TABLE opp_match_pool AS
        SELECT o.*, k.tier AS match_tier
        FROM stg_opp o
        -- INNER join, which does membership AND ranking in one step: a stage with no entry in
        -- sale_match_rank is simply not eligible, so the two cannot drift apart. Exactly one
        -- row matches per opportunity, because cfg_match_rank is keyed on (stage, advent).
        JOIN cfg_match_rank k
          ON k.stage  = COALESCE(UPPER(o.stage_name), '')
        -- Case- and whitespace-insensitive, for the same reason showroom_key is: a raw
        -- 'advent' would otherwise rank as non-Advent and only be tidied to 'Advent' later,
        -- at the output. A NULL lead_type counts as NON-Advent; QA reports the tier split.
         AND k.advent = CASE WHEN LOWER(REGEXP_REPLACE(COALESCE(o.lead_type, ''),
                                                       '\\s+', ' ', 'g'))
                                  = (SELECT advent_key FROM cfg) THEN 'y' ELSE 'n' END
        WHERE o.deal_key IS NOT NULL
    """)

    # Join on advent_dealer_id + dealnumber only -- opportunity VIN is unreliable, and VIN on
    # every sale row comes from sales_credit instead. The store side of the key is the ADVENT
    # store, not the lot, because sales credit only ever knows the coarser code.
    #
    # A deal can carry more than one Finalized opportunity, so this is a ranked join,
    # not a plain one: exactly one opportunity is chosen per credit row. Without the
    # ROW_NUMBER filter a 3-opportunity deal would emit that credit 3 times and triple
    # total_sales. At a split rooftop the candidate pool now spans BOTH lots, which makes
    # the tiebreak load-bearing in a second way: it no longer just picks the dims, it picks
    # the LOT the sale is reported under. Advent being one system across both lots means a
    # deal number should exist at only one of them; this is the guard for when it does not.
    # Preference order -- the TIER decides first, so stage and lead type outrank everything:
    #   1. match_tier from sale_match_rank: Finalized non-Advent, then Finalized Advent, then
    #      Registered Sale. A real lead-sourced sale beats a DMS-originated one.
    #   2. the opportunity owned by the salesperson who actually got the credit. Retained
    #      BELOW the tier, which is what keeps the split-rooftop lot decision stable.
    #   3. LATEST opportunity_created_date -- the most recent lead, not the first. This
    #      REVERSED; it used to prefer the earliest.
    #   4. opportunity_id -- so the result is deterministic run to run
    #
    # Materialized as a TABLE holding only the winning row. The window functions are
    # the most expensive step in the script, so they must run exactly once -- as a view
    # this got re-evaluated by every downstream reference, including each QA query.
    # n_candidates and opp_user_id are carried through so the QA block can describe the
    # tiebreak without touching the losing candidates again.
    con.execute("""
        CREATE OR REPLACE TABLE sales_matched AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT
                r.sales_row_id, r.activity_date, r.user_id, r.vin,
                r.sales_credit, r.front_gross, r.back_gross,
                r.deal_has_any_opp, r.sp_id_other_store,
                -- BOTH stores are carried forward. advent_store is what the credit row
                -- arrived on and is the join key; opp_dealercode is the lot it will be
                -- REPORTED on. They differ only at a split rooftop.
                r.dealercode      AS advent_store,
                o.dealercode      AS opp_dealercode,
                o.opportunity_id,
                o.user_id AS opp_user_id,
                -- The opportunity's CREATED date, aliased because r.activity_date is already
                -- the sale date (Advent's "rs date"). opp_sold_same_day compares the two to
                -- decide the Showroom / Showroom UST split.
                o.activity_date AS opp_created_date,
                o.match_tier,
                o.stage_name AS opp_stage_name,
                o."condition", o.make, o.model, o."year",
                o.lead_type, o.lead_category, o.lead_source,
                COUNT(o.opportunity_id) OVER (PARTITION BY r.sales_row_id) AS n_candidates,
                ROW_NUMBER() OVER (
                    PARTITION BY r.sales_row_id
                    ORDER BY o.match_tier             ASC  NULLS LAST,
                             (o.user_id = r.user_id)  DESC NULLS LAST,
                             o.activity_date          DESC NULLS LAST,
                             o.opportunity_id         ASC  NULLS LAST
                ) AS rn
            FROM sales_resolved r
            LEFT JOIN opp_match_pool o
                   ON o.advent_dealer_id = r.dealercode
                  AND o.deal_key         = r.deal_key
        )
        WHERE rn = 1
    """)

    # -- 4a'. Showroom -> Showroom UST, decided ONCE per opportunity --------
    # The flip is a property of the OPPORTUNITY, not of the sale row, so every fact derived from
    # it agrees: the sale, its appointments, its calls. That is why this sits between the sale
    # match and opp_dim_by_id -- Advent's sale date is not known any earlier, and opp_dim_by_id
    # is what feeds events and tasks.
    #
    # NOTE THE INVERSION: this table lists the opportunities that KEEP the plain label, and
    # everything else flips. There are two reasons to flip -- sold later than created, or never
    # sold at all -- and only one reason not to, so enumerating the exception is both smaller and
    # impossible to get half-right. Plain 'Showroom' therefore means exactly one thing: the deal
    # closed on the visit.
    #
    # MIN(activity_date): a split deal is several credit rows carrying the same date, so MIN is
    # a no-op there. It only bites if one opportunity somehow has credit on two different dates,
    # and then the EARLIEST sale decides -- if any part of the deal closed on the visit, it was
    # not a come-back.
    #
    # Only Finalized opportunities reach the match pool, so a Closed Duplicate is never in here
    # and always flips -- correct, since it never sold.
    con.execute("""
        CREATE OR REPLACE TABLE opp_sold_same_day AS
        SELECT opportunity_id
        FROM sales_matched
        WHERE opportunity_id IS NOT NULL
        GROUP BY opportunity_id
        -- Both dates required EXPLICITLY rather than relying on NULL comparison semantics: an
        -- opportunity with no created date cannot be shown to have closed on the visit, so it
        -- belongs on the flip side. QA counts that case separately.
        HAVING MIN(activity_date)     IS NOT NULL
           AND MIN(opp_created_date)  IS NOT NULL
           AND MIN(activity_date) <= MIN(opp_created_date)
    """)

    # The relabel itself, defined ONCE. Three branches need it -- opportunities_created reads
    # stg_opp, events and tasks read opp_dim_by_id, and sales read sales_matched -- and three
    # hand-copied CASE expressions is three chances for one of them to drift and start
    # reporting the same opportunity under two lead types.
    #
    # The key comparison is case- and whitespace-insensitive because canonicalization runs
    # later, at the output: a raw 'SHOWROOM' would otherwise slip past this and only then be
    # tidied to 'Showroom', leaving the come-back uncounted.
    con.execute("""
        CREATE OR REPLACE MACRO ust_label(lt, is_ust) AS
            CASE WHEN is_ust
                      AND LOWER(REGEXP_REPLACE(lt, '\\s+', ' ', 'g'))
                          = (SELECT showroom_key FROM cfg)
                 THEN (SELECT showroom_ust_lead_label FROM cfg)
                 ELSE lt END
    """)

    # One row per opportunity_id, shared by the event, task and lead joins. This dedupe is
    # load-bearing: many events and tasks point at a single opportunity, so a duplicated
    # opportunity_id here would fan out and inflate every appointment and activity measure.
    #
    # Moved here from section 2 so it can carry the Showroom UST flip. The LEFT JOIN is to a
    # table of at most one row per opportunity, so it cannot fan out.
    con.execute("""
        CREATE OR REPLACE TABLE opp_dim_by_id AS
        SELECT d.opportunity_id, d.dealercode, d.vin, d."condition", d.make, d.model, d."year",
               ust_label(d.lead_type, u.opportunity_id IS NULL) AS lead_type,
               d.lead_category, d.lead_source,
               d.user_id,  -- the opportunity's salesperson; leads are credited to them
               d.bdr_id,   -- and its BDR, which is who the SAME lead credits in Output 3
               d.sale_date, d.stage_name   -- both for the Output 3 BDC credit branch
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY opportunity_id
                ORDER BY activity_date NULLS LAST
            ) AS rn
            FROM stg_opp_all
            WHERE opportunity_id IS NOT NULL
        ) d
        LEFT JOIN opp_sold_same_day u ON u.opportunity_id = d.opportunity_id
        WHERE d.rn = 1
    """)

    # The reported store resolves in three tiers, most specific first:
    #   1. the matched opportunity's LOT -- the point of the whole change
    #   2. the primary lot for the Advent store, when nothing matched and there is no lot to
    #      inherit. Identity for every unsplit store; at a split rooftop this is credit that
    #      genuinely cannot be attributed to a lot, and QA counts and prices it.
    #   3. the raw Advent code, when that store appears on no opportunity at all. Should never
    #      fire; it exists so an unknown store degrades to today's behaviour instead of
    #      dropping the row, and reconcile() asserts it is empty.
    con.execute("""
        CREATE OR REPLACE VIEW sales_with_dims AS
        SELECT
            COALESCE(m.opp_dealercode, w.primary_dealercode, m.advent_store) AS dealercode,
            m.activity_date,
            m.user_id,
            m.vin,
            m."condition", m.make, m.model, m."year",
            -- Orphan label first; otherwise the opportunity's own, relabelled if this is a
            -- come-back. Both the test and the relabel live in opp_sold_same_day /
            -- ust_label, so a sale
            -- row can never disagree with the event and task rows for the same opportunity.
            CASE WHEN m.opportunity_id IS NULL
                      THEN (SELECT advent_lead_label FROM cfg)
                 ELSE ust_label(m.lead_type, ust.opportunity_id IS NULL)
                 END                                                           AS lead_type,
            CASE WHEN m.opportunity_id IS NULL
                 THEN NULL                                ELSE m.lead_category END AS lead_category,
            CASE WHEN m.opportunity_id IS NULL
                 THEN (SELECT advent_lead_label FROM cfg) ELSE m.lead_source   END AS lead_source,
            m.sales_credit,
            -- Carried through the store-fallback and orphan-label logic untouched. Unmatched
            -- credit keeps its gross and lands on the 'Advent' label with it, which is what
            -- keeps the dollars whole even when the opportunity match fails.
            m.front_gross, m.back_gross
        FROM sales_matched m
        LEFT JOIN xwalk_advent_to_lot w ON w.advent_dealer_id = m.advent_store
        -- At most one row per opportunity_id, so this cannot fan credit out.
        LEFT JOIN opp_sold_same_day ust ON ust.opportunity_id = m.opportunity_id
    """)

    # -- 4b. events ---------------------------------------------------------
    # INNER join to the opportunity: an event whose opportunity_id cannot be resolved is
    # dropped entirely, and dealercode comes only from that opportunity. Because the dim
    # pool is unfiltered (Closed Duplicate included), every surviving event has a store.
    #
    # Two grains live in this one table and are split into two fact branches below:
    #   appointments_created   -> created_date + created_by  (usually the BDR)
    #   scheduled / shows      -> sched_date   + sales_user
    con.execute("""
        CREATE OR REPLACE TABLE stg_events AS
        SELECT
            NULLIF(TRIM(CAST(e.event_id AS VARCHAR)), '')                 AS event_id,
            o.opportunity_id                                              AS opportunity_id,
            CAST(e.created_date AS DATE)                                  AS created_date,
            CAST(e.sched_date   AS DATE)                                  AS sched_date,
            COALESCE(NULLIF(TRIM(CAST(e.CreatedById AS VARCHAR)), ''),
                     (SELECT unassigned_user_id FROM cfg))                AS created_by,
            COALESCE(NULLIF(TRIM(CAST(e.salesperson_id AS VARCHAR)), ''),
                     (SELECT unassigned_user_id FROM cfg))                AS sales_user,
            LOWER(TRIM(CAST(e.cancelled AS VARCHAR)))
                IN ('true','1','t','yes','y')                             AS is_cancelled,
            LOWER(TRIM(CAST(e.show AS VARCHAR)))
                IN ('true','1','t','yes','y')                             AS is_show,
            -- Output 3. Parsed with the same truthy set as cancelled/show so a 'TRUE' or '1'
            -- cannot silently zero the measure; the QA block lists the raw values seen.
            LOWER(TRIM(CAST(e.BDC_Sales_Scheduled__c AS VARCHAR)))
                IN ('true','1','t','yes','y')                             AS is_bdc_scheduled,
            -- The BDC show test is PRESENCE of a confirmer, not the show boolean above. Two
            -- different definitions of 'show' across Output 1 and Output 3, by design.
            (NULLIF(TRIM(CAST(e.Show_Confirmed_by__c AS VARCHAR)), '')
                IS NOT NULL)                                              AS is_bdc_show,
            NULLIF(TRIM(CAST(e.BDR_Name__c AS VARCHAR)), '')              AS bdr_name,
            NULLIF(UPPER(TRIM(regexp_replace(
                CAST(e.BDR_Name__c AS VARCHAR), '\\s+', ' ', 'g'))), '')  AS bdr_name_key,
            o.dealercode, o.vin, o."condition", o.make, o.model, o."year",
            o.lead_type, o.lead_category, o.lead_source
        FROM raw_events e
        JOIN opp_dim_by_id o
          ON o.opportunity_id = NULLIF(TRIM(CAST(e.opportunity_id AS VARCHAR)), '')
    """)

    # -- 4c. tasks ----------------------------------------------------------
    # LEFT join, unlike events: an unmatched task survives with lead_type = 'Account'.
    # That works because a task carries its own dealercode, which is also the fallback
    # store when no opportunity matches.
    #
    # activity_type arrives as 'call__outbound'. Only the segment before '__' is used as
    # the mapping key, so direction is deliberately ignored; the QA block reports which
    # suffixes actually appear. An activity_type that maps to nothing keeps its row with
    # all three measures at 0 rather than being dropped, so upstream drift stays visible.
    con.execute("""
        CREATE OR REPLACE TABLE stg_tasks AS
        SELECT
            NULLIF(TRIM(CAST(t.task_id AS VARCHAR)), '')                   AS task_id,
            CAST(t.task_date AS DATE)                                      AS activity_date,
            COALESCE(NULLIF(TRIM(CAST(t.user_id AS VARCHAR)), ''),
                     (SELECT unassigned_user_id FROM cfg))                 AS user_id,
            -- The matched opportunity's store wins, so task activity rolls up to the same
            -- rooftop as the deal's opportunity/appointment/sale rows. Falls back to the
            -- task's own store when there is no match; since that is never blank, the
            -- column is never null. o.dealercode is already normalized upstream.
            COALESCE(o.dealercode,
                     UPPER(NULLIF(TRIM(CAST(t.dealercode AS VARCHAR)), ''))) AS dealercode,
            (o.dealercode IS NOT NULL)                                     AS store_from_opp,
            -- Carried for QA only. Tasks never touch sales credit, so nothing keys on this;
            -- both stores above are already Salesforce lots. Here so a task's own store can
            -- be reconciled against the opportunity's if the two ever disagree.
            UPPER(NULLIF(TRIM(CAST(t.advent_dealer_id AS VARCHAR)), ''))   AS advent_dealer_id,
            -- Channel is the LEADING WORD, not a split on a fixed delimiter. The source has
            -- shipped both 'call__outbound' and 'Call - Outbound'; keying on the first
            -- run of letters handles either, plus 'call_outbound', 'Call/Outbound' or a
            -- bare 'Call', without another edit here when the format shifts again.
            regexp_extract(LOWER(TRIM(CAST(t.activity_type AS VARCHAR))), '^[a-z]+') AS channel,
            -- everything after that word and its separator characters
            NULLIF(regexp_replace(LOWER(TRIM(CAST(t.activity_type AS VARCHAR))),
                                  '^[a-z]+[^a-z]*', ''), '')                       AS direction,
            NULLIF(TRIM(CAST(t.activity_type AS VARCHAR)), '')             AS activity_type_raw,
            m.measure                                                      AS measure,
            (o.opportunity_id IS NOT NULL)                                 AS opp_matched,
            o.vin, o."condition", o.make, o.model, o."year",
            -- keyed on whether the OPPORTUNITY matched, not on whether lead_type is null:
            -- a matched opportunity with a blank lead_type must not be relabelled 'Account'
            CASE WHEN o.opportunity_id IS NULL
                 THEN (SELECT account_lead_label FROM cfg)
                 ELSE o.lead_type END                                      AS lead_type,
            o.lead_category                                                AS lead_category,
            o.lead_source                                                  AS lead_source
        FROM raw_tasks t
        LEFT JOIN opp_dim_by_id o
               ON o.opportunity_id = NULLIF(TRIM(CAST(t.opportunity_id AS VARCHAR)), '')
        LEFT JOIN cfg_activity_map m
               ON m.channel = regexp_extract(LOWER(TRIM(CAST(t.activity_type AS VARCHAR))), '^[a-z]+')
    """)

    # -- 4d. leads ----------------------------------------------------------
    # Every DIMENSION comes from the lead file's own columns -- the lead's captured detail is
    # the truth at lead time. The USER is the converted opportunity's salesperson, which is
    # what lets leads_created sit in Output 1 alongside everything else.
    #
    # INNER join, like events: the source is pre-filtered upstream to leads that carry an
    # opportunity_id, so an unresolvable one means that filter failed. Dropped rows are
    # reported as a WARNING in QA rather than vanishing.
    #
    # VIN is passed through untouched, consistent with the opportunity branch.
    con.execute("""
        CREATE OR REPLACE TABLE stg_leads AS
        SELECT
            NULLIF(TRIM(CAST(l.lead_id AS VARCHAR)), '')                 AS lead_id,
            CAST(l.lead_date AS DATE)                                    AS activity_date,
            UPPER(NULLIF(TRIM(CAST(l.dealercode AS VARCHAR)), ''))       AS dealercode,
            COALESCE(o.user_id, (SELECT unassigned_user_id FROM cfg))    AS user_id,
            -- The same lead credits the salesperson in Output 1 and the BDR in Output 3. Two
            -- outputs, one row each, so neither rollup can double -- the same arrangement
            -- appointments_created already has.
            COALESCE(o.bdr_id, (SELECT unassigned_user_id FROM cfg))     AS bdr_id,
            CAST(l."VIN" AS VARCHAR)                                     AS vin,
            NULLIF(TRIM(CAST(l."condition" AS VARCHAR)), '')             AS "condition",
            NULLIF(TRIM(CAST(l.make  AS VARCHAR)), '')                   AS make,
            NULLIF(TRIM(CAST(l.model AS VARCHAR)), '')                   AS model,
            NULLIF(TRIM(CAST(l."Year" AS VARCHAR)), '')                  AS "year",
            (SELECT ecommerce_lead_label FROM cfg)                       AS lead_type,
            NULLIF(TRIM(CAST(l.lead_category AS VARCHAR)), '')           AS lead_category,
            NULLIF(TRIM(CAST(l.lead_source   AS VARCHAR)), '')           AS lead_source
        FROM raw_leads l
        JOIN opp_dim_by_id o
          ON o.opportunity_id = NULLIF(TRIM(CAST(l.opportunity_id AS VARCHAR)), '')
    """)

    # -- 4e. sold to show ---------------------------------------------------
    # A customer showed for a scheduled appointment AND bought within 14 days of it. Dated
    # and credited to the winning appointment (its sched_date and salesperson), NOT to the
    # sale -- so shows and sold_to_show land on the same row for the same event.
    #
    # The appointment-to-sale link runs through the opportunity, reusing sales_matched, so
    # the sale side is Finalized-gated and already collapsed to one opportunity per credit
    # row. CEILING: a sale that never matched a Finalized opportunity (the Advent bucket)
    # cannot qualify, so this understates true conversion by construction.
    #
    # Three de-duplications, each guarding a different way the flag could exceed 1:
    #   DISTINCT     -- a split deal is two credit rows sharing one opportunity
    #   BETWEEN      -- several distinct sale dates can fall inside one appointment's window
    #   ROW_NUMBER   -- an opportunity can have several qualifying shown appointments;
    #                   keep the latest, tie-broken on event_id for determinism
    con.execute("""
        CREATE OR REPLACE TABLE stg_sold_to_show AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT e.opportunity_id, e.sched_date, e.sales_user, e.dealercode,
                   e.vin, e."condition", e.make, e.model, e."year",
                   e.lead_type, e.lead_category, e.lead_source,
                   ROW_NUMBER() OVER (PARTITION BY e.opportunity_id
                                      ORDER BY e.sched_date DESC, e.event_id) AS rn
            FROM stg_events e
            JOIN (SELECT DISTINCT opportunity_id, activity_date
                  FROM sales_matched WHERE opportunity_id IS NOT NULL) s
              ON s.opportunity_id = e.opportunity_id
             AND s.activity_date BETWEEN e.sched_date AND e.sched_date + INTERVAL 14 DAY
            WHERE e.is_show AND NOT e.is_cancelled AND e.sched_date IS NOT NULL
        )
        WHERE rn = 1
    """)

    # -- 4f. BDC sales credit (Output 3) ------------------------------------
    # Joined to the opportunity by opportunity_id DIRECTLY -- no dealercode + dealnumber
    # matching, no Finalized stage gate, no ranked pick. Nothing here can fan out, because
    # opp_dim_by_id is one row per opportunity_id and xwalk_bdc_user is one row per user_id.
    # A Closed Duplicate opportunity therefore CAN carry BDC credit, unlike the rep side;
    # stage_name is carried so QA can count that rather than let it pass unnoticed.
    #
    # Staged with LEFT joins and flags first so the QA block can attribute each dropped row
    # to its own cause -- four different causes needing four different fixes.
    con.execute("""
        CREATE OR REPLACE TABLE stg_bdc_raw AS
        SELECT
            NULLIF(TRIM(CAST(c.credit_id AS VARCHAR)), '')          AS credit_id,
            NULLIF(TRIM(CAST(c.bdc_id AS VARCHAR)), '')             AS bdc_id,
            NULLIF(TRIM(CAST(c.opportunity_id AS VARCHAR)), '')     AS source_opportunity_id,
            TRY_CAST(c.bdc_sales_credit AS DOUBLE)                  AS bdc_credit,
            (o.opportunity_id IS NOT NULL)                          AS opp_resolved,
            x.bdc_user_id                                           AS user_id,
            CAST(o.sale_date AS DATE)                               AS activity_date,
            o.stage_name,
            o.dealercode, o.vin, o."condition", o.make, o.model, o."year",
            o.lead_type, o.lead_category, o.lead_source
        FROM raw_bdc_credit c
        LEFT JOIN opp_dim_by_id o
               ON o.opportunity_id = NULLIF(TRIM(CAST(c.opportunity_id AS VARCHAR)), '')
        LEFT JOIN xwalk_bdc_user x
               ON x.user_id = NULLIF(TRIM(CAST(c.bdc_id AS VARCHAR)), '')
    """)
    con.execute("""
        CREATE OR REPLACE TABLE stg_bdc_credit AS
        SELECT * FROM stg_bdc_raw
        WHERE opp_resolved
          AND user_id IS NOT NULL
          AND activity_date IS NOT NULL
          AND bdc_credit IS NOT NULL
    """)

    # -- 4g. BDC appointments scheduled / shows -----------------------------
    # The only join in the script keyed on a NAME. Events carry no BDR id, so BDR_Name__c is
    # the sole bridge to the user table. dim_bdc being one row per person is what makes it
    # well defined; name_key normalizes case and internal whitespace on both sides. An
    # unmatched name is dropped (the BDC list is a filter) and reported with samples, since a
    # format drift like "Last, First" would otherwise just look like a quiet decline.
    con.execute("""
        CREATE OR REPLACE TABLE stg_bdc_appts AS
        SELECT
            e.event_id, e.sched_date, e.is_bdc_show, e.bdr_name, e.bdr_name_key,
            d.user_id                                               AS user_id,
            e.dealercode, e.vin, e."condition", e.make, e.model, e."year",
            e.lead_type, e.lead_category, e.lead_source
        FROM stg_events e
        JOIN dim_bdc d ON d.name_key = e.bdr_name_key
        WHERE e.is_bdc_scheduled AND NOT e.is_cancelled AND e.sched_date IS NOT NULL
    """)

    # -- 5. stack the facts -------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW fact_all AS
        -- opportunities created
        -- The opp_sold_same_day join is what makes the flip reach THIS branch too. Without
        -- it the sale, the appointments and the calls for one opportunity would say
        -- 'Showroom UST' while its own created row still said 'Showroom'.
        SELECT o.dealercode, o.activity_date, o.user_id, o.vin,
               o."condition", o.make, o.model, o."year",
               ust_label(o.lead_type, u.opportunity_id IS NULL) AS lead_type,
               o.lead_category, o.lead_source,
               0.0 AS total_sales,
               1   AS opportunities_created,
               0   AS appointments_created,
               0   AS appointments_scheduled,
               0   AS shows,
               0   AS calls,
               0   AS texts,
               0   AS emails,
               0   AS leads_created,
               0   AS sold_to_show,
               -- The two gross measures are APPENDED, not slotted next to total_sales. This is
               -- a positional UNION ALL across seven branches: inserting mid-list means
               -- silently re-aligning every branch's run of literals, and appending leaves
               -- existing Tableau workbooks' column positions alone.
               0.0 AS front_gross,
               0.0 AS back_gross
        FROM stg_opp o
        LEFT JOIN opp_sold_same_day u ON u.opportunity_id = o.opportunity_id
        WHERE o.activity_date IS NOT NULL

        UNION ALL

        -- sales credit
        SELECT dealercode, activity_date, user_id, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               sales_credit AS total_sales,
               0, 0, 0, 0, 0, 0, 0, 0, 0,
               -- The ONLY branch that populates these. Already the employee's share, so this
               -- is a straight carry-through -- one row per credit row with one user, the same
               -- shape total_sales has, so the Output 2 rollup cannot double.
               front_gross, back_gross
        FROM sales_with_dims

        UNION ALL

        -- appointments created: dated by appt_created_date, credited to CreatedById.
        -- No cancelled filter -- a cancelled appointment was still created.
        SELECT dealercode, created_date, created_by, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0,
               1   AS appointments_created,
               0, 0, 0, 0, 0, 0, 0,
               0.0, 0.0
        FROM stg_events
        WHERE created_date IS NOT NULL

        UNION ALL

        -- appointments scheduled + shows: dated by appt_scheduled_date, credited to
        -- salesperson_id. Cancelled excluded from both. shows rides this branch rather
        -- than getting its own, because it shares this date and user exactly.
        SELECT dealercode, sched_date, sales_user, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0, 0,
               1                                   AS appointments_scheduled,
               CASE WHEN is_show THEN 1 ELSE 0 END AS shows,
               0, 0, 0, 0, 0,
               0.0, 0.0
        FROM stg_events
        WHERE sched_date IS NOT NULL AND NOT is_cancelled

        UNION ALL

        -- tasks: one branch, three CASE measures, since every task has exactly one
        -- activity_type. An unmapped type yields 0/0/0 and is reported in QA.
        SELECT dealercode, activity_date, user_id, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0, 0, 0, 0,
               CASE WHEN measure = 'calls'  THEN 1 ELSE 0 END AS calls,
               CASE WHEN measure = 'texts'  THEN 1 ELSE 0 END AS texts,
               CASE WHEN measure = 'emails' THEN 1 ELSE 0 END AS emails,
               0, 0,
               0.0, 0.0
        FROM stg_tasks
        WHERE activity_date IS NOT NULL

        UNION ALL

        -- leads created: dims are the lead's own, user is the converted opportunity's
        -- salesperson. A normal user-bearing branch, so Output 2 rolls it up like any other.
        SELECT dealercode, activity_date, user_id, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0, 0, 0, 0, 0, 0, 0,
               1 AS leads_created,
               0,
               0.0, 0.0
        FROM stg_leads
        WHERE activity_date IS NOT NULL

        UNION ALL

        -- sold to show: one flag per opportunity, on the latest qualifying shown appointment
        SELECT dealercode, sched_date, sales_user, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0, 0, 0, 0, 0, 0, 0, 0,
               1 AS sold_to_show,
               0.0, 0.0
        FROM stg_sold_to_show
    """)

    # -- 6. Output 1: salesperson grain -------------------------------------
    # Materialized as a TABLE so Output 2 is derived from it rather than
    # re-queried from source. That is what makes the two tie by construction.
    con.execute("""
        CREATE OR REPLACE TABLE output_1 AS
        SELECT
            f.dealercode                              AS dealercode,
            f.activity_date                           AS "date",
            COALESCE(u.user_name,
                     (SELECT unassigned_user_name FROM cfg)) AS user_name,
            -- Defaulted, not left NULL: Tableau drops NULL members, so these rows were
            -- disappearing out of every view that grouped on role. A COALESCE here rather than
            -- a synthetic dim_user row because it catches all three ways the NULL arises --
            -- the '-1' user, a real id missing from raw_users, and a blank profile_name.
            -- A real value always wins; these only ever replace a NULL.
            COALESCE(u.user_role,
                     (SELECT unassigned_user_role FROM cfg))  AS user_role,
            COALESCE(u.department,
                     (SELECT unassigned_department FROM cfg)) AS department,
            f.user_id                                 AS user_id,
            COALESCE(u.is_active,
                     (SELECT unassigned_is_active FROM cfg))  AS is_active,
            COALESCE(n_cond.canonical_label,  f."condition")   AS "condition",
            COALESCE(n_make.canonical_label,  f.make)          AS make,
            COALESCE(n_model.canonical_label, f.model)         AS model,
            f."year"                                  AS "year",
            f.vin                                     AS "VIN",
            COALESCE(n_type.canonical_label,  f.lead_type)     AS lead_type,
            COALESCE(n_cat.canonical_label,   f.lead_category) AS lead_category,
            COALESCE(n_src.canonical_label,   f.lead_source)   AS lead_source,
            SUM(f.total_sales)                        AS total_sales,
            SUM(f.opportunities_created)              AS opportunities_created,
            SUM(f.appointments_created)               AS appointments_created,
            SUM(f.appointments_scheduled)             AS appointments_scheduled,
            SUM(f.shows)                              AS shows,
            SUM(f.calls)                              AS calls,
            SUM(f.texts)                              AS texts,
            SUM(f.emails)                             AS emails,
            SUM(f.leads_created)                      AS leads_created,
            SUM(f.sold_to_show)                       AS sold_to_show,
            -- Measures, not dimensions, so they need no xwalk_label join and do not touch the
            -- GROUP BY. They ride existing sale rows, so this adds no rows and cannot move any
            -- other measure.
            SUM(f.front_gross)                        AS front_gross,
            SUM(f.back_gross)                         AS back_gross
        FROM fact_all f
        LEFT JOIN dim_user u ON u.user_id = f.user_id
        -- Canonicalize the text dimensions HERE, before the GROUP BY, so the existing grouping
        -- collapses spellings differing only by case or internal whitespace. Doing it in
        -- finalize() instead would relabel the values but leave two rows carrying identical
        -- dimension tuples -- row counts inflate and the duplicate-tuple check in reconcile()
        -- starts failing. LEFT JOIN + COALESCE so an unlisted value passes through unchanged
        -- rather than becoming NULL. One row per raw_label by construction, so no fanout.
        LEFT JOIN xwalk_label n_cond  ON n_cond.raw_label  = f."condition"
        LEFT JOIN xwalk_label n_make  ON n_make.raw_label  = f.make
        LEFT JOIN xwalk_label n_model ON n_model.raw_label = f.model
        LEFT JOIN xwalk_label n_type  ON n_type.raw_label  = f.lead_type
        LEFT JOIN xwalk_label n_cat   ON n_cat.raw_label   = f.lead_category
        LEFT JOIN xwalk_label n_src   ON n_src.raw_label   = f.lead_source
        GROUP BY ALL
        ORDER BY dealercode, "date", user_name
    """)

    # -- 7. Output 2: store grain -------------------------------------------
    # Drops the five user attributes, is_active included: a user attribute has no
    # meaning at store grain and keeping it would smuggle salesperson detail back in.
    #
    # Derived entirely from output_1, which is what makes the two tie by construction.
    #
    # Rolling up cannot double count because output_1 never emits the same fact twice under
    # different people -- every source row yields one fact row with one user.
    # (opportunities_created keys only on salesperson_id; bdr_id is deliberately unused.
    # Events credit two people but as two different measures. Split sales credit is
    # 0.5 + 0.5. sold_to_show is one flag per opportunity.) Output 2 therefore sits above
    # the salesperson/BDR distinction without needing to know about it.
    #
    # The invariant to preserve when adding a measure: it must never be emitted once per
    # (opportunity, person) pair -- that is the shape that would make this rollup double.
    con.execute("""
        CREATE OR REPLACE TABLE output_2 AS
        SELECT
            dealercode, "date",
            "condition", make, model, "year", "VIN",
            lead_type, lead_category, lead_source,
            SUM(total_sales)             AS total_sales,
            SUM(opportunities_created)   AS opportunities_created,
            SUM(appointments_created)    AS appointments_created,
            SUM(appointments_scheduled)  AS appointments_scheduled,
            SUM(shows)                   AS shows,
            SUM(calls)                   AS calls,
            SUM(texts)                   AS texts,
            SUM(emails)                  AS emails,
            SUM(leads_created)           AS leads_created,
            SUM(sold_to_show)            AS sold_to_show,
            -- The store-level gross that prompted all of this. A pure re-sum of Output 1, so it
            -- ties by construction like every other measure here.
            SUM(front_gross)             AS front_gross,
            SUM(back_gross)              AS back_gross
        FROM output_1
        GROUP BY ALL
        ORDER BY dealercode, "date"
    """)

    # -- 8. Output 3: BDC-person grain --------------------------------------
    # Same stacked-fact architecture, its own population and its own nine measures.
    #
    # Every branch INNER joins the BDC population, which is what makes the BDC user list a
    # FILTER: activity by anyone outside it is absent from Output 3 entirely. Each branch still
    # emits one row per source record with exactly one user, so THE INVARIANT holds here too --
    # which is precisely why BDC credit lives in its own output instead of in Output 1, where
    # crediting both the rep and the BDR would make the Output 2 store rollup double count.
    #
    # NOTE ON 'shows': this is Show_Confirmed_by__c being populated, NOT Output 1's `show`
    # boolean. Same column name, deliberately different definition. Do not diff the two.
    con.execute("""
        CREATE OR REPLACE VIEW fact_bdc AS
        -- BDC sales credit, dated by the OPPORTUNITY's sale_date (Salesforce's rs_date)
        SELECT dealercode, activity_date, user_id, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               bdc_credit AS total_sales,
               0 AS opportunities_created,
               0 AS appointments_created,
               0 AS appointments_scheduled,
               0 AS shows,
               0 AS calls,
               0 AS texts,
               0 AS emails,
               0 AS leads_created
        FROM stg_bdc_credit

        UNION ALL

        -- opportunities created, credited to bdr_id rather than salesperson_id.
        -- The opp_sold_same_day join is here for the same reason as on the rep side. Without it
        -- this is the ONE Output 3 branch that would still say plain 'Showroom' while the same
        -- opportunity's BDC sale, appointments and calls all said 'Showroom UST' -- every other
        -- branch inherits the flip through opp_dim_by_id, this one reads stg_opp directly.
        SELECT o.dealercode, o.activity_date, x.bdc_user_id, o.vin,
               o."condition", o.make, o.model, o."year",
               ust_label(o.lead_type, sd.opportunity_id IS NULL) AS lead_type,
               o.lead_category, o.lead_source,
               0.0, 1, 0, 0, 0, 0, 0, 0, 0
        FROM stg_opp o
        JOIN xwalk_bdc_user x ON x.user_id = o.bdr_id
        LEFT JOIN opp_sold_same_day sd ON sd.opportunity_id = o.opportunity_id
        WHERE o.activity_date IS NOT NULL

        UNION ALL

        -- appointments created, credited to CreatedById. No cancelled filter: a cancelled
        -- appointment was still created.
        SELECT e.dealercode, e.created_date, x.bdc_user_id, e.vin,
               e."condition", e.make, e.model, e."year",
               e.lead_type, e.lead_category, e.lead_source,
               0.0, 0, 1, 0, 0, 0, 0, 0, 0
        FROM stg_events e
        JOIN xwalk_bdc_user x ON x.user_id = e.created_by
        WHERE e.created_date IS NOT NULL

        UNION ALL

        -- appointments scheduled + shows share one branch: same date, same person, so only a
        -- CASE separates them. Cancelled already excluded in stg_bdc_appts.
        SELECT dealercode, sched_date, user_id, vin,
               "condition", make, model, "year",
               lead_type, lead_category, lead_source,
               0.0, 0, 0,
               1                                        AS appointments_scheduled,
               CASE WHEN is_bdc_show THEN 1 ELSE 0 END  AS shows,
               0, 0, 0, 0
        FROM stg_bdc_appts

        UNION ALL

        -- activities, by the same activity_type mapping the rep side uses
        SELECT t.dealercode, t.activity_date, x.bdc_user_id, t.vin,
               t."condition", t.make, t.model, t."year",
               t.lead_type, t.lead_category, t.lead_source,
               0.0, 0, 0, 0, 0,
               CASE WHEN t.measure = 'calls'  THEN 1 ELSE 0 END,
               CASE WHEN t.measure = 'texts'  THEN 1 ELSE 0 END,
               CASE WHEN t.measure = 'emails' THEN 1 ELSE 0 END,
               0
        FROM stg_tasks t
        JOIN xwalk_bdc_user x ON x.user_id = t.user_id
        WHERE t.activity_date IS NOT NULL

        UNION ALL

        -- leads created, credited to the converted opportunity's BDR rather than its
        -- salesperson. Same stg_leads rows Output 1 uses -- dims and date are still the LEAD's
        -- own -- so one lead credits two different people in two different outputs.
        SELECT l.dealercode, l.activity_date, x.bdc_user_id, l.vin,
               l."condition", l.make, l.model, l."year",
               l.lead_type, l.lead_category, l.lead_source,
               0.0, 0, 0, 0, 0, 0, 0, 0,
               1 AS leads_created
        FROM stg_leads l
        JOIN xwalk_bdc_user x ON x.user_id = l.bdr_id
        WHERE l.activity_date IS NOT NULL
    """)

    # Mirrors output_1's shape exactly, joined to dim_bdc instead of dim_user. Every user_id
    # reaching here is already a canonical BDC id, so the join cannot miss and 'unassigned'
    # never appears -- unlike Output 1, this population is defined by the user table itself.
    con.execute("""
        CREATE OR REPLACE TABLE output_3 AS
        SELECT
            f.dealercode                              AS dealercode,
            f.activity_date                           AS "date",
            COALESCE(u.user_name,
                     (SELECT unassigned_user_name FROM cfg)) AS user_name,
            -- user_role is deliberately NOT defaulted here, unlike Output 1. The BDC population
            -- is DEFINED by user_role ILIKE '%BDC Sales%', so it can never be NULL on this
            -- output -- and 'Salesperson' would be an actively wrong label in a BDC dataset.
            -- Do not "fix" this asymmetry.
            u.user_role                               AS user_role,
            COALESCE(u.department,
                     (SELECT unassigned_department FROM cfg)) AS department,
            f.user_id                                 AS user_id,
            COALESCE(u.is_active,
                     (SELECT unassigned_is_active FROM cfg))  AS is_active,
            COALESCE(n_cond.canonical_label,  f."condition")   AS "condition",
            COALESCE(n_make.canonical_label,  f.make)          AS make,
            COALESCE(n_model.canonical_label, f.model)         AS model,
            f."year"                                  AS "year",
            f.vin                                     AS "VIN",
            COALESCE(n_type.canonical_label,  f.lead_type)     AS lead_type,
            COALESCE(n_cat.canonical_label,   f.lead_category) AS lead_category,
            COALESCE(n_src.canonical_label,   f.lead_source)   AS lead_source,
            SUM(f.total_sales)                        AS total_sales,
            SUM(f.opportunities_created)              AS opportunities_created,
            SUM(f.appointments_created)               AS appointments_created,
            SUM(f.appointments_scheduled)             AS appointments_scheduled,
            SUM(f.shows)                              AS shows,
            SUM(f.calls)                              AS calls,
            SUM(f.texts)                              AS texts,
            SUM(f.emails)                             AS emails,
            SUM(f.leads_created)                      AS leads_created
        FROM fact_bdc f
        LEFT JOIN dim_bdc u ON u.user_id = f.user_id
        -- Same canonicalization as output_1, and it has to be repeated rather than inherited:
        -- output_2 gets it free by being derived from output_1, but Output 3 is built from
        -- fact_bdc independently. Leaving it out would make the same lead type render two ways
        -- across the rep and BDC dashboards.
        LEFT JOIN xwalk_label n_cond  ON n_cond.raw_label  = f."condition"
        LEFT JOIN xwalk_label n_make  ON n_make.raw_label  = f.make
        LEFT JOIN xwalk_label n_model ON n_model.raw_label = f.model
        LEFT JOIN xwalk_label n_type  ON n_type.raw_label  = f.lead_type
        LEFT JOIN xwalk_label n_cat   ON n_cat.raw_label   = f.lead_category
        LEFT JOIN xwalk_label n_src   ON n_src.raw_label   = f.lead_source
        GROUP BY ALL
        ORDER BY dealercode, "date", user_name
    """)


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
def one(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


def report_qa(con, opp_rows, event_rows, task_rows, lead_rows):
    """
    One scan per fact table, no correlated subqueries. Everything here reads from
    materialized tables, so the cost is negligible next to the transforms.
    """
    # -- opportunities: a single pass for every count ------------------------
    o = con.execute("""
        SELECT COUNT(*)                                             AS kept,
               COUNT(DISTINCT opportunity_id)                       AS ids,
               COUNT(*) FILTER (activity_date IS NULL)               AS no_date,
               COUNT(*) FILTER (user_id = (SELECT unassigned_user_id FROM cfg)) AS no_person,
               COUNT(*) FILTER (deal_key IS NULL)                    AS no_deal
        FROM stg_opp
    """).fetchone()
    qa_header("OPPORTUNITIES")
    qa("rows kept", f"{o[0]:,}")
    qa(f"dropped by excluded_stages {CONFIG['excluded_stages']}", f"{opp_rows - o[0]:,}")
    qa("dropped for unparseable created date", f"{o[2]:,}")
    qa("no salesperson (kept as unassigned)", f"{o[3]:,}  ({_pct(o[3], o[0])})")
    qa("blank deal_number (cannot match a sale)", f"{o[4]:,}")
    if o[1] != o[0]:
        qa("WARNING: opportunity_id is not unique", f"{o[0] - o[1]:,} duplicate rows")

    # -- the match pool and the ambiguity that drives the tiebreak ----------
    # Grouped on advent_dealer_id, NOT dealercode: this has to measure ambiguity on the same
    # key the match actually uses, or it understates the work the tiebreak is doing at a
    # split rooftop -- where two lots' opportunities land in one candidate pool.
    m = con.execute("""
        SELECT (SELECT COUNT(*) FROM opp_match_pool),
               (SELECT COUNT(*) FROM (SELECT 1 FROM opp_match_pool
                                      GROUP BY advent_dealer_id, deal_key HAVING COUNT(*) > 1)),
               (SELECT COUNT(*) FROM (SELECT 1 FROM stg_opp WHERE deal_key IS NOT NULL
                                      GROUP BY advent_dealer_id, deal_key HAVING COUNT(*) > 1)),
               (SELECT COUNT(*) FROM stg_opp WHERE advent_id_missing),
               (SELECT COUNT(DISTINCT dealercode) FROM stg_opp WHERE advent_id_missing),
               (SELECT COUNT(*) FROM stg_opp WHERE advent_dealer_id IS NULL)
    """).fetchone()
    qa(f"eligible to match a sale {_match_stages()}", f"{m[0]:,}")
    qa("deals with >1 opportunity, all stages", f"{m[2]:,}")
    qa("deals with >1 ELIGIBLE opportunity (drives tiebreak)", f"{m[1]:,}")
    # Coverage, not an error: the dealercode fallback in stg_opp_all makes these rows join
    # correctly anyway. Worth watching because at a SPLIT rooftop the fallback is a guess.
    qa("advent_dealer_id blank (fell back to dealercode)",
       f"{m[3]:,}  ({_pct(m[3], o[0])}) across {m[4]:,} stores")
    if m[5]:
        qa("WARNING: no store code at all (matches no sale)", f"{m[5]:,}")

    # -- the store crosswalk ------------------------------------------------
    # advent_dealer_id is coarser than dealercode by design: one Advent store can cover
    # several Salesforce lots. Listing them makes a new split visible without a code change.
    fan = con.execute("""
        SELECT advent_dealer_id, STRING_AGG(dealercode, ', ' ORDER BY dealercode) AS lots
        FROM store_pairs
        GROUP BY advent_dealer_id HAVING COUNT(*) > 1
        ORDER BY advent_dealer_id
    """).fetchall()
    qa("stores where Advent covers >1 lot", f"{len(fan):,}")
    for advent_store, lots in fan:
        qa(f"  {advent_store} covers", lots)

    x = con.execute("""
        SELECT (SELECT COUNT(*) FROM (SELECT 1 FROM store_pairs
                                      GROUP BY dealercode HAVING COUNT(*) > 1)),
               (SELECT COUNT(*) FROM stg_users u
                JOIN store_pairs p ON p.dealercode = u.user_dealercode
                JOIN xwalk_advent_to_lot w ON w.advent_dealer_id = p.advent_dealer_id
                WHERE u.advent_id IS NOT NULL
                  AND u.user_dealercode <> w.primary_dealercode)
    """).fetchone()
    if x[0]:
        # A lot belonging to two Advent stores breaks the premise this design rests on.
        qa("WARNING: dealercode mapping to >1 advent_dealer_id", f"{x[0]:,}")
    if x[1]:
        # Informational, not a warning: these are the users the Advent-space translation in
        # xwalk_advent exists for. Before that translation their credit fell to 'unassigned'
        # silently. Kept visible because the number moving is the signal that store
        # assignment upstream has shifted.
        qa("users at a non-primary lot (translated to Advent space)", f"{x[1]:,}")

    # -- sales credit: one pass on the raw stage, one on the matched result --
    s = con.execute("""
        SELECT COUNT(*)                                     AS all_rows,
               COUNT(*) FILTER (activity_date IS NULL)        AS no_date,
               COUNT(*) FILTER (sales_credit IS NULL)         AS no_credit,
               -- Upstream coverage of the two new columns, read BEFORE the COALESCE to 0 makes
               -- a missing gross indistinguishable from a real zero.
               COUNT(*) FILTER (front_gross_missing)          AS no_front,
               COUNT(*) FILTER (back_gross_missing)           AS no_back,
               -- Gross on the rows this stage is about to DISCARD. Those drops used to be free;
               -- now they are money leaving the dataset, so they get their own line.
               COALESCE(SUM(front_gross + back_gross) FILTER (
                   activity_date IS NULL OR sales_credit IS NULL), 0)     AS dropped_gross
        FROM stg_sales_raw
    """).fetchone()
    r = con.execute("""
        SELECT COUNT(*)                                                  AS kept,
               COALESCE(SUM(sales_credit), 0)                            AS credit,
               COUNT(*) FILTER (user_id = (SELECT unassigned_user_id FROM cfg)) AS no_user,
               COUNT(*) FILTER (sp_id_other_store)                        AS other_store,
               COUNT(*) FILTER (opportunity_id IS NULL)                   AS advent,
               COUNT(*) FILTER (opportunity_id IS NULL AND deal_has_any_opp) AS stage_gated,
               COUNT(*) FILTER (n_candidates > 1)                         AS ambiguous,
               COUNT(*) FILTER (n_candidates > 1 AND opp_user_id = user_id) AS by_person,
               COALESCE(SUM(front_gross), 0)                              AS front,
               COALESCE(SUM(back_gross), 0)                               AS back,
               -- Gross sitting on credit that matched no opportunity, i.e. dollars with no real
               -- lead dims and only a fallback lot. The gross analogue of the Advent bucket.
               COALESCE(SUM(front_gross + back_gross) FILTER (
                   opportunity_id IS NULL), 0)                            AS orphan_gross
        FROM sales_matched
    """).fetchone()
    (kept, credit, no_user, other_store, advent,
     stage_gated, ambiguous, by_person, front, back, orphan_gross) = r
    qa_header("SALES CREDIT")
    qa("rows kept", f"{kept:,}")
    qa("dropped for unparseable 'rs date'", f"{s[1]:,}")
    qa("dropped for null/unparseable sales_credit", f"{s[2]:,}")
    qa("sum of sales_credit", f"{credit:,.1f}")
    # The two new measures. These are ALREADY the employee's share -- upstream multiplies by
    # sales_credit before writing -- so this is the sum of shares, which reassembles the
    # deal-level total exactly when every deal's credit sums to 1.0. The line below prices the
    # deals where it does not.
    qa("sum of front_gross", f"{front:,.2f}")
    qa("sum of back_gross", f"{back:,.2f}")
    if s[3] or s[4]:
        qa("WARNING: front_gross null upstream (read as 0)", f"{s[3]:,}")
        qa("WARNING: back_gross null upstream (read as 0)", f"{s[4]:,}")
    if s[5]:
        qa("WARNING: gross discarded with unusable credit rows", f"{s[5]:,.2f}")
    # Deliberately NOT a percentage: gross can be negative (a chargeback), so a share of a
    # possibly-negative denominator would be actively misleading. The dollar figure is the point.
    qa("gross on credit that matched no opportunity", f"{orphan_gross:,.2f}")
    # Decisive check on the FixedDecimal conversion: if 0.5 splits were being rounded
    # or truncated anywhere upstream, they would not appear here.
    qa("distinct sales_credit values", ", ".join(
        f"{v}x{n:,}" for v, n in con.execute("""
            SELECT sales_credit, COUNT(*) FROM stg_sales
            GROUP BY 1 ORDER BY 2 DESC LIMIT 6
        """).fetchall()))
    qa("sp_id unresolved on (advent store, advent_id)", f"{no_user:,}  ({_pct(no_user, kept)})")
    qa("  of those, sp_id exists but at another store", f"{other_store:,}  ({_pct(other_store, no_user)})")
    qa("no eligible opportunity (tagged Advent)", f"{advent:,}  ({_pct(advent, kept)})")
    qa("  of those, blocked by the stage gate", f"{stage_gated:,}  ({_pct(stage_gated, advent)})")
    qa("  of those, deal has no opportunity at all", f"{advent - stage_gated:,}")
    qa("matched to a multi-opportunity deal", f"{ambiguous:,}  ({_pct(ambiguous, kept)})")
    qa("  resolved to the credited salesperson's own opp", f"{by_person:,}  ({_pct(by_person, ambiguous)})")

    # -- split integrity: the one failure mode the PRE-SPLIT gross introduces -----------------
    # The gross columns arrive already multiplied by sales_credit, so the sum over a deal's
    # credit rows only reassembles the deal-level figure when those rows sum to 1.0. If one half
    # of a split never arrives, that deal's gross is understated by half -- and nothing else in
    # this script can see it, because each surviving row looks perfectly well formed.
    #
    # total_sales has the identical exposure today, but a deal counting 0.5 is obvious in a sales
    # report where half a deal's gross is not. Hence a line of its own.
    #
    # A plain GROUP BY, not a correlated subquery (gotcha 4). Tolerance rather than != 1.0:
    # these are floats.
    sp = con.execute("""
        SELECT COUNT(*), COALESCE(SUM(gross), 0), COALESCE(SUM(credit), 0)
        FROM (
            SELECT deal_key,
                   SUM(sales_credit)              AS credit,
                   SUM(front_gross + back_gross)  AS gross
            FROM stg_sales
            WHERE deal_key IS NOT NULL
            GROUP BY deal_key
            HAVING ABS(SUM(sales_credit) - 1.0) > 1e-6
        )
    """).fetchone()
    if sp[0]:
        qa("WARNING: deals whose credit does not sum to 1.0", f"{sp[0]:,}")
        qa("  gross on them (understated if credit < 1)", f"{sp[1]:,.2f}")
        qa("  their total credit", f"{sp[2]:,.1f}")

    # -- which tier supplied the reported lot --------------------------------
    # Sales arrive on the Advent store and are reported on the Salesforce lot. Tier 1 is the
    # matched opportunity's lot; tier 2 the primary lot when nothing matched; tier 3 an Advent
    # store on no opportunity at all, which must be empty.
    t = con.execute("""
        SELECT COUNT(*) FILTER (m.opp_dealercode IS NOT NULL)              AS from_opp,
               COUNT(*) FILTER (m.opp_dealercode IS NULL
                                AND w.primary_dealercode IS NOT NULL)      AS from_primary,
               COUNT(*) FILTER (m.opp_dealercode IS NULL
                                AND w.primary_dealercode IS NULL)          AS unresolved,
               COUNT(*) FILTER (m.opp_dealercode IS NULL AND w.n_lots > 1) AS split_rows,
               COALESCE(SUM(m.sales_credit) FILTER (m.opp_dealercode IS NULL
                                                    AND w.n_lots > 1), 0)  AS split_credit
        FROM sales_matched m
        LEFT JOIN xwalk_advent_to_lot w ON w.advent_dealer_id = m.advent_store
    """).fetchone()
    qa("reported lot from the matched opportunity", f"{t[0]:,}  ({_pct(t[0], kept)})")
    qa("  fell back to the store's primary lot", f"{t[1]:,}  ({_pct(t[1], kept)})")
    if t[2]:
        # NOT an error. Some Advent stores have no Salesforce opportunities at all -- five
        # divested rooftops whose history stays on the books, plus a Toyota store that never
        # had Salesforce sales records. Their credit is real and is counted; it just publishes
        # under the raw Advent code, which is still a genuine store code.
        #
        # What matters is telling a KNOWN gap apart from a NEW one, hence the config list.
        # An unlisted store here means either another divestiture to record or a real mapping
        # break, and the self-pair in xwalk_advent_to_lot narrows which: it makes every
        # non-null dealercode a resolvable key, so anything unresolved is genuinely absent
        # from the opportunity file rather than merely mis-keyed.
        #
        # Joined, not correlated: one pass over stg_opp_all, per the QA rule.
        known = {str(s).strip().upper() for s in CONFIG["advent_only_stores"]}
        rows = con.execute("""
            WITH unresolved AS (
                SELECT m.advent_store, COUNT(*) AS n_rows, SUM(m.sales_credit) AS credit
                FROM sales_matched m
                LEFT JOIN xwalk_advent_to_lot w ON w.advent_dealer_id = m.advent_store
                WHERE m.opp_dealercode IS NULL AND w.primary_dealercode IS NULL
                GROUP BY 1
            ), sf_lots AS (
                SELECT dealercode, STRING_AGG(DISTINCT advent_dealer_id, ', ') AS advent_ids
                FROM stg_opp_all WHERE dealercode IS NOT NULL GROUP BY 1
            )
            SELECT u.advent_store, u.n_rows, u.credit,
                   (l.dealercode IS NOT NULL), l.advent_ids
            FROM unresolved u
            LEFT JOIN sf_lots l ON l.dealercode = u.advent_store
            ORDER BY u.credit DESC LIMIT 15
        """).fetchall()
        unexpected = [r for r in rows if (r[0] or "") not in known]
        qa("credit at a store with no Salesforce opportunity",
           f"{t[2]:,} rows across {len(rows)} stores  ({_pct(t[2], kept)})")
        for code, n, credit, in_sf, its_id in rows:
            if (code or "") in known:
                note = "expected: no Salesforce opportunities"
            elif in_sf:
                # Should be unreachable -- see the self-pair note above.
                note = f"** UNEXPECTED ** in Salesforce, its advent_dealer_id = {its_id}"
            else:
                note = "** UNEXPECTED ** absent from the opportunity file"
            qa(f"  {code}: {n:,} rows, {credit:,.1f} credit", note)
        if unexpected:
            qa("WARNING: unlisted store with no Salesforce opportunity",
               f"{len(unexpected)} store(s) -- add to CONFIG['advent_only_stores'] if divested")
    # The residual cost of the design: unmatched credit at a SPLIT rooftop has no lot to
    # inherit and no way to infer one, so it all lands on the primary lot. Everything else in
    # the fallback bucket is a store with one lot, where the fallback is the identity mapping.
    qa("credit at a split store with no lot to inherit",
       f"{t[3]:,} rows, {t[4]:,.1f} credit  ({_pct(t[3], kept)})")

    # Which tier each matched sale resolved on. This prices what opening the pool to Registered
    # Sale actually bought, and it is also the empirical answer to "do opportunities with
    # lead_type 'Advent' even exist" -- tier 2 is unreachable if they do not.
    tiers = con.execute("""
        SELECT COALESCE(CAST(match_tier AS VARCHAR), 'no match')  AS tier,
               ANY_VALUE(opp_stage_name)                          AS stage,
               COUNT(*)                                           AS rows,
               SUM(sales_credit)                                  AS credit
        FROM sales_matched
        GROUP BY match_tier
        ORDER BY match_tier NULLS LAST
    """).fetchall()
    qa("sale match tier (rank order)", f"{_match_stages()}")
    for tier, stage, rows, credit in tiers:
        label = "unmatched -> Advent bucket" if tier == "no match" else f"tier {tier}: {stage}"
        qa(f"  {label}", f"{rows:,} rows, {credit or 0:,.1f} credit  ({_pct(rows, kept)})")

    # 'Advent' in the OUTPUT now means two different things and they are indistinguishable
    # downstream, so they have to be separated here:
    #   - the orphan label, applied because the credit matched no opportunity at all
    #   - a matched opportunity whose OWN lead_type is 'Advent' (DMS-originated, not a real lead)
    # The second only became visible when the ranking started distinguishing them. Reading the
    # Advent bucket as "unmatched credit" without this split overstates it.
    adv = con.execute("""
        SELECT COUNT(*) FILTER (opportunity_id IS NULL)                     AS orphan,
               COALESCE(SUM(sales_credit) FILTER (opportunity_id IS NULL), 0) AS orphan_credit,
               COUNT(*) FILTER (opportunity_id IS NOT NULL AND is_advent)   AS real_advent,
               COALESCE(SUM(sales_credit) FILTER (opportunity_id IS NOT NULL
                                                  AND is_advent), 0)        AS real_credit
        FROM (
            SELECT opportunity_id, sales_credit,
                   LOWER(REGEXP_REPLACE(COALESCE(lead_type, ''), '\\s+', ' ', 'g'))
                       = (SELECT advent_key FROM cfg) AS is_advent
            FROM sales_matched
        )
    """).fetchone()
    qa(f"sale rows labelled {CONFIG['advent_lead_label']!r}", "two different causes:")
    qa("  matched NO opportunity (the orphan label)",
       f"{adv[0]:,} rows, {adv[1]:,.1f} credit")
    qa("  matched an opportunity whose own lead_type is Advent",
       f"{adv[2]:,} rows, {adv[3]:,.1f} credit")

    # Showroom -> Showroom UST, counted at the OPPORTUNITY grain, which is where the rule is
    # decided. Every Showroom opportunity is in scope now, not only the ones carrying credit --
    # never selling is itself a reason to flip. Read post-flip off opp_dim_by_id: the flip only
    # ever maps Showroom -> Showroom UST, so the two labels together are the whole population,
    # which avoids re-deriving that table's dedupe here just to count it.
    #
    # All three buckets print, because a flip count alone cannot tell "the rule works" from
    # "almost nothing closes on the visit" -- and the two flip reasons are different businesses:
    # a come-back is a won deal, a never-sold is not.
    sh = con.execute("""
        SELECT COUNT(*)                                                       AS showroom,
               COUNT(*) FILTER (d.lead_type = (SELECT showroom_lead_label FROM cfg))
                                                                              AS kept_plain,
               COUNT(*) FILTER (d.lead_type = (SELECT showroom_ust_lead_label FROM cfg)
                                AND m.opportunity_id IS NOT NULL)             AS sold_later,
               COUNT(*) FILTER (d.lead_type = (SELECT showroom_ust_lead_label FROM cfg)
                                AND m.opportunity_id IS NULL)                 AS never_sold
        FROM opp_dim_by_id d
        LEFT JOIN (SELECT DISTINCT opportunity_id FROM sales_matched
                   WHERE opportunity_id IS NOT NULL) m
               ON m.opportunity_id = d.opportunity_id
        WHERE d.lead_type IN ((SELECT showroom_lead_label     FROM cfg),
                              (SELECT showroom_ust_lead_label FROM cfg))
    """).fetchone()
    _ust = CONFIG["showroom_ust_lead_label"]
    qa("Showroom opportunities", f"{sh[0]:,}")
    qa("  -> kept plain (closed on the visit)", f"{sh[1]:,}  ({_pct(sh[1], sh[0])})")
    qa(f"  -> {_ust!r}: sold later than created", f"{sh[2]:,}  ({_pct(sh[2], sh[0])})")
    qa(f"  -> {_ust!r}: never sold at all", f"{sh[3]:,}  ({_pct(sh[3], sh[0])})")
    # Credit with no created date to compare cannot be shown to have closed on the visit, so it
    # flips and is counted above under 'sold later'. Surfaced separately because it is a data
    # gap rather than a business outcome, and should be ~0.
    nocd = one(con, """
        SELECT COUNT(DISTINCT opportunity_id) FROM sales_matched
        WHERE opportunity_id IS NOT NULL AND opp_created_date IS NULL
    """)
    if nocd:
        qa("  WARNING: sold but no created date to compare (flipped)", f"{nocd:,}")
    # The flip is decided per opportunity but lands on every fact derived from it. Counting the
    # output rows is what proves that reach -- a number equal to the sale rows alone would mean
    # opp_dim_by_id lost the flip and events and tasks are still saying plain 'Showroom'.
    reach = con.execute("""
        SELECT (SELECT COUNT(*) FROM output_1
                WHERE lead_type = (SELECT showroom_ust_lead_label FROM cfg)),
               (SELECT COUNT(*) FROM output_1
                WHERE lead_type = (SELECT showroom_ust_lead_label FROM cfg)
                  AND total_sales > 0)
    """).fetchone()
    qa("  Output 1 rows carrying the UST label", f"{reach[0]:,}")
    qa("    of which are sale rows", f"{reach[1]:,}  (rest are events, tasks and leads)")

    # -- events -------------------------------------------------------------
    e = con.execute("""
        SELECT COUNT(*)                                                   AS kept,
               COUNT(DISTINCT event_id)                                   AS ids,
               COUNT(*) FILTER (created_date IS NULL)                     AS no_created,
               COUNT(*) FILTER (sched_date IS NULL)                       AS no_sched,
               COUNT(*) FILTER (is_cancelled)                             AS cancelled,
               COUNT(*) FILTER (created_by = (SELECT unassigned_user_id FROM cfg)) AS no_creator,
               COUNT(*) FILTER (sched_date IS NOT NULL AND NOT is_cancelled
                                AND sales_user = (SELECT unassigned_user_id FROM cfg)) AS no_sp,
               COUNT(*) FILTER (NOT is_cancelled AND sched_date IS NOT NULL AND is_show) AS shows
        FROM stg_events
    """).fetchone()
    qa_header("EVENTS")
    qa("rows kept (opportunity_id resolved)", f"{e[0]:,}")
    qa("dropped: opportunity_id unresolvable", f"{event_rows - e[0]:,}  ({_pct(event_rows - e[0], event_rows)})")
    qa("no appt_created_date (excluded from created)", f"{e[2]:,}")
    qa("no appt_scheduled_date (created only)", f"{e[3]:,}")
    qa("cancelled (created only)", f"{e[4]:,}  ({_pct(e[4], e[0])})")
    qa("shows", f"{e[7]:,}")
    qa("no CreatedById (unassigned)", f"{e[5]:,}")
    qa("no salesperson_id on a scheduled appt (unassigned)", f"{e[6]:,}")
    if e[1] != e[0]:
        qa("WARNING: event_id is not unique", f"{e[0] - e[1]:,} duplicate rows")

    # -- tasks --------------------------------------------------------------
    t = con.execute("""
        SELECT COUNT(*)                                                   AS rows,
               COUNT(DISTINCT task_id)                                    AS ids,
               COUNT(*) FILTER (activity_date IS NULL)                    AS no_date,
               COUNT(*) FILTER (NOT opp_matched)                          AS no_opp,
               COUNT(*) FILTER (dealercode IS NULL)                       AS no_store,
               COUNT(*) FILTER (user_id = (SELECT unassigned_user_id FROM cfg)) AS no_user,
               COUNT(*) FILTER (measure IS NULL AND activity_date IS NOT NULL) AS unmapped,
               COUNT(*) FILTER (store_from_opp)                           AS store_from_opp
        FROM stg_tasks
    """).fetchone()
    # A real user_id with no user record is a DIFFERENT problem from a blank one, but both
    # surface in the output as user_name = 'unassigned', so they need separating here.
    no_user_record = one(con, """
        SELECT COUNT(*) FROM stg_tasks t
        LEFT JOIN dim_user u ON u.user_id = t.user_id
        WHERE u.user_id IS NULL
          AND t.user_id <> (SELECT unassigned_user_id FROM cfg)
    """)
    qa_header("TASKS")
    qa("rows kept", f"{t[0] - t[2]:,}")
    qa("dropped for unparseable CompletedDateTime", f"{t[2]:,}")
    if t[2]:
        bad = con.execute("""
            SELECT DISTINCT CAST(CompletedDateTime AS VARCHAR) FROM raw_tasks
            WHERE task_date IS NULL LIMIT 3
        """).fetchall()
        qa("  example unparseable values", ", ".join(repr(b[0]) for b in bad))
    qa("no matching opportunity (lead_type = Account)", f"{t[3]:,}  ({_pct(t[3], t[0])})")
    qa("dealercode taken from the opportunity", f"{t[7]:,}  ({_pct(t[7], t[0])})")
    qa("dealercode fell back to the task's own", f"{t[0] - t[7]:,}")
    qa("blank user_id (defaulted to unassigned)", f"{t[5]:,}")
    qa("user_id not found in user table (shows as unassigned)",
       f"{no_user_record:,}  ({_pct(no_user_record, t[0])})")

    # Unconditional: the raw value next to what it mapped to. If the measures come back
    # empty this line says why immediately, and it does not depend on the date parsing
    # having succeeded -- which the tallies below do.
    qa("activity_type -> measure (top 8 raw values)", "")
    for raw, chan, meas, n in con.execute("""
        SELECT activity_type_raw, channel, COALESCE(measure, '** UNMAPPED **'), COUNT(*)
        FROM stg_tasks GROUP BY 1, 2, 3 ORDER BY 4 DESC LIMIT 8
    """).fetchall():
        qa(f"  {raw!r} -> channel {chan!r}", f"{meas}  ({n:,})")

    for measure, n in con.execute("""
        SELECT measure, COUNT(*) FROM stg_tasks
        WHERE activity_date IS NOT NULL AND measure IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall():
        qa(f"  counted {measure}", f"{n:,}")
    # Direction is deliberately not part of the mapping key. If anything other than
    # 'outbound' shows up here, inbound and outbound are being summed together and the
    # measures probably need splitting.
    qa("direction suffixes seen", ", ".join(
        f"{dirn or '(none)'}x{n:,}" for dirn, n in con.execute("""
            SELECT direction, COUNT(*) FROM stg_tasks GROUP BY 1 ORDER BY 2 DESC LIMIT 5
        """).fetchall()))
    # Counts unmapped rows regardless of whether their date parsed, so a total failure of
    # BOTH the mapping and the date conversion still reports rather than going silent.
    unmapped_any = one(con, "SELECT COUNT(*) FROM stg_tasks WHERE measure IS NULL")
    if unmapped_any:
        # Kept, not dropped: these rows contribute a dim tuple with all measures at 0,
        # which is harmless to any SUM but nudges the Output 1 row count up.
        qa("WARNING: unmapped activity_type (counted in nothing)",
           f"{unmapped_any:,} of {t[0]:,}  ({_pct(unmapped_any, t[0])})")
    if t[4]:
        qa("WARNING: task with a blank dealercode", f"{t[4]:,}")
    if t[1] != t[0]:
        qa("WARNING: task_id is not unique", f"{t[0] - t[1]:,} duplicate rows")

    # -- leads --------------------------------------------------------------
    ld = con.execute("""
        SELECT COUNT(*)                                   AS rows,
               COUNT(DISTINCT lead_id)                    AS ids,
               COUNT(*) FILTER (activity_date IS NULL)    AS no_date,
               COUNT(*) FILTER (dealercode IS NULL)       AS no_store,
               COUNT(*) FILTER (user_id = (SELECT unassigned_user_id FROM cfg)) AS no_user
        FROM stg_leads
    """).fetchone()
    qa_header("LEADS")
    qa("rows kept", f"{ld[0] - ld[2]:,}")
    qa("dropped for unparseable CreatedDate", f"{ld[2]:,}")
    # The source is pre-filtered to leads carrying an opportunity_id, so a non-zero drop
    # here means that upstream filter failed -- not a data characteristic.
    if lead_rows != ld[0]:
        qa("WARNING: opportunity_id unresolvable (upstream pre-filter failed)",
           f"{lead_rows - ld[0]:,} of {lead_rows:,}  ({_pct(lead_rows - ld[0], lead_rows)})")
    if ld[4]:
        qa("WARNING: opportunity had no salesperson (unassigned)", f"{ld[4]:,}")
    if ld[3]:
        qa("WARNING: lead with a blank dealercode", f"{ld[3]:,}")
    if ld[1] != ld[0]:
        qa("WARNING: lead_id is not unique", f"{ld[0] - ld[1]:,} duplicate rows")

    # -- sold to show -------------------------------------------------------
    sts = con.execute("""
        SELECT (SELECT COUNT(*) FROM stg_sold_to_show),
               -- qualifying (appointment, sale-date) pairs BEFORE the latest-only collapse,
               -- so the gap shows how much the de-duplication actually removed
               (SELECT COUNT(*) FROM stg_events e
                JOIN (SELECT DISTINCT opportunity_id, activity_date FROM sales_matched
                      WHERE opportunity_id IS NOT NULL) s
                  ON s.opportunity_id = e.opportunity_id
                 AND s.activity_date BETWEEN e.sched_date AND e.sched_date + INTERVAL 14 DAY
                WHERE e.is_show AND NOT e.is_cancelled AND e.sched_date IS NOT NULL),
               (SELECT COUNT(*) FROM stg_events
                WHERE is_show AND NOT is_cancelled AND sched_date IS NOT NULL)
    """).fetchone()
    qa_header("SOLD TO SHOW")
    qa("opportunities flagged", f"{sts[0]:,}")
    qa("qualifying pairs before latest-only collapse", f"{sts[1]:,}")
    qa("shows (the denominator)", f"{sts[2]:,}")
    qa("flag rate as a share of shows", f"{_pct(sts[0], sts[2])}")
    qa("  NOTE: only sales matched to a Finalized opp can qualify", "")

    # -- users --------------------------------------------------------------
    u = con.execute("""
        SELECT (SELECT COUNT(*) FROM dim_user),
               (SELECT COUNT(*) FROM (SELECT 1 FROM stg_users
                                      GROUP BY user_id HAVING COUNT(*) > 1)),
               (SELECT COUNT(*) FROM (SELECT 1 FROM stg_users
                                      WHERE advent_id IS NOT NULL AND user_dealercode IS NOT NULL
                                      GROUP BY user_dealercode, advent_id
                                      HAVING COUNT(DISTINCT user_id) > 1)),
               (SELECT COUNT(*) FROM stg_users
                WHERE advent_id IS NOT NULL AND user_dealercode IS NULL)
    """).fetchone()
    qa_header("USERS")
    qa("distinct user_id", f"{u[0]:,}")
    qa("user_id on >1 row (collapsed)", f"{u[1]:,}")
    qa("(dealercode, advent_id) mapping to >1 user_id (collapsed)", f"{u[2]:,}")
    if u[3]:
        qa("WARNING: advent_id with no dealercode (unreachable)", f"{u[3]:,}")

    # How much of Output 1 is leaning on the defaults. Worth watching rather than ignoring: the
    # defaults stop Tableau dropping these rows, but they also HIDE a degrading user lookup, so a
    # jump here is the only signal that resolution got worse upstream.
    ud = con.execute("""
        SELECT COUNT(*)                                                          AS rows_1,
               COUNT(*) FILTER (o.user_id = (SELECT unassigned_user_id FROM cfg)) AS synthetic,
               COUNT(*) FILTER (d.user_id IS NULL
                                AND o.user_id <> (SELECT unassigned_user_id FROM cfg)) AS gap,
               COUNT(*) FILTER (d.user_id IS NOT NULL AND d.user_role IS NULL)    AS blank_role
        FROM output_1 o
        LEFT JOIN dim_user d ON d.user_id = o.user_id
    """).fetchone()
    qa("Output 1 rows taking a defaulted user attribute",
       f"{ud[1] + ud[2] + ud[3]:,}  ({_pct(ud[1] + ud[2] + ud[3], ud[0])})")
    qa(f"  synthetic {CONFIG['unassigned_user_id']!r} user (no person on the record)",
       f"{ud[1]:,}")
    qa("  real user_id absent from raw_users (lookup gap)", f"{ud[2]:,}")
    qa("  user present but profile_name blank", f"{ud[3]:,}")

    # -- BDC (Output 3) -----------------------------------------------------
    b = con.execute("""
        SELECT (SELECT COUNT(*) FROM bdc_users),
               (SELECT COUNT(*) FROM dim_bdc),
               -- people carrying more than one user record: the crosswalk's whole reason to
               -- exist. If this is 0 the crosswalk was a no-op and cost nothing.
               (SELECT COUNT(*) FROM (SELECT 1 FROM bdc_users
                                      GROUP BY name_key HAVING COUNT(*) > 1)),
               (SELECT COUNT(*) FROM xwalk_bdc_user WHERE user_id <> bdc_user_id)
    """).fetchone()
    qa_header("BDC  (Output 3)")
    qa(f"user rows matching profile_name ~ '{CONFIG['bdc_profile_match']}'", f"{b[0]:,}")
    qa("distinct BDC people (the reporting grain)", f"{b[1]:,}")
    qa("people with >1 user record (consolidated)", f"{b[2]:,}")
    qa("retired user_ids remapped to a winning id", f"{b[3]:,}")
    if not b[0]:
        # An empty population is almost always a renamed profile, not a real absence.
        qa("WARNING: no BDC users matched -- Output 3 will be empty",
           f"check CONFIG['bdc_profile_match']")
    # Which profile values actually matched, so a rename shows up as a shrinking list.
    qa("profile_name values matched", ", ".join(
        f"{v!r}x{n:,}" for v, n in con.execute("""
            SELECT user_role, COUNT(*) FROM bdc_users
            GROUP BY 1 ORDER BY 2 DESC LIMIT 5
        """).fetchall()) or "(none)")

    # Activity rows recovered by the crosswalk -- what would have been dropped outright.
    rec = con.execute("""
        SELECT (SELECT COUNT(*) FROM stg_tasks t JOIN xwalk_bdc_user x ON x.user_id = t.user_id
                WHERE x.user_id <> x.bdc_user_id AND t.activity_date IS NOT NULL),
               (SELECT COUNT(*) FROM stg_opp o JOIN xwalk_bdc_user x ON x.user_id = o.bdr_id
                WHERE x.user_id <> x.bdc_user_id AND o.activity_date IS NOT NULL),
               (SELECT COUNT(*) FROM stg_bdc_credit c JOIN xwalk_bdc_user x
                     ON x.bdc_user_id = c.user_id
                WHERE x.user_id <> x.bdc_user_id)
    """).fetchone()
    if any(rec):
        qa("fact rows sitting on a retired id (kept by the crosswalk)",
           f"{rec[0]:,} tasks, {rec[1]:,} opps, {rec[2]:,} credit rows")

    bc = con.execute("""
        SELECT COUNT(*)                                                AS all_rows,
               COUNT(*) FILTER (NOT opp_resolved)                       AS no_opp,
               COUNT(*) FILTER (opp_resolved AND user_id IS NULL)       AS not_bdc,
               COUNT(*) FILTER (opp_resolved AND user_id IS NOT NULL
                                AND activity_date IS NULL)              AS no_date,
               COUNT(*) FILTER (opp_resolved AND user_id IS NOT NULL
                                AND activity_date IS NOT NULL
                                AND bdc_credit IS NULL)                 AS no_credit
        FROM stg_bdc_raw
    """).fetchone()
    kept_bdc, credit_bdc, closed_dup = con.execute("""
        SELECT COUNT(*), COALESCE(SUM(bdc_credit), 0),
               COUNT(*) FILTER (UPPER(COALESCE(stage_name, ''))
                                IN (SELECT stage FROM cfg_excluded_stages))
        FROM stg_bdc_credit
    """).fetchone()
    qa("bdc_credit rows read", f"{bc[0]:,}")
    qa("  dropped: opportunity_id unresolvable", f"{bc[1]:,}  ({_pct(bc[1], bc[0])})")
    qa("  dropped: bdc_id is not a BDC user", f"{bc[2]:,}  ({_pct(bc[2], bc[0])})")
    qa("  dropped: opportunity has no rs_date", f"{bc[3]:,}  ({_pct(bc[3], bc[0])})")
    qa("  dropped: null/unparseable bdc_sales_credit", f"{bc[4]:,}")
    qa("bdc_credit rows kept", f"{kept_bdc:,}")
    qa("sum of bdc_sales_credit", f"{credit_bdc:,.1f}")
    # Same decisive split-credit check the rep side gets.
    qa("distinct bdc_sales_credit values", ", ".join(
        f"{v}x{n:,}" for v, n in con.execute("""
            SELECT bdc_credit, COUNT(*) FROM stg_bdc_credit
            GROUP BY 1 ORDER BY 2 DESC LIMIT 6
        """).fetchall()) or "(none)")
    # Allowed here, unlike the rep side, because the join is by opportunity_id with no stage
    # gate. Counted so the asymmetry between the two outputs stays visible.
    qa(f"  of those, on an excluded-stage opportunity {CONFIG['excluded_stages']}",
       f"{closed_dup:,}")

    # The name join -- the one bridge in the script with no id fallback.
    nm = con.execute("""
        SELECT COUNT(*)                                             AS bdc_flagged,
               COUNT(*) FILTER (bdr_name_key IS NULL)                AS no_name,
               COUNT(*) FILTER (bdr_name_key IS NOT NULL
                                AND bdr_name_key NOT IN (SELECT name_key FROM dim_bdc))
                                                                     AS unmatched
        FROM stg_events
        WHERE is_bdc_scheduled AND NOT is_cancelled AND sched_date IS NOT NULL
    """).fetchone()
    qa("BDC-scheduled appts (not cancelled, dated)", f"{nm[0]:,}")
    qa("  dropped: BDR_Name__c blank", f"{nm[1]:,}")
    qa("  dropped: BDR_Name__c matches no BDC user",
       f"{nm[2]:,}  ({_pct(nm[2], nm[0])})")
    if nm[2]:
        # Samples are the point: a systematic 'Last, First' drift looks identical to a real
        # decline in the count alone.
        qa("  example unmatched names", ", ".join(repr(v[0]) for v in con.execute("""
            SELECT DISTINCT bdr_name FROM stg_events
            WHERE is_bdc_scheduled AND NOT is_cancelled AND sched_date IS NOT NULL
              AND bdr_name_key IS NOT NULL
              AND bdr_name_key NOT IN (SELECT name_key FROM dim_bdc)
            LIMIT 5
        """).fetchall()))
    # Raw flag values, so a 'TRUE'/'1'/'Yes' change surfaces instead of zeroing the measure.
    qa("BDC_Sales_Scheduled__c raw values seen", ", ".join(
        f"{v!r}x{n:,}" for v, n in con.execute("""
            SELECT CAST(BDC_Sales_Scheduled__c AS VARCHAR), COUNT(*)
            FROM raw_events GROUP BY 1 ORDER BY 2 DESC LIMIT 5
        """).fetchall()))

    # Free cross-system check, only possible now that both sale dates are in scope: for an
    # opportunity carrying BOTH Advent credit and BDC credit, the two dates should agree.
    # A systematic skew would put the same car in different months across Output 1 and 3.
    dd = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (a.activity_date <> b.activity_date),
               -- SIGNED, BDC minus Advent. The sign is the whole diagnostic: a constant -1
               -- means the Salesforce date is being timezone-shifted (a Date field is midnight
               -- UTC, so converting to Pacific rolls it back a day), whereas a constant +1 is
               -- a real lag in when Advent books the deal. Absolute value hides the difference.
               MEDIAN(DATEDIFF('day', a.activity_date, b.activity_date)),
               MIN(DATEDIFF('day', a.activity_date, b.activity_date)),
               MAX(DATEDIFF('day', a.activity_date, b.activity_date))
        FROM (SELECT DISTINCT opportunity_id, activity_date FROM sales_matched
              WHERE opportunity_id IS NOT NULL) a
        JOIN (SELECT DISTINCT source_opportunity_id AS opportunity_id, activity_date
              FROM stg_bdc_credit) b USING (opportunity_id)
    """).fetchone()
    qa("opportunities with both Advent and BDC credit", f"{dd[0]:,}")
    if dd[0]:
        qa("  sale dates disagree", f"{dd[1]:,}  ({_pct(dd[1], dd[0])})")
        qa("  BDC date minus Advent date, in days",
           f"median {dd[2]:+.0f}, range {dd[3]:+.0f} to {dd[4]:+.0f}")
    # Is the opportunity's rs_date a Salesforce DATE or DATETIME? If every raw value is at
    # midnight it is a Date field, and putting it through to_pacific_date() shifts it back a
    # day for no reason. Same trick that settled activity_type: print the raw values.
    qa("raw opportunity rs_date values (top 3)", ", ".join(
        f"{v!r}x{n:,}" for v, n in con.execute("""
            SELECT CAST(rs_date AS VARCHAR), COUNT(*) FROM raw_opportunity
            WHERE NULLIF(TRIM(CAST(rs_date AS VARCHAR)), '') IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 3
        """).fetchall()) or "(none)")

    # -- label canonicalization ---------------------------------------------
    # This section rewrites PUBLISHED dimension values, so every merge has to be visible. A
    # silent merge is indistinguishable from a data problem when a Tableau member disappears.
    qa_header("LABEL NORMALIZATION")
    qa("columns canonicalized", ", ".join(CONFIG["canonical_columns"]) or "(none -- disabled)")
    lab = con.execute("""
        SELECT COUNT(*)                                              AS spellings,
               COUNT(DISTINCT canonical_label)                       AS canonical,
               COUNT(*) FILTER (raw_label <> canonical_label)        AS remapped,
               COALESCE(SUM(n_rows) FILTER (raw_label <> canonical_label), 0) AS rows_moved
        FROM xwalk_label
    """).fetchone()
    qa("distinct spellings seen", f"{lab[0]:,}")
    qa("distinct values after merge", f"{lab[1]:,}")
    qa("spellings remapped onto another", f"{lab[2]:,}")
    qa("source values carrying a remapped spelling", f"{lab[3]:,}")

    # One line per merged group, so the exact rename is on the record. Capped, and the cap is
    # STATED -- a silent truncation reads as "that was all of them".
    merges = con.execute("""
        SELECT canonical_label,
               STRING_AGG(raw_label || ' (' || CAST(n_rows AS VARCHAR) || ')', ', '
                          ORDER BY n_rows DESC, raw_label) AS others,
               SUM(n_rows) AS n
        FROM xwalk_label
        WHERE n_spellings > 1 AND raw_label <> canonical_label
        GROUP BY canonical_label ORDER BY n DESC, canonical_label
    """).fetchall()
    for canonical, others, _n in merges[:25]:
        qa(f"  merged -> {canonical}", f"was: {others}")
    if len(merges) > 25:
        qa("  (merge list truncated)", f"{len(merges) - 25:,} more groups not shown")
    if not merges:
        qa("  no merges", "every value already had one spelling")

    # Groups that WOULD merge if punctuation and spaces were ignored too. Reported, never acted
    # on -- widening the key could collapse two labels that are deliberately distinct, so that
    # stays a decision rather than a default. This is how 'Ecommerce' beside 'E-commerce' gets
    # found. Add the winner to CONFIG["label_overrides"] and the loser will not merge on its
    # own; that needs the key widened, which is a code change.
    near = con.execute("""
        SELECT STRING_AGG(DISTINCT canonical_label, ' | ' ORDER BY canonical_label) AS forms
        FROM xwalk_label
        WHERE loose_key <> ''
        GROUP BY loose_key
        HAVING COUNT(DISTINCT canonical_label) > 1
        ORDER BY forms
    """).fetchall()
    qa("near-miss groups (differ by punctuation only)", f"{len(near):,}")
    for (forms,) in near[:10]:
        qa("  INFO: not merged", forms)
    if len(near) > 10:
        qa("  (near-miss list truncated)", f"{len(near) - 10:,} more not shown")

    qa_header("OUTPUTS")
    counts = con.execute("""
        SELECT (SELECT COUNT(*) FROM output_1), (SELECT COUNT(*) FROM output_2),
               (SELECT COUNT(*) FROM output_3)
    """).fetchone()
    qa("Output 1 rows (salesperson grain)", f"{counts[0]:,}")
    qa("Output 2 rows (store grain)", f"{counts[1]:,}")
    qa("Output 3 rows (BDC-person grain)", f"{counts[2]:,}")


def _match_stages():
    """The eligible stages, in rank order, for the QA line. Derived so it cannot disagree."""
    seen = []
    for r in CONFIG["sale_match_rank"]:
        if r["stage"] not in seen:
            seen.append(r["stage"])
    return seen


def _pct(part, whole):
    return "n/a" if not whole else f"{100.0 * part / whole:.1f}%"


def reconcile(con):
    """Assertions that must all hold. A failure here means the numbers are wrong."""
    qa_header("RECONCILIATION")
    checks, ok = [], True

    # Relabelling a sale's store cannot change the total, which is what makes this check
    # still meaningful after the lot split: credit must be redistributed, never created.
    src_sales = one(con, "SELECT COALESCE(SUM(sales_credit), 0) FROM stg_sales")
    out1_sales = one(con, "SELECT COALESCE(SUM(total_sales), 0) FROM output_1")
    checks.append(("total_sales: Output 1 == sales_credit source",
                   abs(src_sales - out1_sales) < 1e-6, f"{out1_sales:,.1f} vs {src_sales:,.1f}"))

    # Gross rides the same sale rows, so it ties the same way. Sourced from stg_sales, the
    # POST-filter stage: a credit row legitimately dropped for a null sales_credit or an
    # unparseable date takes its gross with it, and sourcing from stg_sales_raw would then fail
    # this check for a reason that is not a defect. The QA block prices those drops separately.
    #
    # A wider tolerance than the credit check because these are dollars, not multiples of 0.5 --
    # a 2M-row float sum accumulates more error than 1e-6 allows.
    for _m in ("front_gross", "back_gross"):
        src_g = one(con, f"SELECT COALESCE(SUM({_m}), 0) FROM stg_sales")
        out_g = one(con, f"SELECT COALESCE(SUM({_m}), 0) FROM output_1")
        checks.append((f"{_m}: Output 1 == sales credit source",
                       abs(src_g - out_g) < 0.01, f"{out_g:,.2f} vs {src_g:,.2f}"))

    # NOT asserted: a sale can legitimately fail to resolve to a Salesforce lot, because some
    # Advent stores have no Salesforce opportunities at all (divested rooftops, and a Toyota
    # store that never had them). Those sales publish the raw Advent code. total_sales still
    # ties, so this is a labelling gap rather than a correctness failure -- the QA block
    # reports it per store and flags any store not in CONFIG['advent_only_stores'].

    # lot -> Advent store must be a function. The reverse is deliberately one-to-many; this
    # direction being ambiguous would undermine the premise the whole crosswalk rests on.
    multi_store = one(con, """
        SELECT COUNT(*) FROM (SELECT 1 FROM store_pairs
                              GROUP BY dealercode HAVING COUNT(*) > 1)
    """)
    checks.append(("each dealercode maps to exactly one advent_dealer_id",
                   multi_store == 0, f"{multi_store:,} ambiguous"))

    src_opps = one(con, "SELECT COUNT(DISTINCT opportunity_id) FROM stg_opp WHERE activity_date IS NOT NULL")
    out1_opps = one(con, "SELECT COALESCE(SUM(opportunities_created), 0) FROM output_1")
    checks.append(("opportunities_created: Output 1 == distinct opportunity_id",
                   src_opps == out1_opps, f"{out1_opps:,} vs {src_opps:,}"))

    # events: each branch reconciles against its own filter set
    ev = con.execute("""
        SELECT COUNT(*) FILTER (created_date IS NOT NULL),
               COUNT(*) FILTER (sched_date IS NOT NULL AND NOT is_cancelled),
               COUNT(*) FILTER (sched_date IS NOT NULL AND NOT is_cancelled AND is_show)
        FROM stg_events
    """).fetchone()
    o1e = con.execute("""
        SELECT COALESCE(SUM(appointments_created), 0),
               COALESCE(SUM(appointments_scheduled), 0),
               COALESCE(SUM(shows), 0)
        FROM output_1
    """).fetchone()
    checks.append(("appointments_created: Output 1 == events with a created date",
                   ev[0] == o1e[0], f"{o1e[0]:,} vs {ev[0]:,}"))
    checks.append(("appointments_scheduled: Output 1 == scheduled, not cancelled",
                   ev[1] == o1e[1], f"{o1e[1]:,} vs {ev[1]:,}"))
    checks.append(("shows: Output 1 == shows on non-cancelled scheduled appts",
                   ev[2] == o1e[2], f"{o1e[2]:,} vs {ev[2]:,}"))
    checks.append(("shows <= appointments_scheduled",
                   o1e[2] <= o1e[1], f"{o1e[2]:,} <= {o1e[1]:,}"))

    # tasks: each activity measure against its own filtered count
    tk = con.execute("""
        SELECT COUNT(*) FILTER (measure = 'calls'),
               COUNT(*) FILTER (measure = 'texts'),
               COUNT(*) FILTER (measure = 'emails')
        FROM stg_tasks WHERE activity_date IS NOT NULL
    """).fetchone()
    o1t = con.execute("""
        SELECT COALESCE(SUM(calls), 0), COALESCE(SUM(texts), 0), COALESCE(SUM(emails), 0)
        FROM output_1
    """).fetchone()
    for i, name in enumerate(("calls", "texts", "emails")):
        checks.append((f"{name}: Output 1 == tasks of that activity_type",
                       tk[i] == o1t[i], f"{o1t[i]:,} vs {tk[i]:,}"))

    o2 = con.execute("""
        SELECT COALESCE(SUM(total_sales), 0), COALESCE(SUM(opportunities_created), 0),
               COALESCE(SUM(appointments_created), 0), COALESCE(SUM(appointments_scheduled), 0),
               COALESCE(SUM(shows), 0), COALESCE(SUM(calls), 0),
               COALESCE(SUM(texts), 0), COALESCE(SUM(emails), 0),
               COALESCE(SUM(leads_created), 0), COALESCE(SUM(sold_to_show), 0),
               COALESCE(SUM(front_gross), 0), COALESCE(SUM(back_gross), 0)
        FROM output_2
    """).fetchone()
    # leads and sold_to_show now flow through Output 1 like every other measure
    src_leads = one(con, "SELECT COUNT(*) FROM stg_leads WHERE activity_date IS NOT NULL")
    src_sts = one(con, "SELECT COUNT(*) FROM stg_sold_to_show")
    o1l = con.execute("""
        SELECT COALESCE(SUM(leads_created), 0), COALESCE(SUM(sold_to_show), 0),
               COALESCE(SUM(shows), 0)
        FROM output_1
    """).fetchone()
    checks.append(("leads_created: Output 1 == leads with a resolved opp and a date",
                   src_leads == o1l[0], f"{o1l[0]:,} vs {src_leads:,}"))
    checks.append(("sold_to_show: Output 1 == flagged opportunities",
                   src_sts == o1l[1], f"{o1l[1]:,} vs {src_sts:,}"))
    checks.append(("sold_to_show <= shows",
                   o1l[1] <= o1l[2], f"{o1l[1]:,} <= {o1l[2]:,}"))

    o1g = con.execute("""
        SELECT COALESCE(SUM(front_gross), 0), COALESCE(SUM(back_gross), 0)
        FROM output_1
    """).fetchone()
    checks.append(("all 12 measures: Output 1 == Output 2",
                   abs(out1_sales - o2[0]) < 1e-6 and out1_opps == o2[1]
                   and tuple(o1e) == (o2[2], o2[3], o2[4])
                   and tuple(o1t) == (o2[5], o2[6], o2[7])
                   and (o1l[0], o1l[1]) == (o2[8], o2[9])
                   and abs(o1g[0] - o2[10]) < 0.01 and abs(o1g[1] - o2[11]) < 0.01,
                   f"sales {o2[0]:,.1f} / opps {o2[1]:,} / appts {o2[2]:,},{o2[3]:,}"
                   f" / shows {o2[4]:,} / c,t,e {o2[5]:,},{o2[6]:,},{o2[7]:,}"
                   f" / leads {o2[8]:,} / sts {o2[9]:,}"
                   f" / gross {o2[10]:,.2f},{o2[11]:,.2f}"))

    # -- Output 3, asserted against its OWN sources only --------------------
    # Output 3 is a different population, dated by a different field, with total_sales from a
    # different FILE. It is not expected to tie to Output 1 or 2 in any way, and the note below
    # is printed so nobody diffs BDC sales against Advent sales and files a bug.
    b3 = con.execute("""
        SELECT COALESCE(SUM(total_sales), 0), COALESCE(SUM(opportunities_created), 0),
               COALESCE(SUM(appointments_created), 0), COALESCE(SUM(appointments_scheduled), 0),
               COALESCE(SUM(shows), 0), COALESCE(SUM(calls), 0),
               COALESCE(SUM(texts), 0), COALESCE(SUM(emails), 0),
               COALESCE(SUM(leads_created), 0)
        FROM output_3
    """).fetchone()
    s3 = con.execute("""
        SELECT (SELECT COALESCE(SUM(bdc_credit), 0) FROM stg_bdc_credit),
               (SELECT COUNT(*) FROM stg_opp o JOIN xwalk_bdc_user x ON x.user_id = o.bdr_id
                WHERE o.activity_date IS NOT NULL),
               (SELECT COUNT(*) FROM stg_events e JOIN xwalk_bdc_user x
                     ON x.user_id = e.created_by
                WHERE e.created_date IS NOT NULL),
               (SELECT COUNT(*) FROM stg_bdc_appts),
               (SELECT COUNT(*) FROM stg_bdc_appts WHERE is_bdc_show),
               (SELECT COUNT(*) FROM stg_tasks t JOIN xwalk_bdc_user x ON x.user_id = t.user_id
                WHERE t.activity_date IS NOT NULL AND t.measure = 'calls'),
               (SELECT COUNT(*) FROM stg_tasks t JOIN xwalk_bdc_user x ON x.user_id = t.user_id
                WHERE t.activity_date IS NOT NULL AND t.measure = 'texts'),
               (SELECT COUNT(*) FROM stg_tasks t JOIN xwalk_bdc_user x ON x.user_id = t.user_id
                WHERE t.activity_date IS NOT NULL AND t.measure = 'emails'),
               (SELECT COUNT(*) FROM stg_leads l JOIN xwalk_bdc_user x ON x.user_id = l.bdr_id
                WHERE l.activity_date IS NOT NULL)
    """).fetchone()
    checks.append(("Output 3 total_sales == bdc_sales_credit source",
                   abs(b3[0] - s3[0]) < 1e-6, f"{b3[0]:,.1f} vs {s3[0]:,.1f}"))
    for i, name in enumerate(("opportunities_created", "appointments_created",
                              "appointments_scheduled", "shows",
                              "calls", "texts", "emails", "leads_created"), start=1):
        checks.append((f"Output 3 {name} == its own BDC-filtered source",
                       b3[i] == s3[i], f"{b3[i]:,} vs {s3[i]:,}"))
    checks.append(("Output 3 shows <= appointments_scheduled",
                   b3[4] <= b3[3], f"{b3[4]:,} <= {b3[3]:,}"))
    # A canonical id that is not in dim_bdc would mean the crosswalk and the dim disagree.
    orphan_bdc = one(con, """
        SELECT COUNT(*) FROM fact_bdc f
        LEFT JOIN dim_bdc d ON d.user_id = f.user_id
        WHERE d.user_id IS NULL
    """)
    checks.append(("every Output 3 row resolves to a BDC person",
                   orphan_bdc == 0, f"{orphan_bdc:,} orphaned"))

    # The Tableau symptom this fixes is invisible in the data itself -- a NULL member just
    # silently drops out of a view -- so it needs a real assertion rather than a QA line.
    nulls = con.execute("""
        SELECT COUNT(*) FILTER (user_role IS NULL),
               COUNT(*) FILTER (department IS NULL),
               COUNT(*) FILTER (is_active IS NULL)
        FROM output_1
    """).fetchone()
    checks.append(("no NULL user attributes in Output 1",
                   nulls == (0, 0, 0),
                   f"role {nulls[0]:,} / dept {nulls[1]:,} / active {nulls[2]:,}"))

    # Six LEFT JOINs to xwalk_label sit in front of both output GROUP BYs, so a duplicated
    # raw_label would fan every fact row out and inflate every measure. That would surface as a
    # measure mismatch above, but this names the cause directly instead of leaving someone to
    # work back from a wrong total.
    xw_dupes = one(con, """
        SELECT COUNT(*) FROM (
            SELECT 1 FROM xwalk_label GROUP BY raw_label HAVING COUNT(*) > 1)
    """)
    checks.append(("xwalk_label has one row per spelling (no join fanout)",
                   xw_dupes == 0, f"{xw_dupes:,} duplicated"))

    dupes_1 = one(con, """
        SELECT COUNT(*) FROM (
            SELECT 1 FROM output_1
            GROUP BY dealercode, "date", user_name, user_role, department, user_id, is_active,
                     "condition", make, model, "year", "VIN", lead_type, lead_category, lead_source
            HAVING COUNT(*) > 1)
    """)
    dupes_2 = one(con, """
        SELECT COUNT(*) FROM (
            SELECT 1 FROM output_2
            GROUP BY dealercode, "date", "condition", make, model, "year", "VIN",
                     lead_type, lead_category, lead_source
            HAVING COUNT(*) > 1)
    """)
    checks.append(("no duplicate dimension tuples in either output",
                   dupes_1 == 0 and dupes_2 == 0, f"{dupes_1} / {dupes_2}"))

    for label, passed, detail in checks:
        qa(f"[{'PASS' if passed else 'FAIL'}] {label}", detail)
        ok = ok and passed
    qa("NOTE: Output 3 is NOT expected to tie to Outputs 1 or 2",
       "different population, dates and credit file")

    if not ok and CONFIG["strict_checks"]:
        raise AssertionError(
            "Reconciliation failed -- see the QA block above. The outputs do not tie; "
            "do not publish them. Set CONFIG['strict_checks'] = False to write them anyway."
        )
    return ok


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------
def finalize(df):
    """
    Cast to plain object / float64 / int64 and normalize every missing value to None.

    MUTATES df in place and returns it. The only caller passes a freshly materialized
    con.execute(...).df(), which nothing else holds a reference to, so there is nothing to
    protect -- and the .copy() this used to make doubled peak memory on a 2.2M x 23 frame,
    which is the wrong thing to spend right before writing three of them.
    """
    out = df
    for col in out.columns:
        if col in MONEY_MEASURES:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float64")
        elif col in COUNT_MEASURES:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
        elif col == "date":
            dt = pd.to_datetime(out[col], errors="coerce")
            if CONFIG["date_as_datetime"]:
                out[col] = dt.astype("datetime64[ns]")
            else:
                out[col] = dt.dt.strftime("%Y-%m-%d").astype(object).where(dt.notna(), None)
        else:
            s = out[col].astype(object)
            out[col] = s.where(pd.notna(s), None)
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_parquet(df, table):
    """
    Write one output atomically: full file to <name>.tmp, then os.replace().

    os.replace is atomic within a filesystem, so a downstream Alteryx Input Data tool either
    sees the previous complete file or the new complete one -- never a half-written parquet
    from a run that died partway. Without this, a crash mid-write publishes a truncated
    dataset that reads as valid-but-short.
    """
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    final = os.path.join(CONFIG["output_dir"], CONFIG["output_files"][table])
    tmp = final + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)
    return final


def write_logs(status, error=None):
    """
    Persist the QA block. Run Command discards stdout, so this is the ONLY record of a run.

    Called from the `finally` path, so a crash still leaves the diagnostics that explain it --
    losing the QA block exactly when something went wrong is the failure mode to avoid.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    body = "\n".join(QA_LINES)
    footer = f"\n\nSTATUS: {status}" + (f"\nERROR: {error}" if error else "") + "\n"
    written = []
    try:
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        for name in (f"dataset_{stamp}.log", "last_run.log"):
            path = os.path.join(CONFIG["log_dir"], name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"run started {stamp}\n{body}{footer}")
            written.append(path)
        if CONFIG["write_qa_csv"]:
            rows = QA_RECORDS + [{"seq": len(QA_RECORDS) + 1, "section": "STATUS",
                                  "severity": "FAIL" if status != "OK" else "INFO",
                                  "label": status, "value": str(error or "")}]
            path = os.path.join(CONFIG["log_dir"], "qa_log.csv")
            tmp = path + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, path)
            written.append(path)
    except Exception as exc:                                     # noqa: BLE001
        # Never let logging mask the real failure -- say so and carry on to the exit code.
        print(f"[warn] could not write logs to {CONFIG['log_dir']}: {exc}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """
    Returns a process exit code. 0 = published, 1 = did not publish.

    Alteryx's Run Command decides success purely on the exit code, so this is the contract
    that keeps strict_checks meaningful outside the Python tool. The .bat wrapper must end
    with `exit /b %ERRORLEVEL%` or the workflow reports success no matter what happens here.
    """
    status, error, written = "OK", None, []
    try:
        opp, sales, users, events, tasks, leads, bdc = load_inputs()
        con = duckdb.connect()
        try:
            build(con, opp, sales, users, events, tasks, leads, bdc)
            report_qa(con, len(opp), len(events), len(tasks), len(leads))
            reconcile(con)

            # One output at a time: materialize, cast, write, release. Holding all three at
            # once peaked at roughly 4M rows of object-dtype pandas for no reason now that
            # nothing downstream needs them in memory.
            qa_header("WRITING")
            for table in ("output_1", "output_2", "output_3"):
                df = finalize(con.execute(f"SELECT * FROM {table}").df())
                rows = len(df)
                path = write_parquet(df, table)
                del df
                qa(f"wrote {os.path.basename(path)}", f"{rows:,} rows")
                written.append(path)
                if CONFIG["write_to_alteryx"] and IN_ALTERYX:
                    # Vestigial path; re-reads the file rather than keeping the frame alive.
                    Alteryx.write(pd.read_parquet(path), int(table[-1]))
        finally:
            con.close()
    except Exception as exc:                                     # noqa: BLE001
        status, error = type(exc).__name__, str(exc)
        qa_header("FAILED")
        qa(f"WARNING: {status}", str(exc)[:200])
        write_logs(status, error)
        print(f"\n{status}: {exc}", file=sys.stderr)
        return 1

    write_logs(status)
    print(f"\nOK: wrote {len(written)} parquet outputs to {CONFIG['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
