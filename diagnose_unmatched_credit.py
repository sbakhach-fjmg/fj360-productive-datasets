"""
List the sales credit that matched an opportunity but lost its sales rep.

This targets ONE quadrant on purpose -- the fixable one:

    the deal DID match a Finalized opportunity, so it carries real lead and vehicle dims,
    but the Advent sp_id did NOT resolve to a Salesforce user, so the credit sits on
    'unassigned' and a named rep is silently losing their number.

Every row is self-diagnosing: it puts Advent's id and name next to who Salesforce thinks the
salesperson is and what Advent id that user actually holds. Reading across a row tells you the
fix without leaving the CSV:

    names agree, sf_assigned_advent_id blank  -> the rep has no Advent id. Populate it.
    names agree, advent ids differ            -> wrong id on the user record.
    names differ                              -> different people. Not a data bug.
    sf_salesperson_id = -1                    -> the opportunity has no salesperson either.

Usage (from the folder holding the venv):

    .venv\\Scripts\\python.exe diagnose_unmatched_credit.py 2026-07-01 2026-07-31
    .venv\\Scripts\\python.exe diagnose_unmatched_credit.py 2026-07-01 2026-07-31 C:\\temp
    .venv\\Scripts\\python.exe diagnose_unmatched_credit.py           # no bounds, everything

Both dates are INCLUSIVE. Output goes to the current directory unless a third argument gives a
folder.

WHY THIS IMPORTS THE MAIN SCRIPT rather than re-deriving anything: the Advent bridge is
`(advent store, advent_id)` where the user's store is first translated into Advent space, and the
sale match is a ranked pick over Finalized opportunities keyed on advent_dealer_id + dealnumber.
Reimplementing either here would eventually disagree with the real outputs and send you chasing
mismatches that are only artefacts of this file. So it runs the real load_inputs() and build(),
then queries the same staging tables the outputs are built from.

Importing is safe: create_sf_datasets.py guards its entry point with `if __name__ ==
"__main__"`, so nothing is written and no outputs are published by running this.
"""

import importlib.util
import os
import sys

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "create_sf_datasets.py")

RANGE = """
    s.activity_date BETWEEN COALESCE(TRY_CAST(? AS DATE), DATE '1900-01-01')
                        AND COALESCE(TRY_CAST(? AS DATE), DATE '2999-12-31')
"""

# DISTINCT because a split deal is two credit rows and none of these columns distinguishes them --
# without it the same deal would appear twice for no visible reason. Two DIFFERENT reps splitting
# one deal still yield two rows, correctly, because advent_sp_id differs.
DETAIL_SQL = f"""
    SELECT DISTINCT
           s.dealercode          AS dealercode,
           s.deal_key            AS deal_number,
           s.activity_date       AS rs_date,
           s.sp_id               AS advent_sp_id,
           s.sp_name             AS advent_sp_name,
           ar.registered_at      AS advent_id_registered_at,
           m.opportunity_id      AS opportunity_id,
           m.opp_user_id         AS sf_salesperson_id,
           du.user_name          AS sf_salesperson_name,
           ua.user_stores        AS sf_salesperson_store,
           ua.assigned_advent_id AS sf_assigned_advent_id
    FROM stg_sales s
    JOIN sales_matched m USING (sales_row_id)
    LEFT JOIN dim_user du ON du.user_id = m.opp_user_id
    -- One row per user_id. stg_users carries one row per (user, store), so a rep at two stores
    -- would otherwise fan this out; STRING_AGG also means a rep holding two different Advent
    -- ids or working two lots shows both rather than an arbitrary pick.
    -- assigned_advent_id NULL IS the finding: the rep has no Advent id at all.
    -- user_stores is the SALESFORCE lot(s) off the user record, untranslated.
    LEFT JOIN (SELECT user_id,
                      STRING_AGG(DISTINCT advent_id, ', ')       AS assigned_advent_id,
                      STRING_AGG(DISTINCT user_dealercode, ', ') AS user_stores
               FROM stg_users
               GROUP BY user_id) ua ON ua.user_id = m.opp_user_id
    -- Where the credited Advent id IS registered, if anywhere. Read from xwalk_advent, so this
    -- is already in ADVENT space and directly comparable to s.dealercode -- which is exactly
    -- the comparison the bridge makes. Non-null and different from dealercode means the store
    -- is why this fell out, not the id. NULL means the id is in no user record at all.
    LEFT JOIN (SELECT advent_id, STRING_AGG(DISTINCT advent_store, ', ') AS registered_at
               FROM xwalk_advent GROUP BY advent_id) ar ON ar.advent_id = s.sp_id
    WHERE {RANGE}
      -- matched an opportunity, so it has real dims and a real lead_type...
      AND m.opportunity_id IS NOT NULL
      -- ...but the rep did not resolve. Filtering on the join result rather than on the
      -- 'Advent' lead_type label avoids depending on a display value.
      AND m.user_id = (SELECT unassigned_user_id FROM cfg)
    ORDER BY dealercode, advent_sp_name, rs_date
"""

# The four-way split, so the targeted quadrant can be read in proportion to the rest.
SPLIT_SQL = f"""
    SELECT COUNT(*)                                                          AS in_range,
           COUNT(*) FILTER (rep_ok AND opp_ok)                               AS both_ok,
           COUNT(*) FILTER (NOT rep_ok AND opp_ok)                           AS target,
           COUNT(*) FILTER (NOT rep_ok AND NOT opp_ok)                       AS neither,
           COUNT(*) FILTER (rep_ok AND NOT opp_ok)                           AS no_opp_only
    FROM (
        SELECT (m.user_id <> (SELECT unassigned_user_id FROM cfg)) AS rep_ok,
               (m.opportunity_id IS NOT NULL)                      AS opp_ok
        FROM stg_sales s JOIN sales_matched m USING (sales_row_id)
        WHERE {RANGE}
    )
"""


def load_main():
    """Import create_sf_datasets.py without triggering its main()."""
    spec = importlib.util.spec_from_file_location("create_sf_datasets", MAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_sf_datasets"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    dfrom = argv[0] if len(argv) > 0 else None
    dto = argv[1] if len(argv) > 1 else None
    outdir = argv[2] if len(argv) > 2 else os.getcwd()

    mod = load_main()

    # This file and create_sf_datasets.py have to move together: the diagnostic imports the main
    # script from its OWN directory, so copying only one of them to another machine leaves them
    # skewed. Without this check that surfaces ~100 lines later as
    # `Binder Error: Table "s" does not have a column named "sp_name"`, which points at the
    # parquet input rather than at the real cause.
    missing = [c for c in ("sp_name",) if c not in mod.SALES_COLUMNS]
    if missing:
        print(f"\nVERSION SKEW: {os.path.basename(MAIN)} in this folder does not read "
              f"{missing} -- it is an older copy.\n"
              f"  imported: {MAIN}\n"
              f"  Copy the matching create_sf_datasets.py here and re-run. Both files must be "
              f"updated together.", file=sys.stderr)
        return 1

    # The QA block belongs to a publishing run, not to this. Keep the output to the point.
    mod.CONFIG["verbose_qa"] = False

    print(f"reading inputs from {mod.CONFIG['input_dir']} ...")
    opp, sales, users, events, tasks, leads, bdc = mod.load_inputs()
    con = duckdb.connect()
    try:
        print("building (same staging the outputs use) ...")
        mod.build(con, opp, sales, users, events, tasks, leads, bdc)
        detail = con.execute(DETAIL_SQL, [dfrom, dto]).df()
        split = con.execute(SPLIT_SQL, [dfrom, dto]).fetchone()
    finally:
        con.close()

    tag = f"{dfrom or 'all'}_{dto or 'all'}".replace("-", "")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"unmatched_rep_{tag}.csv")
    detail.to_csv(path, index=False, encoding="utf-8")

    in_range = split[0]
    pct = lambda p: "n/a" if not in_range else f"{100.0 * p / in_range:.1f}%"
    print(f"\nrange   {dfrom or '(unbounded)'} .. {dto or '(unbounded)'}")
    print(f"sales-credit rows in range                      {in_range:,}")
    print(f"  rep resolved, opportunity matched             {split[1]:,}  ({pct(split[1])})")
    print(f"  rep UNRESOLVED, opportunity matched  <-- this {split[2]:,}  ({pct(split[2])})")
    print(f"  rep unresolved, no opportunity                {split[3]:,}  ({pct(split[3])})")
    print(f"  rep resolved, no opportunity                  {split[4]:,}  ({pct(split[4])})")

    if len(detail):
        # Mutually exclusive triage, most-specific first. Each bucket is a different
        # conversation with a different person, which is the whole point of the extra columns.
        unassigned = str(mod.CONFIG["unassigned_user_id"])
        key = lambda c: (detail[c].fillna("").str.upper()
                         .str.replace(r"\s+", " ", regex=True).str.strip())
        same_name = key("advent_sp_name").eq(key("sf_salesperson_name")) \
            & detail.advent_sp_name.notna() & detail.sf_salesperson_name.notna()

        no_sf_rep = detail.sf_salesperson_id.eq(unassigned)
        name_differs = ~no_sf_rep & ~same_name
        no_advent = ~no_sf_rep & same_name & detail.sf_assigned_advent_id.isna()
        id_differs = (~no_sf_rep & same_name & detail.sf_assigned_advent_id.notna()
                      & (detail.sf_assigned_advent_id != detail.advent_sp_id))
        id_same = (~no_sf_rep & same_name & detail.sf_assigned_advent_id.notna()
                   & (detail.sf_assigned_advent_id == detail.advent_sp_id))

        print(f"\nof the {len(detail):,} rows in the file "
              f"({detail.deal_number.nunique():,} distinct deals):")
        print(f"  same person, NO advent_id -> populate it       {int(no_advent.sum()):,}")
        print(f"  same person, advent ids differ -> wrong id     {int(id_differs.sum()):,}")
        print(f"  same person, ids agree -> STORE mismatch       {int(id_same.sum()):,}")
        print(f"  different people -> not a data bug             {int(name_differs.sum()):,}")
        print(f"  opportunity has no salesperson either          {int(no_sf_rep.sum()):,}")
        print(f"  advent_sp_name blank (column not populated)    "
              f"{int(detail.advent_sp_name.isna().sum()):,}")

        # Cuts across the buckets above, so counted separately: is the STORE the reason the
        # bridge missed, rather than the id? Both columns are already in Advent space here.
        reg = detail.advent_id_registered_at
        wrong_store = reg.notna() & (reg != detail.dealercode)
        nowhere = reg.isna()
        print(f"\nwhy the bridge missed, independent of the above:")
        print(f"  advent id registered at a DIFFERENT store      {int(wrong_store.sum()):,}")
        print(f"  advent id in no user record at all             {int(nowhere.sum()):,}")
    print(f"\nwrote {path}  ({len(detail):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
