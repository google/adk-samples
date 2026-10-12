#!/usr/bin/env python3
"""Read-only probe of a BlueZoo Data Warehouse account.

Answers one question: **what does this specific tenant's warehouse actually
look like?** — because table availability, cluster hostname, and group/sensor
configuration all vary per BlueZoo customer, and our connector has to work for
any of them (see `working-docs/bluezoo-live-verification/findings.md`).

Read-only by construction: only `list_tables`, `desc_table`, and `SELECT`
statements are ever sent (`run_query` is server-side SELECT-only regardless).
Nothing is written to BlueZoo, and the AccessKey is never printed or persisted.

Usage
-----
    # key from app/.env (BLUEZOO_ACCESS_KEY=...) or the environment
    python scripts/bluezoo_probe.py --out .docs/version2-plan/working-docs/bluezoo-live-verification/scan

    # another customer / cluster
    BLUEZOO_BASE_URL=https://<their-cluster-host>/v2/dwh python scripts/bluezoo_probe.py

Environment
-----------
    BLUEZOO_ACCESS_KEY   required. Dashboard -> Profile -> AccessKey. Read from
                         app/.env or the environment.
    BLUEZOO_BASE_URL     optional. Defaults to the Apollo cluster host. Read from
                         app/.env or the environment. The hostname is
                         CLUSTER-scoped, not global: a key from cluster A returns
                         BAD_TOKEN against cluster B's host. The dashboard Profile
                         screen names the cluster, and each cluster issues its own
                         AccessKey.

Cost
----
BlueZoo meters a monthly BYTES-SCANNED allowance (BigQuery style) as an abuse
guard. The ceiling is per-tenant configuration they raise on request (one live
tenant sits at 500 GB per sensor location), but the DEFAULT is 1 GB per sensor
location — low enough that ~735 MB of full-history aggregates exhausted it in
practice. Once spent, EVERY `run_query` fails until it resets or is raised,
including single-day ones. Metadata (`list_tables`, `desc_table`) and the
Real-time API are exempt. This probe therefore counts over the last
DEFAULT_WINDOW_DAYS by default; `--full-history` is an explicit opt-in and is
only safe on a tenant known to be empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

DEFAULT_BASE_URL = "https://hermes.apollo.bluezoo.io/v2/dwh"

# BlueZoo rejects any run_query without a constraint on one of these columns.
# Undocumented (their own published example would fail), enforced server-side.
# The likely reason is partition pruning: these tables appear to be time-
# partitioned, and a query with no time predicate would scan all of history.
TIME_CONSTRAINT_COLUMNS = ("timestamp", "date_start", "date_end", "date")

# BlueZoo meters a monthly BYTES-SCANNED quota (BigQuery style), defaulting to
# "1GB per sensor location" — and on a tenant with ~5M rows/table, ~735MB of
# full-history aggregates was enough to exhaust it, after which EVERY run_query
# fails, including single-day ones. The ceiling is negotiable (they raise it on
# request), but the default is not generous. Hence: count over a narrow recent
# window by default, and make full history an explicit opt-in.
DEFAULT_WINDOW_DAYS = 30

# Approximate on-disk width per BigQuery's documented type sizes, used to warn
# before a query rather than discover the cost by hitting the wall.
TYPE_BYTES = {
    "INT64": 8,
    "FLOAT64": 8,
    "NUMERIC": 16,
    "BOOL": 1,
    "TIMESTAMP": 8,
    "DATE": 8,
    "DATETIME": 8,
}
STRING_BYTES = 20  # 2 bytes + UTF-8 length; observed values run ~6-20 chars

# Tables worth sampling for row counts, if the tenant has them. Everything
# else is schema-only — this list exists to keep the probe cheap, not to
# assert what a tenant "should" have.
COUNT_TABLES = (
    "sensor_visits",
    "sensor_visitors",
    "sensor_dwell",
    "sensor_visitors_per_minute",
    "group_uv_daily",
    "group_sensor_history",
)

FULL_HISTORY = (
    date(2020, 1, 1),
    date(2030, 12, 31),
)  # opt-in only: --full-history


class BlueZooError(RuntimeError):
    """An API call failed or the response was not usable."""


class QuotaExceeded(BlueZooError):
    """The tenant's monthly bytes-scanned allowance is spent.

    Distinct from a generic 400 because the remedy is completely different:
    nothing you can do to the query helps, every `run_query` fails until the
    allowance resets or BlueZoo raises it, and metadata plus the Real-time API
    keep working meanwhile.
    """


class BlueZooProbe:
    """Minimal read-only Data Warehouse client."""

    def __init__(
        self,
        access_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 60,
    ):
        if not access_key:
            raise BlueZooError(
                "No BlueZoo AccessKey. Set BLUEZOO_ACCESS_KEY in app/.env or the "
                "environment (dashboard -> Profile -> AccessKey). Never commit it."
            )
        self._key = access_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(
        self,
        path: str,
        body: str | None = None,
        content_type: str | None = None,
    ):
        request = urllib.request.Request(  # noqa: S310
            f"{self.base_url}/{path}", data=body.encode() if body else None
        )
        request.add_header("Authorization", f"AccessKey {self._key}")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self.timeout
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:  # 400/403 carry a JSON body
            detail = exc.read().decode(errors="replace")[:300]
            if "fair use limit" in detail or "data scanned" in detail:
                raise QuotaExceeded(
                    f"{path}: monthly bytes-scanned allowance exhausted — every run_query "
                    f"will fail until it resets or BlueZoo raises it. Metadata and the "
                    f"Real-time API still work. Server said: {detail}"
                ) from exc
            raise BlueZooError(f"{path}: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise BlueZooError(f"{path}: {exc.reason}") from exc
        try:
            return json.loads(body)
        except ValueError as exc:
            # A 200 that isn't JSON usually means a proxy or a login page —
            # likeliest failure when pointed at the wrong cluster host.
            raise BlueZooError(
                f"{path}: non-JSON response: {body[:200]!r}"
            ) from exc

    def list_tables(self) -> list[str]:
        """Tables this account is entitled to — a per-tenant capability probe."""
        rows = self._call("list_tables")
        try:
            return sorted(row["table_name"] for row in rows)
        except (TypeError, KeyError) as exc:
            raise BlueZooError(
                f"list_tables returned an unexpected shape: {rows!r:.200}"
            ) from exc

    def describe(self, table: str) -> list[dict]:
        return self._call(
            "desc_table", json.dumps({"table_name": table}), "application/json"
        )

    def query(self, sql: str):
        """Run a SELECT.

        Two pre-flight refusals, both before anything leaves the process:

        * Non-SELECT statements. `run_query` is SELECT-only server-side, so
          this is belt-and-braces — but it makes "read-only" a property of
          this class rather than a promise about its call sites.
        * Statements that mention none of the time columns BlueZoo demands a
          constraint on. This is a *best-effort* check: it scans the whole
          statement rather than parsing the WHERE clause, so it catches the
          realistic mistake (forgot the filter entirely) and would pass a
          contrived one (`order by timestamp`). Worth having anyway, because
          the failure then names the real cause instead of surfacing BlueZoo's
          message after a round trip.
        """
        lowered = sql.lower()
        if not lowered.lstrip().startswith("select"):
            raise BlueZooError(
                f"This probe issues SELECT statements only: {sql}"
            )
        if not any(column in lowered for column in TIME_CONSTRAINT_COLUMNS):
            raise BlueZooError(
                "BlueZoo requires every query to constrain one of "
                f"{', '.join(TIME_CONSTRAINT_COLUMNS)} in its WHERE clause: {sql}"
            )
        return self._call("run_query", sql, "text/plain")


def time_column_for(columns: list[dict]) -> str | None:
    """Which time column this table can be filtered on, if any.

    Tables disagree: sensor_* carry `timestamp`, group_uv_daily carries `date`,
    the group_convert family carries `date_start`. Pick whichever exists rather
    than assuming one.
    """
    names = {column["column_name"] for column in columns}
    for candidate in TIME_CONSTRAINT_COLUMNS:
        if candidate in names:
            return candidate
    return None


def row_width_bytes(
    columns: list[dict], selected: list[str] | None = None
) -> int:
    """Approximate bytes read per scanned row for `selected` columns.

    BigQuery bills columns actually read, so `select *` on a wide table is the
    expensive mistake: `sensor_dwell` is 120 columns / ~1 KB per row, which
    over 5M rows is ~5 GB in a single statement. Call this before writing a
    query, not after.
    """
    wanted = set(selected) if selected is not None else None
    return sum(
        TYPE_BYTES.get(c["data_type"], STRING_BYTES)
        for c in columns
        if wanted is None or c["column_name"] in wanted
    )


def scan(
    probe: BlueZooProbe,
    count_tables: tuple[str, ...] = COUNT_TABLES,
    window: tuple[date, date] | None = None,
) -> dict:
    """Read-only scan: entitlements, every table's schema, row counts.

    `window` defaults to the last DEFAULT_WINDOW_DAYS rather than all history,
    because the counting queries read the time column of every row they span —
    cheap on an empty tenant, expensive on a populated one.
    """
    tables = probe.list_tables()
    schemas: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for table in tables:
        try:
            schemas[table] = probe.describe(table)
        except BlueZooError as exc:
            errors[table] = str(exc)

    counts: dict[str, object] = {}
    start, end = window or (
        datetime.now(UTC).date() - timedelta(days=DEFAULT_WINDOW_DAYS),
        datetime.now(UTC).date() + timedelta(days=1),
    )
    for table in count_tables:
        if table not in schemas:
            continue  # not entitled on this tenant — expected, not an error
        column = time_column_for(schemas[table])
        if column is None:
            counts[table] = "no filterable time column"
            continue
        try:
            # Interpolation is safe here: `column` is a server-returned column
            # name, and `table` had to match a server-returned table name to
            # get past the membership check above.
            rows = probe.query(
                f"select count(*) as n from {table} "  # noqa: S608
                f"where {column} >= '{start.isoformat()}' and {column} <= '{end.isoformat()}'"
            )
            # COUNT(*) always yields exactly one row with an `n`. Anything else
            # is a surprise worth surfacing, not worth papering over as 0 — the
            # counts in this artifact are load-bearing evidence.
            counts[table] = (
                rows[0]["n"]
                if rows and "n" in rows[0]
                else f"unexpected: {rows!r:.120}"
            )
        except BlueZooError as exc:
            counts[table] = f"error: {exc}"

    return {
        "scanned_at": datetime.now(UTC).isoformat(),
        "base_url": probe.base_url,
        "count_window": [start.isoformat(), end.isoformat()],
        "table_count": len(tables),
        "tables": tables,
        "schemas": schemas,
        "schema_errors": errors,
        "row_counts": counts,
        # What a `select *` would read per row. Recorded so the next person
        # can see which tables are expensive before writing a query.
        "select_star_bytes_per_row": {
            t: row_width_bytes(c) for t, c in schemas.items()
        },
    }


def render_markdown(result: dict) -> str:
    """Human-readable digest of a scan — the reviewable half of the artifact."""
    lines = [
        "# BlueZoo live schema scan",
        "",
        f"- **Scanned:** {result['scanned_at']}",
        f"- **Base URL:** `{result['base_url']}`",
        f"- **Tables entitled:** {result['table_count']}",
        "",
        "Generated by `scripts/bluezoo_probe.py` (read-only).",
        "",
        "## Row counts",
        "",
        f"Counted over `{result['count_window'][0]}` … `{result['count_window'][1]}`.",
        "",
        "| Table | Rows |",
        "|---|---|",
    ]
    for table, count in result["row_counts"].items():
        lines.append(f"| `{table}` | {count} |")
    lines += ["", "## Schemas", ""]
    for table in result["tables"]:
        columns = result["schemas"].get(table)
        if columns is None:
            lines += [
                f"### `{table}`",
                "",
                f"ERROR: {result['schema_errors'].get(table)}",
                "",
            ]
            continue
        lines += [
            f"### `{table}` ({len(columns)} columns)",
            "",
            "| Column | Type |",
            "|---|---|",
        ]
        lines += [
            f"| `{c['column_name']}` | {c['data_type']} |" for c in columns
        ]
        lines.append("")
    return "\n".join(lines)


def load_env_var(name: str, env_file: Path | None = None) -> str:
    """Environment wins; otherwise app/.env (gitignored, never committed)."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = (
        env_file or Path(__file__).resolve().parent.parent / "app" / ".env"
    )
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, raw = line.partition("=")
            if key.strip() == name:
                return raw.strip().strip("'\"")
    return ""


def load_access_key() -> str:
    return load_env_var("BLUEZOO_ACCESS_KEY")


def run_sql(probe: BlueZooProbe, sql: str) -> str:
    """One guarded SELECT, rendered as JSON for the terminal."""
    return json.dumps(probe.query(sql), indent=2, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="Directory to write scan.json + scan.md into"
    )
    parser.add_argument(
        "--base-url",
        default=load_env_var("BLUEZOO_BASE_URL") or DEFAULT_BASE_URL,
        help="Cluster-specific Data Warehouse base URL (env or app/.env BLUEZOO_BASE_URL)",
    )
    parser.add_argument(
        "--sql",
        help="Run one SELECT (read-only + time-constraint guards apply) and print rows as JSON",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help=(
            f"Count over {FULL_HISTORY[0]}..{FULL_HISTORY[1]} instead of the last "
            f"{DEFAULT_WINDOW_DAYS} days. EXPENSIVE on a populated tenant: the counting "
            "queries read the time column of every row they span, against a metered "
            "monthly bytes-scanned allowance. Use only on a tenant known to be empty."
        ),
    )
    args = parser.parse_args()

    try:
        probe = BlueZooProbe(load_access_key(), args.base_url)
        if args.sql:
            print(run_sql(probe, args.sql))
            return 0
        result = scan(probe, window=FULL_HISTORY if args.full_history else None)
    except QuotaExceeded as exc:
        print(f"QUOTA EXHAUSTED: {exc}", file=sys.stderr)
        return 2
    except BlueZooError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    markdown = render_markdown(result)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "scan.md").write_text(markdown + "\n")
        print(f"Wrote {args.out / 'scan.json'} and {args.out / 'scan.md'}")
    else:
        print(markdown)

    empty = [t for t, n in result["row_counts"].items() if n == 0]
    if empty:
        print(
            f"\nNOTE: {len(empty)} sampled table(s) returned zero rows in "
            f"{result['count_window'][0]}..{result['count_window'][1]} — schema is "
            "verifiable, values are not.",
            file=sys.stderr,
        )
    widest = sorted(
        result["select_star_bytes_per_row"].items(), key=lambda kv: -kv[1]
    )[:1]
    if widest:
        table, width = widest[0]
        print(
            f"\nCOST NOTE: `select * from {table}` reads ~{width} bytes/row. BlueZoo "  # noqa: S608
            "meters a monthly bytes-scanned allowance, so name your columns explicitly "
            "and keep time windows narrow.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
