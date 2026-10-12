"""Unit tests for the BlueZoo probe's pure helpers (no network).

The probe's value is that it works against ANY BlueZoo tenant, so the things
worth pinning are the tenant-variable behaviors: the undocumented mandatory
time filter, per-table time-column disagreement, table entitlements differing
per account, and a missing key failing loudly instead of silently.
"""

import importlib.util
import io
import json
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bluezoo_probe.py"
)
_spec = importlib.util.spec_from_file_location("bluezoo_probe", _MODULE_PATH)
bluezoo_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bluezoo_probe)

BlueZooError = bluezoo_probe.BlueZooError
BlueZooProbe = bluezoo_probe.BlueZooProbe


class FakeProbe(BlueZooProbe):
    """Probe with only the HTTP layer replaced.

    Overrides `_call`, not `query`/`list_tables`, so the real guards and the
    real response handling stay in the path — a fake that reimplemented them
    would keep passing after they changed.
    """

    def __init__(self, tables, schemas, counts=None, describe_errors=()):
        super().__init__(access_key="fake-key")
        self._tables = tables
        self._schemas = schemas
        self._counts = counts or {}
        self._describe_errors = set(describe_errors)
        self.queries = []

    def _call(self, path, body=None, content_type=None):
        if path == "list_tables":
            return [{"table_name": name} for name in self._tables]
        if path == "desc_table":
            table = json.loads(body)["table_name"]
            if table in self._describe_errors:
                raise BlueZooError(f"desc_table: HTTP 403 {table}")
            return self._schemas[table]
        self.queries.append(body)
        table = body.split(" from ")[1].split(" ")[0]
        return [{"n": self._counts.get(table, 0)}]


def _cols(*names):
    return [{"column_name": n, "data_type": "INT64"} for n in names]


class TestMissingKey:
    def test_empty_key_raises_actionable_error(self):
        with pytest.raises(BlueZooError, match="AccessKey"):
            BlueZooProbe(access_key="")


class TestQueryGuards:
    """BlueZoo rejects any query lacking a date_start/date_end/timestamp
    filter — undocumented, and their own published example would fail. The
    probe also refuses non-SELECTs client-side, so read-only is a property of
    the class rather than a promise about its callers."""

    def test_query_without_time_constraint_is_refused(self):
        probe = BlueZooProbe(access_key="fake-key")
        with pytest.raises(BlueZooError, match=r"time constraint|constrain"):
            probe.query("select * from sensor_visitors limit 1")

    @pytest.mark.parametrize(
        "statement",
        [
            "delete from sensor_visits where timestamp > '2020-01-01'",
            "update sensor_visits set valid = false where timestamp > '2020-01-01'",
            "drop table sensor_visits -- timestamp",
        ],
    )
    def test_non_select_statements_never_leave_the_process(
        self, statement, monkeypatch
    ):
        probe = BlueZooProbe(access_key="fake-key")
        monkeypatch.setattr(
            probe,
            "_call",
            lambda *a, **k: pytest.fail("a non-SELECT was transmitted"),
        )
        with pytest.raises(BlueZooError, match="SELECT"):
            probe.query(statement)

    @pytest.mark.parametrize(
        "column", ["timestamp", "date_start", "date_end", "date"]
    )
    def test_each_accepted_time_column_passes_the_guard(
        self, column, monkeypatch
    ):
        probe = BlueZooProbe(access_key="fake-key")
        monkeypatch.setattr(probe, "_call", lambda *a, **k: [{"n": 1}])
        assert probe.query(
            f"select count(*) as n from t where {column} >= '2026-01-01'"  # noqa: S608
        )


class TestTimeColumnSelection:
    """Tables disagree on which column is filterable."""

    def test_sensor_tables_use_timestamp(self):
        assert (
            bluezoo_probe.time_column_for(_cols("sensor_id", "timestamp"))
            == "timestamp"
        )

    def test_group_uv_daily_uses_date_not_timestamp(self):
        assert (
            bluezoo_probe.time_column_for(_cols("group_id", "date")) == "date"
        )

    def test_group_convert_uses_date_start(self):
        assert (
            bluezoo_probe.time_column_for(_cols("campaign_id", "date_start"))
            == "date_start"
        )

    def test_table_with_no_time_column_returns_none(self):
        assert (
            bluezoo_probe.time_column_for(_cols("group_id", "sensor_id"))
            is None
        )


class TestTenantVariability:
    """Table availability is a per-account entitlement, so the scan must
    degrade rather than fail when a tenant lacks an optional table."""

    def test_missing_optional_table_is_skipped_not_an_error(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
            counts={"sensor_visits": 42},
        )
        result = bluezoo_probe.scan(
            probe, count_tables=("sensor_visits", "sensor_visitors_per_minute")
        )
        assert result["row_counts"] == {"sensor_visits": 42}
        assert "sensor_visitors_per_minute" not in result["row_counts"]

    def test_scan_records_entitlements_and_schemas(self):
        probe = FakeProbe(
            tables=["sensor_visits", "group_sensor_history"],
            schemas={
                "sensor_visits": _cols("sensor_id", "timestamp"),
                "group_sensor_history": _cols(
                    "group_id", "sensor_id", "timestamp"
                ),
            },
        )
        result = bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        assert result["table_count"] == 2
        assert set(result["schemas"]) == {
            "sensor_visits",
            "group_sensor_history",
        }
        assert result["schema_errors"] == {}

    def test_zero_rows_is_a_valid_answer_not_a_failure(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
            counts={"sensor_visits": 0},
        )
        result = bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        assert result["row_counts"]["sensor_visits"] == 0

    def test_one_unreadable_table_does_not_abort_the_scan(self):
        """A tenant where one desc_table 403s while the rest succeed is
        precisely what this script exists to survive."""
        probe = FakeProbe(
            tables=["sensor_visits", "sensor_pulses"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
            describe_errors=["sensor_pulses"],
        )
        result = bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        assert set(result["schemas"]) == {"sensor_visits"}
        assert "sensor_pulses" in result["schema_errors"]
        assert result["table_count"] == 2  # entitlement is still recorded
        assert "sensor_pulses" in bluezoo_probe.render_markdown(result)

    def test_counted_query_carries_the_time_filter(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
        )
        bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        assert "where timestamp >=" in probe.queries[0]


class TestScanCost:
    """BlueZoo meters a small monthly bytes-scanned allowance, and exhausting
    it disables run_query entirely. Cost control is a correctness concern for
    this script, not a nicety — 735 MB of full-history aggregates spent a real
    tenant's whole month."""

    def test_default_window_is_recent_not_all_history(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
        )
        result = bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        start, end = result["count_window"]
        assert start > "2024", (
            f"default window reaches back to {start} — too expensive"
        )
        span = date.fromisoformat(end) - date.fromisoformat(start)
        assert span <= timedelta(days=bluezoo_probe.DEFAULT_WINDOW_DAYS + 1)

    def test_full_history_is_available_but_only_when_asked_for(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
        )
        result = bluezoo_probe.scan(
            probe,
            count_tables=("sensor_visits",),
            window=bluezoo_probe.FULL_HISTORY,
        )
        assert result["count_window"] == ["2020-01-01", "2030-12-31"]

    def test_row_width_flags_the_wide_table_trap(self):
        """sensor_dwell's 106 float bins are why `select *` is dangerous."""
        wide = _cols("timestamp") + [
            {"column_name": f"distribution_bin_{i}", "data_type": "FLOAT64"}
            for i in range(106)
        ]
        assert bluezoo_probe.row_width_bytes(wide) == 107 * 8
        # Naming columns explicitly is the mitigation, and it must be dramatic.
        assert bluezoo_probe.row_width_bytes(wide, ["timestamp"]) == 8

    def test_strings_are_costed_above_fixed_width_types(self):
        cols = [{"column_name": "sensor_name", "data_type": "STRING"}]
        assert (
            bluezoo_probe.row_width_bytes(cols)
            == bluezoo_probe.STRING_BYTES
            > 8
        )

    def test_quota_exhaustion_is_its_own_error_type(self, monkeypatch):
        """A spent allowance is unfixable by retrying or narrowing, so callers
        need to tell it apart from an ordinary bad request."""
        probe = BlueZooProbe(access_key="fake-key")

        def raise_quota(*a, **k):
            raise urllib.error.HTTPError(
                "u",
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    b'{"message":"You\'ve reached your monthly fair use limit of 1GB '
                    b'data scanned per sensor."}'
                ),
            )

        monkeypatch.setattr(urllib.request, "urlopen", raise_quota)
        with pytest.raises(
            bluezoo_probe.QuotaExceeded, match="allowance exhausted"
        ):
            probe.query("select 1 from t where timestamp >= '2026-01-01'")

    def test_ordinary_400_is_not_mistaken_for_quota(self, monkeypatch):
        probe = BlueZooProbe(access_key="fake-key")

        def raise_bad_sql(*a, **k):
            raise urllib.error.HTTPError(
                "u",
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    b'{"message":"invalidQuery: Unrecognized name: nope at [1:8]"}'
                ),
            )

        monkeypatch.setattr(urllib.request, "urlopen", raise_bad_sql)
        with pytest.raises(BlueZooError) as caught:
            probe.query("select nope from t where timestamp >= '2026-01-01'")
        assert not isinstance(caught.value, bluezoo_probe.QuotaExceeded)


class TestLoadEnvVar:
    """Base URL and key both come from the environment or app/.env —
    the env file half previously existed only for the AccessKey."""

    def test_environment_wins_over_env_file(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("BLUEZOO_BASE_URL=https://file.example/v2/dwh\n")
        monkeypatch.setenv("BLUEZOO_BASE_URL", "https://env.example/v2/dwh")
        assert (
            bluezoo_probe.load_env_var("BLUEZOO_BASE_URL", env_file)
            == "https://env.example/v2/dwh"
        )

    def test_env_file_value_is_read_and_unquoted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BLUEZOO_BASE_URL", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            'OTHER=x\nBLUEZOO_BASE_URL="https://file.example/v2/dwh"\n'
        )
        assert (
            bluezoo_probe.load_env_var("BLUEZOO_BASE_URL", env_file)
            == "https://file.example/v2/dwh"
        )

    def test_missing_everywhere_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BLUEZOO_BASE_URL", raising=False)
        assert (
            bluezoo_probe.load_env_var("BLUEZOO_BASE_URL", tmp_path / "absent")
            == ""
        )


class TestRunSql:
    def test_renders_rows_as_json(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
            counts={"sensor_visits": 42},
        )
        out = bluezoo_probe.run_sql(
            probe,
            "select count(*) as n from sensor_visits where timestamp >= '2026-07-01'",
        )
        assert json.loads(out) == [{"n": 42}]

    def test_guards_still_apply(self):
        probe = FakeProbe(tables=[], schemas={})
        with pytest.raises(BlueZooError, match="constrain"):
            bluezoo_probe.run_sql(probe, "select 1 from t")
        with pytest.raises(BlueZooError, match="SELECT"):
            bluezoo_probe.run_sql(
                probe, "delete from t where timestamp > '2020-01-01'"
            )


class TestRendering:
    def test_markdown_digest_includes_counts_and_columns(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
            counts={"sensor_visits": 7},
        )
        markdown = bluezoo_probe.render_markdown(
            bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        )
        assert "`sensor_visits`" in markdown
        assert "| 7 |" in markdown
        assert "`timestamp`" in markdown

    def test_scan_never_contains_the_access_key(self):
        probe = FakeProbe(
            tables=["sensor_visits"],
            schemas={"sensor_visits": _cols("sensor_id", "timestamp")},
        )
        payload = bluezoo_probe.render_markdown(
            bluezoo_probe.scan(probe, count_tables=("sensor_visits",))
        )
        assert "fake-key" not in payload
