# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for loaders and configuration. No network or model calls."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from app import config, forecast
from app.forecast import PRECIP, WIND


def test_sample_forecast_shape():
    sites = forecast.load_sites()
    assert set(sites) == {
        "prairie-ridge",
        "highland-moor",
        "elbe-marsh",
        "dogger-offshore",
    }
    f = forecast.SampleSource().get(sites["prairie-ridge"])
    assert f.kind == "members"
    assert f.init_time == datetime(2026, 10, 12, tzinfo=UTC)
    assert len(f.times) == 120
    assert f.times[1] - f.times[0] == timedelta(hours=1)
    assert len(f.members[WIND]) == 32
    assert all(len(m) == 120 for m in f.members[WIND] + f.members[PRECIP])
    assert all(v >= 0 for m in f.members[PRECIP] for v in m)


def test_sample_unknown_site_raises():
    site = forecast.Site("nowhere", "n", 0, 0, 0, 100)
    with pytest.raises(KeyError):
        forecast.SampleSource().get(site)


def test_custom_sites_file(tmp_path):
    path = tmp_path / "sites.json"
    path.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "site_id": "x",
                        "name": "X",
                        "latitude": 1,
                        "longitude": 2,
                        "utc_offset_hours": 0,
                        "hub_height_m": 100,
                    }
                ]
            }
        )
    )
    assert forecast.load_sites(path)["x"].longitude == 2


def test_index_range():
    f = forecast.SampleSource().get(forecast.load_sites()["elbe-marsh"])
    lo, hi = f.index_range(f.times[10], f.times[20])
    assert (lo, hi) == (10, 20)
    assert f.index_range(None, None) == (0, 120)


def test_parse_time_defaults_to_utc():
    assert forecast.parse_time("2026-10-12T06:00:00").tzinfo == UTC
    assert forecast.parse_time("2026-10-12T06:00:00Z").hour == 6


@pytest.mark.parametrize(
    "table",
    ["", "dataset.table", "p.d.t`; DROP TABLE x; --", "p.d.t.u", "p.d.t name"],
)
def test_validate_table_rejects_unsafe_ids(table):
    with pytest.raises(ValueError):
        forecast.validate_table(table)


def test_forecast_query_uses_documented_schema():
    sql = forecast.build_forecast_query(
        "my-proj.weathernext.weathernext_3_0_0_0p1deg"
    )
    assert (
        "`my-proj.weathernext.weathernext_3_0_0_0p1deg` AS t, t.forecast AS f"
        in sql
    )
    assert "t.init_time = @init_time" in sql
    assert "ST_INTERSECTS(t.geography_polygon, ST_GEOGPOINT(@lon, @lat))" in sql
    for col in (
        "wind_speed_100m_p10",
        "wind_speed_100m_p90",
        "total_precipitation_1hr_p50",
    ):
        assert f"f.{col}" in sql
    assert "init_time >= TIMESTAMP_SUB" in forecast.build_latest_init_query(
        "p.d.t"
    )


def _rows(init, hours):
    rows = []
    for h in hours:
        row = {"init_time": init, "time": init + timedelta(hours=h), "hours": h}
        for col, level in forecast.QUANTILE_COLUMNS.items():
            row[f"{WIND}_{col}"] = 5.0 + 10 * level
            row[f"{PRECIP}_{col}"] = 0.001 * level  # metres
        rows.append(row)
    return rows


def test_rows_to_forecast_sorts_dedups_and_converts_units():
    init = datetime(2026, 10, 12, tzinfo=UTC)
    rows = _rows(init, [3, 1, 2, 2])
    f = forecast.rows_to_forecast(rows, "s", "bq")
    assert f.kind == "quantiles"
    assert [t.hour for t in f.times] == [1, 2, 3]
    assert f.quantiles[WIND][0.5] == [10.0, 10.0, 10.0]
    assert f.quantiles[PRECIP][0.9] == pytest.approx([0.9, 0.9, 0.9])  # mm
    with pytest.raises(ValueError):
        forecast.rows_to_forecast([], "s", "bq")


class FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, init):
        self.init = init
        self.queries = []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        if "MAX(init_time)" in sql:
            return FakeJob([{"init_time": self.init}])
        return FakeJob(_rows(self.init, range(1, 7)))


def test_bigquery_source_with_fake_client():
    init = datetime(2026, 10, 12, 6, tzinfo=UTC)
    client = FakeClient(init)
    source = forecast.BigQuerySource(
        "p.d.weathernext_3_0_0_0p1deg", client=client
    )
    site = forecast.load_sites()["elbe-marsh"]
    f = source.get(site)
    assert f.init_time == init
    assert len(f.times) == 6
    params = {p.name: p.value for p in client.queries[1][1].query_parameters}
    assert params == {
        "init_time": init,
        "lon": 9.2,
        "lat": 53.9,
        "max_hours": 120,
    }
    source.get(site)  # cached: no new queries
    assert len(client.queries) == 2


def test_bigquery_source_pinned_init_skips_lookup():
    client = FakeClient(datetime(2026, 10, 12, tzinfo=UTC))
    source = forecast.BigQuerySource(
        "p.d.t", init_time="2026-10-12T00:00:00Z", client=client
    )
    source.get(forecast.load_sites()["elbe-marsh"])
    assert len(client.queries) == 1


def test_get_source_rejects_unknown_kind(monkeypatch):
    forecast.get_source.cache_clear()
    monkeypatch.setenv("WEATHERNEXT_SOURCE", "csv")
    try:
        with pytest.raises(ValueError):
            forecast.get_source()
    finally:
        forecast.get_source.cache_clear()


def test_task_overrides_inline_and_file(tmp_path):
    tasks = config.load_tasks('{"main_component_lift": {"max_wind_ms": 8.5}}')
    assert tasks["main_component_lift"].max_wind_ms == 8.5
    assert tasks["main_component_lift"].min_hours == 8  # untouched
    path = tmp_path / "limits.json"
    path.write_text(
        json.dumps(
            {
                "cable_pull": {
                    "description": "Ground cable pull",
                    "max_wind_ms": 20,
                    "max_gust_ms": 25,
                    "max_precip_mm": 5,
                    "min_hours": 3,
                }
            }
        )
    )
    assert config.load_tasks(str(path))["cable_pull"].min_hours == 3


def test_task_overrides_reject_bad_input():
    with pytest.raises(ValueError, match="Unknown fields"):
        config.load_tasks('{"blade_repair": {"max_wind": 9}}')
    with pytest.raises(ValueError, match="Incomplete"):
        config.load_tasks('{"new_task": {"max_wind_ms": 9}}')
    assert config.load_tasks("") == config.DEFAULT_TASKS


def test_empty_env_values_mean_defaults(monkeypatch):
    for var in (
        "WEATHERNEXT_SOURCE",
        "WEATHERNEXT_MAX_HOURS",
        "SITES_JSON",
        "SAMPLE_FORECAST_JSON",
        "TASK_LIMITS_JSON",
        "GUST_FACTOR",
        "SHEAR_EXPONENT",
    ):
        monkeypatch.setenv(var, "")
    forecast.get_source.cache_clear()
    try:
        assert isinstance(forecast.get_source(), forecast.SampleSource)
        assert len(forecast.load_sites()) == 4
        assert config.load_tasks() == config.DEFAULT_TASKS
        assert config.gust_factor() == config.DEFAULT_GUST_FACTOR
        assert config.shear_exponent() == config.DEFAULT_SHEAR_EXPONENT
    finally:
        forecast.get_source.cache_clear()
