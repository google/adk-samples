# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Price-related utility functions for FOMC Research Agent."""

import datetime
import logging
import math
import os
from collections.abc import Sequence

from absl import app
from google.cloud import bigquery

bqclient = bigquery.Client()
logger = logging.getLogger(__name__)

MOVE_SIZE_BP = 25
DATASET_NAME = os.getenv("GOOGLE_CLOUD_BQ_DATASET", "fomc_research_agent")
TIMESERIES_CODES = os.getenv(
    "GOOGLE_GENAI_FOMC_AGENT_TIMESERIES_CODES",
    "SFRH5,SFRZ5")


def fetch_prices_from_bq(
    timeseries_codes: list[str], dates: list[datetime.date]
) -> dict[dict[datetime.date, float]]:
    """Fetches prices from Bigquery.

    Args:
      timeseries_codes: List of timeseries codes to fetch.
      dates: List of dates to fetch.

    Returns:
      Dictionary of timeseries codes to dictionaries of dates to prices.
    """

    logger.debug("fetch_prices_from_bq: timeseries_codes: %s", timeseries_codes)
    logger.debug("fetch_prices_from_bq: dates: %s", dates)

    query = f"""
SELECT DISTINCT timeseries_code, date, value
FROM {DATASET_NAME}.timeseries_data
WHERE timeseries_code IN UNNEST(@timeseries_codes)
  AND date IN UNNEST(@dates)
"""

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "timeseries_codes", "STRING", timeseries_codes
            ),
            bigquery.ArrayQueryParameter("dates", "DATE", dates),
        ]
    )

    prices = {}
    query_job = bqclient.query(query, job_config=job_config)
    results = query_job.result()
    for row in results:
        logger.debug(
            "code: %s, date: %s, value: %s",
            row.timeseries_code,
            row.date,
            row.value,
        )
        if row.timeseries_code not in prices:
            prices[row.timeseries_code] = {row.date: row.value}
        else:
            prices[row.timeseries_code][row.date] = row.value

    return prices


def number_of_moves(
    front_ff_future_px: float, back_ff_future_px: float
) -> float:
    """Computes the expected number of rate moves between two prices.

    Args:
      front_ff_future_px: Front fed funds future price.
      back_ff_future_px: Back fed funds future price.

    Returns:
      Number of moves.

    For calculation details see
    https://www.biancoresearch.com/bianco/samples/SR2v1.pdf

    """

    move_size_pct = MOVE_SIZE_BP / 100
    front_implied_rate = 100 - front_ff_future_px
    back_implied_rate = 100 - back_ff_future_px
    rate_delta = back_implied_rate - front_implied_rate
    num_moves = rate_delta / move_size_pct
    return num_moves


def fed_meeting_probabilities(nmoves: float) -> dict:
    move_text = "hike" if nmoves > 0 else "cut"
    if nmoves > 1:
        move_text = move_text + "s"

    max_expected_moves = math.ceil(abs(nmoves))
    max_expected_move_bp = max_expected_moves * MOVE_SIZE_BP
    move_odds = round(math.modf(abs(nmoves))[0], 2)

    output = {
        f"odds of {max_expected_move_bp}bp {move_text}": move_odds,
        f"odds of no {move_text}": round(1 - move_odds, 2),
    }

    return output


def compute_probabilities(meeting_date_str: str) -> dict:
    """Computes the probabilities of a rate move for a specific date.

    Args:
      meeting_date_str: Date of the Fed meeting.

    Returns:
      Dictionary of probabilities.
    """
    meeting_date = datetime.date.fromisoformat(meeting_date_str)
    meeting_date_day_before = meeting_date - datetime.timedelta(days=1)
    timeseries_codes = [x.strip() for x in TIMESERIES_CODES.split(",")]

    prices = fetch_prices_from_bq(
        timeseries_codes, [meeting_date, meeting_date_day_before]
    )

    error = None
    for code in timeseries_codes:
        if code not in prices:
            error = f"No data for {code}"
            break
        elif meeting_date not in prices[code]:
            error = f"No data for {code} on {meeting_date}"
            break
        elif meeting_date_day_before not in prices[code]:
            error = f"No data for {code} on {meeting_date_day_before}"
            break

    logger.debug("compute_probabilities: found prices: %s", prices)

    if error:
        return {"status": "ERROR", "message": error}

    near_code = timeseries_codes[0]
    far_code = timeseries_codes[1]
    num_moves_post = number_of_moves(
        prices[near_code][meeting_date], prices[far_code][meeting_date]
    )
    num_moves_pre = number_of_moves(
        prices[near_code][meeting_date_day_before],
        prices[far_code][meeting_date_day_before],
    )

    probs_pre = fed_meeting_probabilities(num_moves_pre)
    probs_post = fed_meeting_probabilities(num_moves_post)

    output = {
        (
            "Odds of a rate move within the next year ",
            "(computed before Fed meeting):",
        ): (probs_pre),
        (
            "Odds of a rate move within the next year ",
            "(computed after Fed meeting)",
        ): (probs_post),
    }

    return {"status": "OK", "output": output}


def main(argv: Sequence[str]) -> None:
    if len(argv) > 2:
        raise app.UsageError("Too many command-line arguments.")

    meeting_date = argv[1]
    print("meeting_date: ", meeting_date)

    print(compute_probabilities(meeting_date))


if __name__ == "__main__":
    app.run(main)
