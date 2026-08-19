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
"""The fixture layer: hardcoded data, no clock, no network.

Search gets the most attention here for one reason. A query that finds
nothing falls back to the shopper's profile, which looks like a working
recommendation -- so a broken matcher does not fail, it just quietly stops
using what the shopper asked for. ``trail shoes`` returning nothing against a
``trail-shoes`` category is exactly that failure, and it is the bug these
tests were written after finding.
"""

from __future__ import annotations

import pytest

from app import store
from app.profile import DEFAULT_PROFILE


def ids(query: str) -> list[str]:
    return [p["id"] for p in store.find_products(query)]


# --- search -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # The words a shopper types, against a slug in the data.
        ("trail shoes", "cirrus-trail-3"),
        ("trail-shoes", "cirrus-trail-3"),
        ("TRAIL SHOES", "cirrus-trail-3"),
        # Brand, name, and tag all reachable from one field-agnostic query.
        ("aera", "cirrus-trail-3"),
        ("merino crew", "loft-merino-crew"),
        ("waterproof shell", "haldenshell-ultra"),
    ],
)
def test_a_shopper_s_wording_reaches_the_product(
    query: str, expected: str
) -> None:
    assert expected in ids(query)


def test_every_token_has_to_match() -> None:
    """Otherwise "trail shoes" also returns every shoe in the catalog.

    Token-wise matching is loose enough to cross the slug boundary and strict
    enough that adding a word narrows the result instead of widening it.
    """
    shoes = set(ids("shoes"))
    trail = set(ids("trail shoes"))
    assert trail < shoes
    assert "quickstep-road-2" in shoes - trail


def test_a_token_matches_inside_a_word() -> None:
    """ "shoe" finds shoes; no stemmer required for a demo catalog."""
    assert ids("trail shoe") == ids("trail shoes")


@pytest.mark.parametrize("query", ["", "   ", "zzzz", "trail zzzz"])
def test_a_query_with_no_match_returns_nothing(query: str) -> None:
    """The empty result is what triggers the profile fallback in the tool."""
    assert store.find_products(query) == []


# --- the catalog ------------------------------------------------------------


def test_every_product_carries_what_the_ranking_reads() -> None:
    """A product missing a field would be silently unrankable."""
    for item in store.products():
        assert set(item) >= {
            "id",
            "name",
            "brand",
            "category",
            "price",
            "rating",
            "sizes",
            "colors",
            "tags",
            "attributes",
        }, item["id"]
        assert item["sizes"], f"{item['id']} is stocked in no size"


def test_the_default_profile_names_categories_that_exist() -> None:
    """A typo here would make "what should I get" return nothing.

    An empty query recommends from ``preferred_categories``, so a category
    with no products behind it is indistinguishable from an empty catalog.
    """
    stocked = {p["category"] for p in store.products()}
    assert set(DEFAULT_PROFILE["preferred_categories"]) <= stocked
    assert set(DEFAULT_PROFILE["favorite_brands"]) <= {
        p["brand"] for p in store.products()
    }


def test_product_ids_are_unique() -> None:
    all_ids = [p["id"] for p in store.products()]
    assert len(all_ids) == len(set(all_ids))


def test_an_unknown_id_is_none_not_an_error() -> None:
    assert store.product("no-such-sku") is None


def test_products_are_returned_as_copies() -> None:
    """The cache is shared, so a caller mutating a product would poison it."""
    first = store.product("cirrus-trail-3")
    assert first is not None
    first["price"] = 1.0
    assert store.product("cirrus-trail-3")["price"] != 1.0


# --- orders and the fixture clock -------------------------------------------


def test_orders_are_newest_first() -> None:
    placed = [o["placed_on"] for o in store.orders()]
    assert placed == sorted(placed, reverse=True)


def test_the_open_order_is_the_one_still_moving() -> None:
    open_order = store.latest_open_order()
    assert open_order is not None
    assert any(not s["reached"] for s in open_order["stages"])


def test_a_cancelled_order_does_not_count_as_owned() -> None:
    """The parcel never arrived, so the item is still recommendable."""
    cancelled = next(
        o
        for o in store.orders()
        if any(s.get("state") == "cancelled" for s in o["stages"])
    )
    owned = store.purchased_product_ids()
    assert cancelled["item_ids"]
    assert all(
        item_id not in owned
        for item_id in cancelled["item_ids"]
        if not any(
            item_id in other["item_ids"]
            for other in store.orders()
            if other["id"] != cancelled["id"]
        )
    )


def test_today_comes_from_the_fixture_not_the_clock() -> None:
    """A demo whose ETA drifts into the past reads as broken.

    Pinning "today" in the data is also what keeps the timeline tests from
    failing on a Tuesday.
    """
    assert store.today() == store._load("orders.json")["today"]
    assert store.today() >= max(o["placed_on"] for o in store.orders())


# --- spend ------------------------------------------------------------------


def test_spend_history_is_oldest_first_and_a_full_year() -> None:
    """The window is clamped to 12, so anything shorter would clamp short."""
    months = store.spend_months()
    assert len(months) == 12
    assert [m["month"] for m in months] == [
        m["month"] for m in sorted(months, key=lambda m: m["month"])
    ]
    assert all(isinstance(m["amount"], (int, float)) for m in months)


def test_the_currency_is_stated_in_the_data() -> None:
    assert store.currency() == "USD"
