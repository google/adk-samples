# Copyright 2026 Attenu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""The recipe's tools, over a small in-memory order book.

Every tool appends to `EXECUTED` as its first statement. That list is
how the demo and the tests show that a refused call was refused *before*
its body ran, rather than run and then reported. Nothing else in the
recipe depends on it; delete it when you adapt this to your own tools.
"""

from typing import Any

# (tool_name, notable_argument) for every tool body that actually ran.
EXECUTED: list[tuple[str, Any]] = []


def reset() -> None:
    """Clear the execution record between runs."""
    EXECUTED.clear()


_ORDERS = {
    "ORD-8812": {
        "customer": "Ada Ellis",
        "email": "ada@example.com",
        "invoice_id": "INV-4471",
        "total_cents": 48000,
        "status": "delivered",
    }
}

_INVOICES = {
    "INV-4471": {
        "order_id": "ORD-8812",
        "total_cents": 48000,
        "paid": True,
        "issued": "2026-07-02",
    }
}


def lookup_order(order_id: str) -> dict:
    """Look up one order by its identifier.

    Args:
        order_id: the order reference, e.g. "ORD-8812".
    """
    EXECUTED.append(("lookup_order", order_id))
    return _ORDERS.get(order_id, {"error": "unknown order"})


def get_invoice(invoice_id: str) -> dict:
    """Read one invoice.

    Args:
        invoice_id: the invoice reference, e.g. "INV-4471".
    """
    EXECUTED.append(("get_invoice", invoice_id))
    return _INVOICES.get(invoice_id, {"error": "unknown invoice"})


def issue_refund(invoice_id: str, amount_cents: int) -> dict:
    """Move money back to the customer. Irreversible.

    Args:
        invoice_id: the invoice to refund.
        amount_cents: how much to refund, in cents.
    """
    EXECUTED.append(("issue_refund", (invoice_id, amount_cents)))
    return {"refunded": amount_cents, "invoice_id": invoice_id}


def email_customer(to: str, body: str) -> dict:
    """Send an email to the customer.

    Args:
        to: recipient address.
        body: message body.
    """
    EXECUTED.append(("email_customer", to))
    return {"sent_to": to}
