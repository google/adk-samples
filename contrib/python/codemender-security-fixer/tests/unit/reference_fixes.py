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
"""Reference fixes for the bundled sample target, used by the unit tests.

They double as proof that every vulnerability in the sample is fixable within
the validator's rules.
"""

DB_FIX = '''
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer TEXT NOT NULL,
    status TEXT NOT NULL,
    total_cents INTEGER NOT NULL
);
"""

SEED_ORDERS = [
    ("ORD-1001", "acme", "shipped", 129900),
    ("ORD-1002", "acme", "pending", 4500),
    ("ORD-1003", "globex", "shipped", 78000),
    ("ORD-1004", "initech", "cancelled", 1999),
]


def connect(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?)", SEED_ORDERS)
    conn.commit()
    return conn


def find_orders(conn, customer, status=None):
    query = "SELECT * FROM orders WHERE customer = ?"
    params = [customer]
    if status:
        query += " AND status = ?"
        params.append(status)
    return [dict(row) for row in conn.execute(query, params)]
'''

# Parameterized, but drops the status filter: fixes the injection and
# breaks a functional test.
DB_BROKEN = DB_FIX.replace(
    """    if status:
        query += " AND status = ?"
        params.append(status)
""",
    "",
)

INVOICES_FIX = """
import os
import re
import subprocess

_MONTH = re.compile(r"\\d{4}-\\d{2}")


def read_invoice(invoice_dir, name):
    base = os.path.realpath(invoice_dir)
    path = os.path.realpath(os.path.join(base, name))
    if os.path.commonpath([base, path]) != base:
        raise ValueError("invoice path escapes the invoice directory")
    with open(path, "rb") as f:
        return f.read()


def archive_invoices(invoice_dir, month, out_dir):
    if not _MONTH.fullmatch(month):
        raise ValueError("month must be YYYY-MM")
    archive = os.path.join(out_dir, f"invoices-{month}.tar.gz")
    subprocess.run(["tar", "-czf", archive, "-C", invoice_dir, month], check=True)
    return archive
"""

# The classic incomplete fix: strip "../" once. "....//" survives it, absolute
# paths are untouched, and the shell injection is still there.
INVOICES_INCOMPLETE = """
import os
import subprocess


def read_invoice(invoice_dir, name):
    path = os.path.join(invoice_dir, name.replace("../", ""))
    with open(path, "rb") as f:
        return f.read()


def archive_invoices(invoice_dir, month, out_dir):
    archive = os.path.join(out_dir, f"invoices-{month}.tar.gz")
    subprocess.run(
        f"tar -czf {archive} -C {invoice_dir} {month}", shell=True, check=True
    )
    return archive
"""

CART_FIX = """
import base64
import json


def dump_cart(cart):
    return base64.b64encode(json.dumps(cart).encode()).decode("ascii")


def load_cart(cookie):
    cart = json.loads(base64.b64decode(cookie))
    if not isinstance(cart, dict) or not all(
        isinstance(k, str) and isinstance(v, int) for k, v in cart.items()
    ):
        raise ValueError("cart must map SKUs to quantities")
    return cart
"""

AUTH_FIX = """
import hmac
import os


def is_admin(authorization_header):
    token = os.environ.get("ORDERS_API_TOKEN")
    if not token or not authorization_header:
        return False
    return hmac.compare_digest(
        authorization_header.encode(), f"Bearer {token}".encode()
    )
"""

ALL_FIXES = {
    "orders/db.py": DB_FIX,
    "orders/invoices.py": INVOICES_FIX,
    "orders/cart.py": CART_FIX,
    "orders/auth.py": AUTH_FIX,
}
