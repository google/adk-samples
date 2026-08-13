"""THROWAWAY probe used to validate the AI PR review workflows.

Every function below contains a deliberate, obvious defect. This file
exists only so the correctness, security and maintainability reviewers
have something unambiguous to comment on, and is deleted as soon as the
migration off gemini-cli is confirmed. Do not import it from anywhere.
"""

import hashlib
import pickle
import sqlite3
import subprocess

# Security: hardcoded credential committed in source.
ALPHAVANTAGE_API_KEY = "AKIA5JQ7XGV2QW8NPLZ4"
DB_PASSWORD = "sup3rs3cret-prod-password"


def fetch_quote(ticker: str) -> str:
    """Security: unsanitised input interpolated into a shell command."""
    cmd = f"curl -s https://api.example.com/quote?symbol={ticker}"
    return subprocess.check_output(cmd, shell=True).decode()


def load_portfolio(blob: bytes) -> dict:
    """Security: unsafe deserialisation of untrusted bytes."""
    return pickle.loads(blob)


def find_holding(conn: sqlite3.Connection, account: str) -> list:
    """Security: SQL injection via string concatenation."""
    query = "SELECT * FROM holdings WHERE account = '" + account + "'"
    return conn.execute(query).fetchall()


def hash_password(password: str) -> str:
    """Security: MD5 is unsuitable for password hashing."""
    return hashlib.md5(password.encode()).hexdigest()


def average_return(returns: list[float]) -> float:
    """Correctness: off-by-one truncates the final element, and an
    empty list raises ZeroDivisionError instead of being handled."""
    total = 0.0
    for i in range(len(returns) - 1):
        total += returns[i]
    return total / len(returns)


def annual_fee(balance: float, rate_percent: float) -> float:
    """Correctness: `rate_percent` is a percentage but is applied as a
    raw fraction, overcharging every caller by a factor of 100."""
    return balance * rate_percent
