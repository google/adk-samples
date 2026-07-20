"""
Checks that every wheel and sdist distribution in a uv.lock file has a hash
in a recognised format.

Usage: python check_lockfile_hashes.py <path-to-uv.lock>

Exit codes:
  0  all distributions have valid hashes
  1  one or more distributions are missing or have a malformed hash
  2  usage error (wrong number of arguments)
"""

import re
import sys

import tomllib

# Hashes produced by uv are always sha256; sha384 and sha512 are accepted
# defensively in case a future uv release adds stronger algorithms.
_HASH_RE = re.compile(r"^(sha256|sha384|sha512):[0-9a-f]+$")

# Source kinds that carry no downloadable artifact hash. Virtual workspace
# roots have no artifact at all. Path, editable, directory, and git deps are
# rejected by upstream pipeline checks (Check 3 / Check 2) before this script
# runs, so skipping them here is primarily defence-in-depth: the script stays
# correct even if the pipeline order ever changes.
_NO_HASH_SOURCE_KEYS = frozenset(
    {"virtual", "path", "editable", "directory", "git"}
)


def _check_package(
    pkg: dict,
    missing: list[str],
    bad_hash: list[str],
) -> None:
    """Validate hashes for all distributions in one lockfile package entry."""
    source = pkg.get("source", {})
    if any(key in source for key in _NO_HASH_SOURCE_KEYS):
        return

    name = pkg.get("name", "<unknown>")
    version = pkg.get("version", "")
    dists = pkg.get("wheels", []) + ([pkg["sdist"]] if "sdist" in pkg else [])

    for dist in dists:
        url_or_path = dist.get("url", dist.get("path", "?"))
        hash_val = dist.get("hash", "")
        if not hash_val:
            missing.append(f"  {name}=={version}: {url_or_path}")
        elif not _HASH_RE.match(hash_val):
            bad_hash.append(f"  {name}=={version}: malformed hash {hash_val!r}")


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: check_lockfile_hashes.py <path-to-uv.lock>",
            file=sys.stderr,
        )
        sys.exit(2)

    lockfile_path = sys.argv[1]

    try:
        with open(lockfile_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        print(f"[FAIL] {lockfile_path}: file not found", file=sys.stderr)
        sys.exit(1)
    except tomllib.TOMLDecodeError as exc:
        print(
            f"[FAIL] {lockfile_path} is not valid TOML: {exc}", file=sys.stderr
        )
        sys.exit(1)

    missing: list[str] = []
    bad_hash: list[str] = []

    for pkg in data.get("package", []):
        _check_package(pkg, missing, bad_hash)

    if missing or bad_hash:
        print(f"[FAIL] {lockfile_path} — hash issues:")
        for entry in missing:
            print(entry)
        for entry in bad_hash:
            print(entry)
        sys.exit(1)


if __name__ == "__main__":
    main()
