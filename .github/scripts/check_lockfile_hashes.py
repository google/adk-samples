"""
Checks that every wheel and sdist distribution in a uv.lock file has a hash.

Usage: python check_lockfile_hashes.py <path-to-uv.lock>

Exits 0 if all distributions have hashes, 1 otherwise.
"""

import sys

import tomllib

lockfile_path = sys.argv[1]
with open(lockfile_path, "rb") as f:
    data = tomllib.load(f)

missing = []
for pkg in data.get("package", []):
    name = pkg.get("name", "<unknown>")
    version = pkg.get("version", "")
    source = pkg.get("source", {})
    if source.get("virtual"):
        continue  # workspace virtual package, no hash needed
    for dist in pkg.get("wheels", []) + (
        [pkg["sdist"]] if "sdist" in pkg else []
    ):
        if not dist.get("hash"):
            missing.append(
                f"  {name}=={version}: {dist.get('url', dist.get('path', '?'))}"
            )

if missing:
    print(f"[FAIL] {lockfile_path} — distributions missing hashes:")
    print("\n".join(missing))
    sys.exit(1)
