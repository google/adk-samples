"""CLI for product onboarding — thin wrapper over the SAME functions the
Campaign agent's tools use (app/tools/onboarding_tools.py). No logic here
beyond argument parsing; parity is by construction and pinned by
tests/unit/test_onboard_products_cli.py.

Usage:
  python -m scripts.onboard_products create --name "Aurora Cold Brew 330ml" \
      --category beverage --description "nitro cold brew" --attr volume_ml=330
  python -m scripts.onboard_products import-folder /path/to/images --category homeware
  python -m scripts.onboard_products generate-image --product-id 3 --style-hint "warm light"

Environment: same as the app (app/.env). Leave GCS_BUCKET unset for
local-first mode; set DEMO_DATASET=none for an empty from-scratch catalog.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.onboarding_tools import (
    create_product,
    generate_product_image,
    import_products_from_folder,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Onboard products into the catalog"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create one product")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--category", default="")
    p_create.add_argument("--description", default="")
    p_create.add_argument(
        "--attr",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable vertical-specific attribute",
    )

    p_import = sub.add_parser(
        "import-folder", help="Bulk-import a folder of product images"
    )
    p_import.add_argument("folder_path")
    p_import.add_argument("--category", default="")

    p_gen = sub.add_parser(
        "generate-image", help="Generate a reference product photo"
    )
    p_gen.add_argument("--product-id", type=int, default=0)
    p_gen.add_argument("--product-name", default="")
    p_gen.add_argument("--style-hint", default="")

    args = parser.parse_args(argv)

    if args.command == "create":
        attributes = {}
        for pair in args.attr:
            key, _, value = pair.partition("=")
            if not key or not _:
                parser.error(f"--attr must be KEY=VALUE, got {pair!r}")
            attributes[key] = value
        result = create_product(
            name=args.name,
            category=args.category,
            description=args.description,
            attributes=attributes,
        )
    elif args.command == "import-folder":
        result = import_products_from_folder(
            folder_path=args.folder_path, category=args.category
        )
    else:  # generate-image
        result = asyncio.run(
            generate_product_image(
                product_id=args.product_id,
                product_name=args.product_name,
                style_hint=args.style_hint,
            )
        )

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
