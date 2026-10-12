"""CLI parity: scripts/onboard_products.py dispatches to the SAME functions the
Campaign agent's tools use, with the same arguments — no logic drift."""


def test_create_dispatch(monkeypatch, capsys):
    import scripts.onboard_products as cli

    calls = {}

    def fake_create(**kwargs):
        calls.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(cli, "create_product", fake_create)
    rc = cli.main(
        [
            "create",
            "--name",
            "Aurora Cold Brew 330ml",
            "--category",
            "beverage",
            "--description",
            "nitro",
            "--attr",
            "volume_ml=330",
            "--attr",
            "caffeine_mg=120",
        ]
    )
    assert rc == 0
    assert calls == {
        "name": "Aurora Cold Brew 330ml",
        "category": "beverage",
        "description": "nitro",
        "attributes": {"volume_ml": "330", "caffeine_mg": "120"},
    }
    assert '"status": "success"' in capsys.readouterr().out


def test_import_folder_dispatch(monkeypatch):
    import scripts.onboard_products as cli

    calls = {}
    monkeypatch.setattr(
        cli,
        "import_products_from_folder",
        lambda **kw: calls.update(kw) or {"status": "success"},
    )
    rc = cli.main(["import-folder", "/tmp/imgs", "--category", "homeware"])
    assert rc == 0
    assert calls == {"folder_path": "/tmp/imgs", "category": "homeware"}


def test_generate_image_dispatch(monkeypatch):
    import scripts.onboard_products as cli

    calls = {}

    async def fake_generate(**kwargs):
        calls.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(cli, "generate_product_image", fake_generate)
    rc = cli.main(
        [
            "generate-image",
            "--product-id",
            "3",
            "--style-hint",
            "warm morning light",
        ]
    )
    assert rc == 0
    assert calls == {
        "product_id": 3,
        "product_name": "",
        "style_hint": "warm morning light",
    }


def test_error_result_returns_nonzero(monkeypatch):
    import scripts.onboard_products as cli

    monkeypatch.setattr(
        cli,
        "create_product",
        lambda **kw: {"status": "error", "message": "nope"},
    )
    assert cli.main(["create", "--name", "X"]) == 1
