"""Import validation tests for retail-virtual-tryon scripts."""


def test_import_setup_tryon():
    import setup_tryon
    assert hasattr(setup_tryon, "create_bucket_if_needed")
    assert hasattr(setup_tryon, "verify_gemini_image_api")
    assert hasattr(setup_tryon, "resolve_model_id")
    assert hasattr(setup_tryon, "setup")
    assert hasattr(setup_tryon, "load_config")
    assert hasattr(setup_tryon, "GEMINI_IMAGE_MODELS")
    assert hasattr(setup_tryon, "SAFETY_LEVELS")
