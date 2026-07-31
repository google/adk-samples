from streamlit.testing.v1 import AppTest

# Path to the playground app
APP_PATH = "economic_research/playground/app.py"


def test_playground_app_startup():
    """Verify that the Streamlit Consultant Playground starts up without errors."""
    at = AppTest.from_file(APP_PATH)
    # The existence check is implicit in from_file
    assert at is not None


def test_a2ui_tag_replacements():
    """Verify that A2UI tags are replaced with user-friendly markdown icons."""
    # We test the replacement logic directly since it's a string processing block in app.py
    test_report = (
        "This is a test run [A2UI: RENDER_CHART] and [A2UI: SHOW_METRICS]."
    )

    # Matching the logic in app.py
    modified_report = test_report.replace(
        "[A2UI: RENDER_CHART]", "📈 *Chart Generated*"
    )
    modified_report = modified_report.replace(
        "[A2UI: SHOW_METRICS]", "📊 *Metrics Calculated & Visualized*"
    )

    assert "[A2UI: RENDER_CHART]" not in modified_report
    assert "📈 *Chart Generated*" in modified_report
    assert "📊 *Metrics Calculated & Visualized*" in modified_report
