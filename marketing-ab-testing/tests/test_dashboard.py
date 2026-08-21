"""
Smoke test for the dashboard using Streamlit's AppTest -- runs the app
headlessly and checks it doesn't crash, and that the key numbers are present.
"""

from streamlit.testing.v1 import AppTest


def test_dashboard_runs_without_error():
    at = AppTest.from_file("../dashboard/streamlit_app.py")
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised an exception: {at.exception}"


def test_dashboard_shows_key_metrics():
    at = AppTest.from_file("../dashboard/streamlit_app.py")
    at.run(timeout=30)

    metric_labels = [m.label for m in at.metric]
    for expected in ["p-value", "Effect size (Cohen's h)", "ROI"]:
        assert expected in metric_labels, f"Missing expected metric: {expected}"


if __name__ == "__main__":
    test_dashboard_runs_without_error()
    print("test_dashboard_runs_without_error: passed")
    test_dashboard_shows_key_metrics()
    print("test_dashboard_shows_key_metrics: passed")
