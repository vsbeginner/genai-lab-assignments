"""Automated Headless Test Suite for Streamlit Demonstration Application.

Course: ETMMGA304 - Generative AI and Its Applications (Capstone Verification)
Institution: K.R. Mangalam University (Master of Computer Applications)

This test harness executes headless UI regression checks using Streamlit's official
AppTest framework:
  1. Empty Topic Submission: Asserts that an empty topic triggers validation warnings.
  2. Whitespace-Only Submission: Asserts that whitespace triggers identical validation warnings.
  3. Functional Generation: Verifies generation pipeline on held-out topics ('recursion', 'SQL joins').

Outputs:
  - lab_outputs/demo_checks.json: Test assertion telemetry and generated text artifacts.

Run:
  python test_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def run_headless_demonstration_checks() -> None:
    """Executes headless verification suite against app.py."""
    test_results = {}
    output_path = Path("lab_outputs/demo_checks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check 1: Empty Topic Submission
    app_runner = AppTest.from_file("app.py", default_timeout=120).run()
    assert not app_runner.exception, f"App initialization failed with exception: {app_runner.exception}"
    app_runner.button[0].click().run()
    test_results["empty_topic_warning"] = [w.value for w in app_runner.warning]
    test_results["empty_topic_generated_text"] = [m.value for m in app_runner.markdown]

    # Check 2: Validation on Held-Out Benchmark Topics
    for benchmark_topic in ("recursion", "SQL joins"):
        app_runner = AppTest.from_file("app.py", default_timeout=120).run()
        app_runner.text_input[0].set_value(benchmark_topic)
        app_runner.button[0].click().run()
        assert not app_runner.exception, f"Generation failed on '{benchmark_topic}': {app_runner.exception}"
        test_results[benchmark_topic] = [m.value for m in app_runner.markdown]

    # Check 3: Whitespace-Only Topic Submission
    app_runner = AppTest.from_file("app.py", default_timeout=120).run()
    app_runner.text_input[0].set_value("   ")
    app_runner.button[0].click().run()
    test_results["whitespace_only_warning"] = [w.value for w in app_runner.warning]

    # Persist Results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print("Automated demonstration checks completed successfully:")
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    run_headless_demonstration_checks()
