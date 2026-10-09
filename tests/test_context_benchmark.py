"""Ensure the five paid context tasks have real, behavior/scope-based grading."""

import pytest

from scripts.benchmark_checks import source_snapshot
from scripts.benchmark_token_usage import build_sandbox, verify_task
from scripts.context_benchmark_tasks import TASKS_CONTEXT

REPLACEMENTS = {
    "context_symbol_validation": (
        '        return {"valid": "alice"}.get(token)',
        '        if not token:\n            raise ValueError("empty token")\n        return {"valid": "alice"}.get(token)',
    ),
    "context_callers_rounding": (
        "return apply_discount(sum(prices), discount)",
        "return round(apply_discount(sum(prices), discount), 2)",
    ),
    "context_dependency_discount": (
        "    return amount * (1 - discount)",
        '    if not 0 <= discount <= 1:\n        raise ValueError("invalid discount")\n    return amount * (1 - discount)',
    ),
    "context_summary_doctest": (
        '"""Return whether a role has administrator privileges."""',
        '"""Return whether a role has administrator privileges.\n\n    >>> can_administer("admin")\n    True\n    """',
    ),
    "context_config_alignment": (":8000", ":8080"),
}


@pytest.mark.parametrize("task", TASKS_CONTEXT, ids=lambda t: t["id"])
def test_context_task_baseline_fails_and_correct_edit_passes(tmp_path, task):
    root = build_sandbox(tmp_path, "context")
    before = source_snapshot(root)
    assert not verify_task(task, root, before)["edit_applied"]
    path = root / task["check"]["file"]
    old, new = REPLACEMENTS[task["id"]]
    path.write_text(path.read_text().replace(old, new), encoding="utf-8")
    assert verify_task(task, root, before)["edit_applied"]
    (root / "unrelated.py").write_text("value = 1\n")
    assert not verify_task(task, root, before)["edit_applied"]


def test_representative_context_tools_find_fixture_evidence(tmp_path):
    from gigacode.navigation_index import NavigationIndex

    root = build_sandbox(tmp_path, "context")
    index = NavigationIndex(root)
    index.refresh()
    assert index.select("AuthService.authenticate")
    assert index.relations(index.select("invoice_total")[0], "callers")["items"][0]["symbol"] == "checkout"
    assert index.dependency_rows("billing.py")[0]["files"] == ["pricing.py"]
    assert index.summary("auth.py")["used_by"]["total"] == 2
    assert index.select("service.port", "deployment/values.yaml")
    assert index.task_context("authentication role policy")["relevant_tests"]
