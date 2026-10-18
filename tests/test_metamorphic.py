"""Metamorphic invariants: relations between runs that must hold no matter
what the right answer is. These need no ground truth, so they catch defects
that calibration against synthetic ground truth can miss (e.g. Findings 2
and 4 in the paper)."""
import numpy as np
import pytest

from ai_bootstrap_auditor import AIBootstrapAuditor, AuditBatch
from conftest import null_metrics

B = 2000
METHODS = ("bonferroni", "benjamini_hochberg")


def _datasets(n_sets=40, n=150):
    """Mix of regressions, nulls, borderline and clear wins."""
    for s in range(n_sets):
        r = np.random.RandomState(1000 + s)
        p_after = 0.70 + [-0.10, -0.05, 0.0, 0.05, 0.10][s % 5]
        yield r.binomial(1, 0.70, n), r.binomial(1, p_after, n)


@pytest.mark.parametrize("method", METHODS)
def test_batch_of_one_agrees_with_single_audit(method):
    mismatches = 0
    for before, after in _datasets():
        single = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=7)
        batch = AuditBatch(correction_method=method)
        batch.add_metric("m", before, after)
        batched = batch.run(n_iterations=B, verbose=False, seed=7)[0]
        mismatches += single["status"] != batched["status"]
    assert mismatches == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k_null", [1, 5, 19])
def test_adding_null_metrics_never_demotes_a_no_go(method, k_null):
    for before, after in _datasets():
        single = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=7)
        if single["status"] != "NO-GO":
            continue
        batch = AuditBatch(correction_method=method)
        batch.add_metric("m", before, after)
        for j, (b, a) in enumerate(null_metrics(k_null, n=len(before))):
            batch.add_metric(f"null_{j}", b, a)
        assert batch.run(n_iterations=B, verbose=False, seed=7)[0]["status"] == "NO-GO"


@pytest.mark.parametrize("method", METHODS)
def test_correction_never_adds_a_go(method):
    for before, after in _datasets():
        single = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=7)
        batch = AuditBatch(correction_method=method)
        batch.add_metric("m", before, after)
        for j, (b, a) in enumerate(null_metrics(4, n=len(before))):
            batch.add_metric(f"null_{j}", b, a)
        if batch.run(n_iterations=B, verbose=False, seed=7)[0]["status"] == "GO":
            assert single["status"] == "GO"


def test_goal_mirror_symmetry():
    """Negating the data and flipping the goal must mirror the verdict."""
    for before, after in _datasets():
        up = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=3)
        down = AIBootstrapAuditor("m", "decrease").audit(-before, -after, n_iterations=B, seed=3)
        assert up["status"] == down["status"]
        assert up["ci_lower"] == pytest.approx(-down["ci_upper"])
        assert up["ci_upper"] == pytest.approx(-down["ci_lower"])


@pytest.mark.parametrize("scale, shift", [(1000.0, 0.0), (0.001, 5.0), (3.7, -12.0)])
def test_positive_affine_invariance(scale, shift):
    """Changing units (ms -> s, 0-1 -> 0-100) must not change the verdict."""
    for before, after in _datasets(n_sets=15):
        base = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=3)
        moved = AIBootstrapAuditor("m", "increase").audit(before * scale + shift, after * scale + shift,
                                                          n_iterations=B, seed=3)
        assert base["status"] == moved["status"]
        assert moved["ci_lower"] == pytest.approx(base["ci_lower"] * scale, rel=1e-9, abs=1e-12)


def test_display_string_agrees_with_status():
    prefixes = {"GO": "GO (", "NO-GO": "NO-GO (", "REVIEW": "REVIEW / RETEST"}
    for before, after in _datasets():
        for goal in ("increase", "decrease"):
            result = AIBootstrapAuditor("m", goal).audit(before, after, n_iterations=B, seed=3)
            assert result["verdict"].startswith(prefixes[result["status"]])
