"""Regression tests for each defect reported in the paper. Each test checks
the fix AND that the historical code (legacy/) fails the same check, so the
test is known to exercise the defect rather than only the fixed path."""
import numpy as np
import pytest

from ai_bootstrap_auditor import (AIBootstrapAuditor, AuditBatch, GuardrailAuditor,
                                  HarnessCalibrationSuite, DataIntegrityError,
                                  MinimumSampleFloorError, PairingError,
                                  clopper_pearson, exact_binomial_test)
from conftest import null_metrics

B = 4000


# --- Finding 1: "GO" in "NO-GO" (paper Section 5) ---------------------------

def test_substring_bug_pattern_misclassifies_no_go(regression_pair):
    before, after = regression_pair
    result = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)
    assert result["status"] == "NO-GO"
    assert ("GO" in result["verdict"]) is True      # historical bug pattern accepts it
    assert (result["status"] == "GO") is False      # structured-field check rejects it


# --- Finding 2: Bonferroni demoted a confident regression (paper Section 3.1) -

def _bonferroni_batch(module, before, after, k_null=14):
    batch = module.AuditBatch(correction_method="bonferroni")
    batch.add_metric("target", before, after)
    for j, (b, a) in enumerate(null_metrics(k_null)):
        batch.add_metric(f"null_{j}", b, a)
    return batch.run(n_iterations=B, verbose=False, seed=1)[0]


def test_bonferroni_preserves_raw_no_go(regression_pair):
    before, after = regression_pair
    result = _bonferroni_batch(__import__("ai_bootstrap_auditor"), before, after)
    assert result["raw_status"] == "NO-GO"
    assert result["status"] == "NO-GO"
    # the simultaneous (widened) interval is still the one reported
    assert result["ci_lower"] < 0 < result["ci_upper"]


def test_legacy_bonferroni_demoted_no_go(legacy, regression_pair):
    before, after = regression_pair
    assert _bonferroni_batch(legacy, before, after)["status"] == "REVIEW"


# --- Finding 4: BH path ran at alpha while single audit runs at alpha/2 ------

def _bh_single(module, before, after):
    batch = module.AuditBatch(correction_method="benjamini_hochberg")
    batch.add_metric("m", before, after)
    return batch.run(n_iterations=B, verbose=False, seed=1)[0]


def _borderline_pairs(n_pairs=150):
    for s in range(n_pairs):
        r = np.random.RandomState(s)
        yield r.binomial(1, 0.70, 200), r.binomial(1, 0.76, 200)


def test_legacy_bh_batch_of_one_was_more_liberal_than_single_audit(legacy):
    extra_gos = 0
    for before, after in _borderline_pairs():
        single = legacy.AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)
        if single["status"] == "REVIEW" and _bh_single(legacy, before, after)["status"] == "GO":
            extra_gos += 1
    assert extra_gos > 0


def test_bh_never_adds_a_go_the_single_audit_did_not_make():
    for before, after in _borderline_pairs():
        single = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)
        if _bh_single(__import__("ai_bootstrap_auditor"), before, after)["status"] == "GO":
            assert single["status"] == "GO"


def test_p_value_is_never_zero():
    before = np.zeros(50)
    after = np.ones(50)
    result = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=999, seed=1)
    assert result["p_value"] == pytest.approx(1 / 1000)


# --- Calibration check itself must be able to fail ---------------------------

def test_calibration_target_is_one_sided():
    suite = HarnessCalibrationSuite()
    result = suite.null_calibration(n_questions=50, n_trials=20, seed=0)
    assert result["expected_false_positive_rate"] == pytest.approx(0.025)


def test_calibration_check_detects_doubled_false_positive_rate():
    # 100/2000 = 5% observed against a 2.5% target: the TAE-2026 tolerance
    # (|obs - exp| < 0.05 vs a 5% target) would have called this calibrated.
    report = HarnessCalibrationSuite()._rate_report(100, 2000, 0.025)
    assert report["well_calibrated"] is False
    assert report["ci_low"] > 0.025


def test_exact_binomial_helpers_match_reference_values():
    lo, hi = clopper_pearson(5, 200, confidence=0.95)
    assert lo == pytest.approx(0.008166, abs=1e-5)
    assert hi == pytest.approx(0.057374, abs=1e-5)
    assert exact_binomial_test(5, 200, 0.025) == pytest.approx(1.0)


# --- Input validation ---------------------------------------------------------

@pytest.mark.parametrize("before, after, error", [
    (np.array([]), np.array([]), MinimumSampleFloorError),
    (np.array([0.7]), np.array([0.8]), MinimumSampleFloorError),
    (np.ones(9), np.ones(9), MinimumSampleFloorError),
    (np.array([0.7, np.nan, 0.72] * 4), np.array([0.8, 0.79, 0.81] * 4), DataIntegrityError),
    (np.ones(20), np.ones(21), PairingError),
])
def test_audit_refuses_bad_input(before, after, error):
    with pytest.raises(error):
        AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=200)


def test_guardrail_below_floor_is_inconclusive_not_pass():
    result = GuardrailAuditor("latency").audit(np.ones(5), np.ones(5) * 2, seed=1)
    assert result["status"] == "INCONCLUSIVE"
