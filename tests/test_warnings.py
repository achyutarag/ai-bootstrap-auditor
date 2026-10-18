"""Low-sample warning band (reviewer XfWC-W4) and retest tracking (XfWC-W2)."""
import numpy as np
import pytest

from ai_bootstrap_auditor import (AIBootstrapAuditor, GuardrailAuditor, RetestTracker,
                                  LOW_SAMPLE_WARNING_N, MIN_PAIRED_OBSERVATIONS,
                                  low_sample_warning, retest_warning)


@pytest.mark.parametrize("n, expect_warning", [(10, True), (25, True), (49, True), (50, False), (500, False)])
def test_low_sample_warning_band(n, expect_warning):
    # continuous data, so only the n-based band applies (binary data can also
    # trigger the separate discordant-pair note; see test_exact_guard.py)
    r = np.random.RandomState(n)
    result = AIBootstrapAuditor("m").audit(r.normal(0.70, 0.1, n), r.normal(0.75, 0.1, n),
                                           n_iterations=300, seed=1)
    assert (result["low_sample_warning"] is not None) == expect_warning


def test_low_sample_warning_quotes_measured_coverage():
    assert "87%" in low_sample_warning(10)      # worst of (87.1, 87.5) at the n=10 grid point
    assert "n=20" in low_sample_warning(29)     # nearest calibrated size at or below n
    assert low_sample_warning(MIN_PAIRED_OBSERVATIONS - 1) is None   # refused, not warned
    assert low_sample_warning(LOW_SAMPLE_WARNING_N) is None


def test_guardrail_carries_low_sample_warning():
    r = np.random.RandomState(0)
    result = GuardrailAuditor("latency").audit(r.normal(100, 10, 20), r.normal(101, 10, 20), seed=1)
    assert result["low_sample_warning"] is not None


def test_retest_tracker_counts_distinct_datasets_only():
    tracker = RetestTracker()
    b, a = np.ones(20), np.zeros(20)
    assert tracker.record("acc", b, a) == 1
    assert tracker.record("acc", b.copy(), a.copy()) == 1   # same data again (UI rerun / re-roll)
    assert tracker.record("acc", np.r_[b, 1], np.r_[a, 0]) == 2   # more data appended -> new look
    assert tracker.record("latency", b, a) == 1               # separate comparison
    tracker.reset("acc")
    assert tracker.looks("acc") == 0 and tracker.looks("latency") == 1


def test_retest_warning_uses_measured_inflation():
    assert retest_warning(1) is None
    assert "2.8%" in retest_warning(2)
    assert "4.9%" in retest_warning(7)    # nearest measured grid point at or below 7 looks is 5
    assert "8.6%" in retest_warning(25)
