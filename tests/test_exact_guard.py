"""Finding 6: for yes/no metrics with few discordant pairs, the percentile
bootstrap was overconfident (GO where the exact sign/McNemar test is not
significant). GO/NO-GO on dichotomous data must now also pass the exact test."""
from math import comb

import numpy as np
import pytest

from ai_bootstrap_auditor import (AIBootstrapAuditor, AuditBatch, MIN_DISCORDANT_PAIRS,
                                  sign_test_upper_tail)

B = 20000


def discordant_data(up, down, n=100):
    """n paired yes/no outcomes: `up` questions v0.1 wrong -> v0.2 right,
    `down` questions right -> wrong, the rest tied (both right)."""
    before, after = np.ones(n), np.ones(n)
    before[:up] = 0
    after[up:up + down] = 0
    return before, after


def exact_p(k_up, k):
    return sum(comb(k, i) for i in range(k_up, k + 1)) / 2 ** k


@pytest.mark.parametrize("up, down", [(4, 0), (5, 0), (7, 1)])
def test_few_discordant_pairs_no_longer_give_go(up, down, legacy):
    before, after = discordant_data(up, down)
    assert legacy.AIBootstrapAuditor("m").audit(before, after, n_iterations=B, seed=1)["status"] == "GO"
    result = AIBootstrapAuditor("m").audit(before, after, n_iterations=B, seed=1)
    assert result["status"] == "REVIEW"
    assert result["exact_guard_applied"] is True
    assert result["exact_sign_test_p"] == pytest.approx(exact_p(up, up + down))


@pytest.mark.parametrize("up, down", [(6, 0), (9, 1), (10, 2)])
def test_go_kept_when_exact_test_agrees(up, down):
    result = AIBootstrapAuditor("m").audit(*discordant_data(up, down), n_iterations=B, seed=1)
    assert exact_p(up, up + down) <= 0.025
    assert result["status"] == "GO" and result["exact_guard_applied"] is False


def test_guard_covers_no_go_and_decrease_goal():
    before, after = discordant_data(0, 4)            # 4 regressions, 0 improvements
    assert AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)["status"] == "REVIEW"
    # mirrored: for a 'decrease' metric, the same 4 changes are improvements
    assert AIBootstrapAuditor("m", "decrease").audit(before, after, n_iterations=B, seed=1)["status"] == "REVIEW"
    before, after = discordant_data(0, 8)
    assert AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)["status"] == "NO-GO"


def test_all_ties_is_flagged_as_no_information():
    result = AIBootstrapAuditor("m").audit(np.ones(100), np.ones(100), n_iterations=2000, seed=1)
    assert result["status"] == "REVIEW"
    assert result["discordant_pairs"] == 0
    assert "no information" in result["low_sample_warning"]


def test_thin_evidence_note_below_threshold_only():
    few = AIBootstrapAuditor("m").audit(*discordant_data(10, 5, n=200), n_iterations=2000, seed=1)
    many = AIBootstrapAuditor("m").audit(*discordant_data(15, 10, n=200), n_iterations=2000, seed=1)
    assert "changed outcome" in few["low_sample_warning"]
    assert many["low_sample_warning"] is None
    assert MIN_DISCORDANT_PAIRS == 20


def test_continuous_metrics_are_untouched():
    r = np.random.RandomState(0)
    result = AIBootstrapAuditor("latency", "decrease").audit(r.normal(100, 10, 200), r.normal(97, 10, 200),
                                                             n_iterations=2000, seed=1)
    assert result["discordant_pairs"] is None and result["exact_guard_applied"] is False


@pytest.mark.parametrize("method", ["bonferroni", "benjamini_hochberg"])
def test_batches_apply_the_same_guard(method):
    batch = AuditBatch(correction_method=method)
    batch.add_metric("few_changes", *discordant_data(5, 0))
    batch.add_metric("other", *discordant_data(30, 10, n=300))
    assert batch.run(n_iterations=B, verbose=False, seed=1)[0]["status"] == "REVIEW"


def test_sign_test_helper():
    assert sign_test_upper_tail(0, 0) == 1.0
    assert sign_test_upper_tail(5, 5) == pytest.approx(1 / 32)
    assert sign_test_upper_tail(14, 20) == pytest.approx(0.0577, abs=1e-4)
