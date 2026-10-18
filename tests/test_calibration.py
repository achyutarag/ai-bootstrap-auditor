"""Long-run behavior checks (A/A false-positive rate, power vs a closed-form
oracle). These are statistical, so they use fixed seeds and exact binomial
bounds rather than ad-hoc tolerances."""
import numpy as np
import pytest

from ai_bootstrap_auditor import HarnessCalibrationSuite, _correlated_binomial_pair


def test_null_go_rate_matches_one_sided_target():
    suite = HarnessCalibrationSuite(n_bootstrap_iterations=500)
    result = suite.null_calibration(true_p=0.70, n_questions=200, n_trials=1500, seed=11)
    assert result["well_calibrated"], result


def test_null_go_rate_with_correlated_pairs():
    suite = HarnessCalibrationSuite(n_bootstrap_iterations=500)
    result = suite.null_calibration(true_p=0.70, n_questions=200, n_trials=1500, seed=12,
                                    correlation=0.6)
    assert result["well_calibrated"], result


def _closed_form_power(n, p1, p2, pearson_r, z_alpha=1.959964):
    """Normal-approximation power of the one-sided alpha/2 paired test."""
    from math import erf, sqrt
    var = p1 * (1 - p1) + p2 * (1 - p2) - 2 * pearson_r * sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    z = (p2 - p1) / sqrt(var / n) - z_alpha
    return 0.5 * (1 + erf(z / sqrt(2)))


@pytest.mark.parametrize("latent_rho", [0.0, 0.6])
def test_simulated_power_matches_closed_form_oracle(latent_rho):
    """Differential oracle: the simulator's power must agree with the
    textbook paired-proportion formula (within Monte-Carlo error)."""
    n, p1, p2 = 800, 0.70, 0.78
    rng = np.random.RandomState(0)
    b, a = _correlated_binomial_pair(rng, p1, p2, 200_000, latent_rho)
    pearson = 0.0 if latent_rho == 0 else np.corrcoef(b, a)[0, 1]
    expected = _closed_form_power(n, p1, p2, pearson)

    suite = HarnessCalibrationSuite(n_bootstrap_iterations=500)
    row = suite.power_curve(true_p_before=p1, effect_sizes=(p2 - p1,), sample_sizes=(n,),
                            n_trials=400, seed=5, correlation=latent_rho)[0]
    assert row["ci_low"] - 0.02 <= expected <= row["ci_high"] + 0.02, (row, expected)
