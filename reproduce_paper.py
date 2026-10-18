"""
Regenerates every number reported in the paper from the current code, with
fixed seeds and the simulation parameters printed next to each result.

    python reproduce_paper.py                 # all sections, fast settings
    python reproduce_paper.py --full          # paper-grade trial counts (slow)
    python reproduce_paper.py --section table1 --full

Sections: speedup, bonferroni, bh_sidedness, calibration, table1, retest, floor
"""
import argparse
import time
from math import erf, sqrt

import numpy as np

from ai_bootstrap_auditor import (AIBootstrapAuditor, AuditBatch, HarnessCalibrationSuite,
                                  SampleSizeOptimizer, _correlated_binomial_pair, clopper_pearson)

Z_975, Z_80 = 1.959964, 0.841621


def header(title, **params):
    print("\n" + "=" * 78)
    print(f" {title}")
    if params:
        print("   " + ", ".join(f"{k}={v}" for k, v in params.items()))
    print("=" * 78)


def rate(k, n):
    lo, hi = clopper_pearson(k, n, 0.95)
    return f"{k / n:6.1%} [{lo:5.1%}, {hi:5.1%}]"


# --- Section 2: vectorization speedup ----------------------------------------

def section_speedup(full):
    reps = 3 if full else 1
    B = 10_000
    header("Section 2 -- vectorized vs. looped paired bootstrap", B=B, repeats=reps)
    rng = np.random.RandomState(0)
    for n in (300, 1500):
        before = rng.binomial(1, 0.7, n).astype(float)
        after = rng.binomial(1, 0.75, n).astype(float)
        loop_t, vec_t = [], []
        for _ in range(reps):
            r = np.random.RandomState(1)
            t0 = time.perf_counter()
            lifts = np.empty(B)
            for i in range(B):
                idx = r.choice(n, size=n, replace=True)
                lifts[i] = after[idx].mean() - before[idx].mean()
            loop_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)
            vec_t.append(time.perf_counter() - t0)
        print(f"  n={n:>5}: loop {min(loop_t) * 1000:8.1f} ms   vectorized {min(vec_t) * 1000:7.1f} ms   "
              f"speedup {min(loop_t) / min(vec_t):4.1f}x")


# --- Section 3.1: Bonferroni NO-GO preservation ------------------------------

def section_bonferroni(full):
    B = 4000
    header("Section 3.1 -- regression kept NO-GO inside a 15-metric Bonferroni batch",
           n=180, true_rates="0.72 -> 0.60", B=B, seed=1)
    r = np.random.RandomState(0)
    before, after = r.binomial(1, 0.72, 180), r.binomial(1, 0.60, 180)
    single = AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=B, seed=1)
    batch = AuditBatch(correction_method="bonferroni").add_metric("target", before, after)
    for j in range(14):
        rr = np.random.RandomState(10000 + j)
        batch.add_metric(f"null_{j}", rr.binomial(1, 0.7, 180), rr.binomial(1, 0.7, 180))
    res = batch.run(n_iterations=B, verbose=False, seed=1)[0]
    print(f"  uncorrected : lift {single['observed_lift']:+.3f}  CI [{single['ci_lower']:+.3f}, "
          f"{single['ci_upper']:+.3f}]  -> {single['status']}")
    print(f"  Bonferroni  : CI [{res['ci_lower']:+.3f}, {res['ci_upper']:+.3f}] (simultaneous)  "
          f"-> {res['status']}  (status from corrected CI alone would be REVIEW)")


# --- Finding 4: BH sidedness --------------------------------------------------

def section_bh_sidedness(full):
    n_sets = 400 if full else 150
    B = 4000
    header("Finding 4 -- BH batch of one vs. single audit (should never disagree)",
           datasets=n_sets, n=200, true_rates="0.70 -> 0.76", B=B)
    disagree = 0
    for s in range(n_sets):
        r = np.random.RandomState(s)
        b, a = r.binomial(1, 0.70, 200), r.binomial(1, 0.76, 200)
        single = AIBootstrapAuditor("m", "increase").audit(b, a, n_iterations=B, seed=1)
        bh = AuditBatch(correction_method="benjamini_hochberg").add_metric("m", b, a)
        disagree += single["status"] != bh.run(n_iterations=B, verbose=False, seed=1)[0]["status"]
    print(f"  disagreements: {disagree}/{n_sets}   (TAE-2026 code: 49/400, BH more liberal)")


# --- Section 4.1: calibration ------------------------------------------------

def section_calibration(full):
    trials = 2000 if full else 500
    B = 1000 if full else 500
    header("Section 4.1 -- A/A false-positive rate and CI coverage (exact 99% bounds)",
           n_trials=trials, B=B, true_p=0.70, effect_for_coverage=0.05)
    suite = HarnessCalibrationSuite(n_bootstrap_iterations=B)
    for n in (20, 50, 200):
        for rho in (0.0, 0.6):
            null = suite.null_calibration(true_p=0.70, n_questions=n, n_trials=trials, seed=n, correlation=rho)
            cov = suite.ci_coverage(true_p_before=0.70, true_effect=0.05, n_questions=n, n_trials=trials,
                                    seed=n + 1, correlation=rho)
            print(f"  n={n:>4} rho={rho:.1f}  A/A GO rate {null['observed_false_positive_rate']:6.2%} "
                  f"[{null['ci_low']:.2%}, {null['ci_high']:.2%}] target 2.50% "
                  f"{'ok ' if null['well_calibrated'] else 'OFF'}   coverage {cov['observed_coverage']:6.1%} "
                  f"[{cov['ci_low']:.1%}, {cov['ci_high']:.1%}] {'ok' if cov['well_calibrated'] else 'OFF'}")


# --- Table 1: correlation-aware required n + closed-form oracle -------------

def closed_form_n(p1, p2, r):
    var = p1 * (1 - p1) + p2 * (1 - p2) - 2 * r * sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    return (Z_975 + Z_80) ** 2 * var / (p2 - p1) ** 2


def section_table1(full):
    n_trials = 100 if full else 30
    B = 800 if full else 300
    header("Table 1 -- required n (80% power, +3pt at 70% baseline) vs. closed-form oracle",
           n_trials_per_point=n_trials, B=B, search="adaptive, start_n=500, seed=5")
    if not full:
        print("  (fast mode: simulated n is noisy at n_trials=30; use --full for paper numbers)")
    optimizer = SampleSizeOptimizer(n_bootstrap_iterations=B)
    for rho in (0.0, 0.3, 0.6, 0.85):
        b, a = _correlated_binomial_pair(np.random.RandomState(1), 0.70, 0.73, 200_000, rho)
        pearson = 0.0 if rho == 0 else float(np.corrcoef(b, a)[0, 1])
        np.random.seed(5)  # audits inside the search draw from the global RNG
        t0 = time.perf_counter()
        res = optimizer.find_required_n_adaptive(true_p_before=0.70, effect_size=0.03, target_power=0.80,
                                                 start_n=500, n_trials=n_trials, seed=5, correlation=rho)
        print(f"  latent rho={rho:.2f}  Pearson r={pearson:.2f}  simulated n={res['required_n']}  "
              f"closed-form n={closed_form_n(0.70, 0.73, pearson):.0f}  ({time.perf_counter() - t0:.0f}s)")


# --- Reviewer XfWC-W2: REVIEW -> retest inflation ----------------------------

def section_retest(full):
    trials = 2000 if full else 600
    B = 800
    header("REVIEW -> RETEST under the null: P(eventually GO) vs. number of looks",
           n_trials=trials, start_n=100, increment=100, B=B, true_p=0.70)
    for looks in (1, 2, 3, 5, 10):
        fp = 0
        for s in range(trials):
            r = np.random.RandomState(90000 + s)
            before, after = r.binomial(1, 0.7, 100 * looks), r.binomial(1, 0.7, 100 * looks)
            for k in range(1, looks + 1):
                status = AIBootstrapAuditor("m", "increase").audit(
                    before[:100 * k], after[:100 * k], n_iterations=B, seed=k)["status"]
                if status != "REVIEW":
                    fp += status == "GO"
                    break
        print(f"  up to {looks:>2} looks: {rate(fp, trials)}   (nominal single look: 2.5%)")


# --- Reviewer XfWC-W4: small-n coverage near the floor -----------------------

def section_floor(full):
    trials = 2000 if full else 600
    B = 1000
    header("Coverage of the nominal 95% percentile CI near the n=10 floor",
           n_trials=trials, B=B, binary="0.70 -> 0.75", lognormal="mu 0 -> 0.2, sigma 1")
    true_ln = np.exp(0.2 + 0.5) - np.exp(0.5)
    for n in (10, 20, 30, 50, 100, 200):
        cov_bin = cov_ln = 0
        for s in range(trials):
            r = np.random.RandomState(s + n * 1000)
            res = AIBootstrapAuditor("m", "increase").audit(r.binomial(1, .7, n), r.binomial(1, .75, n),
                                                            n_iterations=B, seed=s)
            cov_bin += res["ci_lower"] <= 0.05 <= res["ci_upper"]
            r = np.random.RandomState(s + n * 7000)
            res = AIBootstrapAuditor("m", "increase").audit(r.lognormal(0, 1, n), r.lognormal(0.2, 1, n),
                                                            n_iterations=B, seed=s)
            cov_ln += res["ci_lower"] <= true_ln <= res["ci_upper"]
        print(f"  n={n:>4}: binary {rate(cov_bin, trials)}   lognormal {rate(cov_ln, trials)}")


SECTIONS = {
    "speedup": section_speedup, "bonferroni": section_bonferroni, "bh_sidedness": section_bh_sidedness,
    "calibration": section_calibration, "table1": section_table1, "retest": section_retest,
    "floor": section_floor,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=list(SECTIONS), action="append")
    parser.add_argument("--full", action="store_true", help="paper-grade trial counts (slow)")
    args = parser.parse_args()
    for name in args.section or SECTIONS:
        SECTIONS[name](args.full)
