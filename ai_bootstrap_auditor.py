import math
import time
import numpy as np
import pandas as pd

# Minimum paired observations required before ANY class in this module will
# issue a verdict. Below this, percentile bootstrap CIs are unstable enough
# that a confident-looking GO/NO-GO is itself a "fake peak" -- exactly the
# failure mode this tool exists to catch, just occurring inside the tool.
# 10 is a hard floor, not a recommendation -- real confidence needs more.
MIN_PAIRED_OBSERVATIONS = 10

# Between the hard floor and this size, a verdict is still issued but carries
# a warning. Justified by measurement, not assumed: in the calibration study
# (reproduce_paper.py --section floor --full; 2,000 trials, B=1,000) the
# nominal 95% percentile interval covered the true lift less than 95% of the
# time at every n <= 50, for both binary and skewed (lognormal) metrics, with
# exact 95% intervals entirely below nominal.
LOW_SAMPLE_WARNING_N = 50

# Measured coverage (%) of the nominal 95% CI: n -> (binary, lognormal).
# Source: results/floor_full.txt.
MEASURED_SMALL_N_COVERAGE = {10: (87.1, 87.5), 20: (92.4, 90.7), 30: (93.7, 92.0), 50: (93.8, 92.2)}

# Measured one-sided false-GO rate (%) under a true null when REVIEW results
# are retested with +100 observations per look, starting at n=100.
# Source: results/retest_full.txt (2,000 trials, B=800; binary metric, so the
# exact sign-test guard is active). Nominal single look: 2.5%.
MEASURED_RETEST_FALSE_GO = {1: 1.6, 2: 2.8, 3: 3.0, 5: 4.9, 10: 8.6}


# For yes/no metrics, the information lives in the discordant pairs (questions
# whose outcome changed between versions). Below this many, a verdict carries a
# thin-evidence note; GO/NO-GO additionally always has to pass the exact sign
# test (see AIBootstrapAuditor.audit). 20 is a practical threshold, not a
# statistical bound: even at 20 changed questions a clear 70/30 split (14 up,
# 6 down) is not significant (exact one-sided p = 0.058).
MIN_DISCORDANT_PAIRS = 20


def sign_test_upper_tail(successes: int, trials: int) -> float:
    """Exact one-sided sign-test p-value P(X >= successes), X ~ Binomial(trials, 1/2).
    Returns 1.0 when there are no trials (no discordant pairs = no evidence)."""
    if trials == 0:
        return 1.0
    return float(sum(math.comb(trials, i) for i in range(successes, trials + 1)) / 2 ** trials)


def low_sample_warning(n: int):
    """Returns a warning string for MIN_PAIRED_OBSERVATIONS <= n < LOW_SAMPLE_WARNING_N,
    quoting the measured coverage at the nearest calibrated size at or below n;
    None otherwise."""
    if n < MIN_PAIRED_OBSERVATIONS or n >= LOW_SAMPLE_WARNING_N:
        return None  # below the floor the auditors refuse outright instead of warning
    grid_n = max(k for k in MEASURED_SMALL_N_COVERAGE if k <= n)
    worst = min(MEASURED_SMALL_N_COVERAGE[grid_n])
    return (f"Small sample (n={n}, below {LOW_SAMPLE_WARNING_N}). In this tool's own calibration runs, "
            f"the nominal 95% interval contained the true effect only about {worst:.0f}% of the time "
            f"at n={grid_n}, so this verdict is weaker evidence than it looks. Collect more data before "
            f"acting on a borderline result.")


def retest_warning(looks: int):
    """Returns a warning string once the same comparison has been audited on
    `looks` >= 2 different datasets (e.g. REVIEW -> add data -> rerun),
    quoting the measured false-GO inflation; None for a single look."""
    if looks < 2:
        return None
    grid = max(k for k in MEASURED_RETEST_FALSE_GO if k <= looks)
    return (f"This comparison has been re-run on {looks} different datasets. Re-testing until a "
            f"result appears inflates false GOs: in this tool's calibration runs, a true no-effect "
            f"change was eventually called GO {MEASURED_RETEST_FALSE_GO[grid]:.1f}% of the time with up "
            f"to {grid} looks, versus 2.5% for a single pre-planned test. Prefer planning n with the "
            f"Sample Size Planner and testing once.")


class RetestTracker:
    """
    Counts how many DISTINCT datasets each named comparison has been audited
    on, so repeated REVIEW -> retest cycles are visible instead of silent.
    Datasets are fingerprinted by content, so re-auditing identical data
    (e.g. a UI rerun or a new resampling seed) does not count as a new look.
    """
    def __init__(self):
        self._seen = {}  # comparison key -> set of data fingerprints

    @staticmethod
    def _fingerprint(before, after) -> int:
        b = np.ascontiguousarray(np.asarray(before, dtype=float))
        a = np.ascontiguousarray(np.asarray(after, dtype=float))
        return hash((b.tobytes(), a.tobytes()))

    def record(self, key: str, before, after) -> int:
        """Registers this dataset under `key`; returns the number of distinct looks so far."""
        seen = self._seen.setdefault(key, set())
        seen.add(self._fingerprint(before, after))
        return len(seen)

    def looks(self, key: str) -> int:
        return len(self._seen.get(key, ()))

    def reset(self, key: str = None):
        if key is None:
            self._seen.clear()
        else:
            self._seen.pop(key, None)

class ValidationError(ValueError):
    """Base class for input validation failures in this module. Subclasses
    ValueError so every existing `except ValueError` call site (across the
    app and this module) continues to work unchanged -- this is purely
    additive, not a breaking change."""
    pass


class DataIntegrityError(ValidationError):
    """Raised when input contains missing, non-numeric, or infinite values."""
    pass


class MinimumSampleFloorError(ValidationError):
    """Raised when paired observations fall below MIN_PAIRED_OBSERVATIONS."""
    pass


class PairingError(ValidationError):
    """Raised when before/after arrays aren't the same length."""
    pass


class AIBootstrapAuditor:
    def __init__(self, metric_name: str, goal: str = "increase", confidence_level: float = 0.95,
                 n_comparisons: int = 1):
        """
        An enterprise-grade statistical auditing class for AI product features.

        Parameters:
        - metric_name: Name of the metric (e.g., "Accuracy", "Latency")
        - goal: "increase" (higher is better) or "decrease" (lower is better)
        - confidence_level: Usually 0.95 (95% confidence)
        - n_comparisons: How many metrics/tests are being evaluated together in this
          audit round (e.g., if 5 teams each test their own metric in the same
          review cycle, set n_comparisons=5 for each). Applies a Bonferroni
          correction to the significance threshold: alpha_effective = alpha / n_comparisons.
          Default 1 (no correction) preserves prior single-test behavior.

          WHY THIS MATTERS: running this auditor independently across many metrics
          at the nominal alpha=0.05 means you should EXPECT roughly one false "GO"
          per ~20 metrics tested, purely by chance -- even if nothing actually
          improved anywhere. This is the same "fake peak" problem the auditor
          exists to catch, just at the multi-metric level instead of the
          multi-sample level. If you're running this across a batch of metrics in
          one review cycle, set n_comparisons to the batch size.
        """
        if goal not in ["increase", "decrease"]:
            raise ValueError("Goal must be either 'increase' or 'decrease'")
        if n_comparisons < 1:
            raise ValueError("n_comparisons must be >= 1")

        self.metric_name = metric_name
        self.goal = goal
        self.confidence_level = confidence_level
        self.n_comparisons = n_comparisons

    def audit(self, before_scores: np.ndarray, after_scores: np.ndarray, n_iterations: int = 10000,
              seed: int = None) -> dict:
        """
        Runs the paired bootstrap resampling simulation and generates an executive report.
        Vectorized: draws all n_iterations resamples in one call instead of looping
        in pure Python, which is the dominant cost at realistic sample sizes.

        seed: if provided, makes the resampling reproducible -- same inputs +
        same seed always give the same verdict and CI. Leave as None for a
        fresh random draw each call (the original behavior).
        """
        before = np.asarray(before_scores, dtype=float)
        after = np.asarray(after_scores, dtype=float)

        if len(before) != len(after):
            raise PairingError("Datasets must be paired! Length of 'before' and 'after' must match.")

        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            raise DataIntegrityError(
                "Data contains missing, non-numeric, or infinite values. Clean the input before "
                "auditing -- the bootstrap cannot produce a meaningful result otherwise."
            )

        if len(before) < MIN_PAIRED_OBSERVATIONS:
            raise MinimumSampleFloorError(
                f"Only {len(before)} paired observation(s) provided. This tool exists to catch "
                f"false confidence from small samples, so it refuses to issue a verdict below "
                f"{MIN_PAIRED_OBSERVATIONS} paired observations -- results at this size are not "
                f"statistically meaningful, no matter how confident they look."
            )

        n = len(before)
        observed_before_mean = np.mean(before)
        observed_after_mean = np.mean(after)
        observed_lift = observed_after_mean - observed_before_mean

        # --- Vectorized paired resampling (fix #1) ---
        # Draw all (n_iterations x n) index sets at once, rather than looping
        # n_iterations times in Python and calling np.random.choice/np.mean
        # separately each time. Same statistical procedure; the speedup is
        # modest and hardware-dependent (measured 1.4-4.9x at n in {300, 1500},
        # B=10,000 across machines) -- an earlier, unbenchmarked "10-50x"
        # estimate was wrong. Re-measure with reproduce_paper.py --section speedup.
        rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = rng.choice(n, size=(n_iterations, n), replace=True)
        resampled_before = before[indices]   # shape: (n_iterations, n)
        resampled_after = after[indices]     # shape: (n_iterations, n)
        bootstrap_lifts = resampled_after.mean(axis=1) - resampled_before.mean(axis=1)

        # --- Confidence interval ---
        # Bonferroni-adjusted alpha if this audit is one of several run together
        # (fix #2). With n_comparisons=1 this reduces exactly to the original
        # behavior.
        alpha = 1.0 - self.confidence_level
        alpha_effective = alpha / self.n_comparisons
        lower_percentile = (alpha_effective / 2.0) * 100
        upper_percentile = (1.0 - (alpha_effective / 2.0)) * 100

        ci_lower = np.percentile(bootstrap_lifts, lower_percentile)
        ci_upper = np.percentile(bootstrap_lifts, upper_percentile)

        # --- Significance, using the SAME corrected alpha ---
        # Three-state verdict, not two: a CI that straddles zero means "not
        # enough evidence either way" (REVIEW/RETEST), which is a genuinely
        # different situation from a CI that's confidently on the bad side
        # (NO-GO/REGRESSION). Collapsing those into one "NO-GO" bucket (the
        # original design) is the same "confident-sounding conclusion from
        # ambiguous evidence" problem this tool exists to catch elsewhere --
        # caught in practitioner review, credited in the docstring below.
        # p_value: ONE-sided percentile-bootstrap p-value for "no improvement",
        # (#{lift* on the bad side of 0} + 1) / (B + 1). The +1 keeps p > 0
        # (a literal 0 is not a valid p-value and distorts BH ordering). The
        # verdict rule (CI bound vs 0) is a one-sided test at alpha_effective/2,
        # so p_value should be compared against alpha_effective/2, not alpha.
        if self.goal == "increase":
            p_value = (np.sum(bootstrap_lifts <= 0) + 1) / (n_iterations + 1)
            if ci_lower > 0:
                status, executive_verdict = "GO", "GO (CONFIRMED IMPROVEMENT)"
                risk_statement = "The observed improvement is unlikely to be sampling noise."
            elif ci_upper < 0:
                status, executive_verdict = "NO-GO", "NO-GO (REGRESSION DETECTED)"
                risk_statement = "The data confidently shows this got worse, not better."
            else:
                status, executive_verdict = "REVIEW", "REVIEW / RETEST (NOT ENOUGH EVIDENCE)"
                risk_statement = ("The sample can't yet distinguish a real improvement from noise -- "
                                   "the confidence range still includes zero. Collect more data or "
                                   "retest before deciding either way.")
        else:  # goal == "decrease"
            p_value = (np.sum(bootstrap_lifts >= 0) + 1) / (n_iterations + 1)
            if ci_upper < 0:
                status, executive_verdict = "GO", "GO (CONFIRMED REDUCTION)"
                risk_statement = "The observed reduction is unlikely to be sampling noise."
            elif ci_lower > 0:
                status, executive_verdict = "NO-GO", "NO-GO (REGRESSION DETECTED)"
                risk_statement = "The data confidently shows this got worse, not better."
            else:
                status, executive_verdict = "REVIEW", "REVIEW / RETEST (NOT ENOUGH EVIDENCE)"
                risk_statement = ("The sample can't yet distinguish a real reduction from noise -- "
                                   "the confidence range still includes zero. Collect more data or "
                                   "retest before deciding either way.")

        # --- Exact guard for yes/no (dichotomous) metrics ---
        # With paired binary outcomes, ties (both right / both wrong) carry no
        # information about the difference; everything rests on the DISCORDANT
        # pairs. When there are few of them, the percentile bootstrap is
        # overconfident: in our checks it called GO with 4/4, 5/5 or 7/8
        # discordant pairs in the improving direction, where the exact one-sided
        # sign (McNemar) test gives p = 0.0625, 0.031, 0.035 > 0.025. So for
        # dichotomous data a GO or NO-GO must ALSO pass the exact test at the
        # same alpha_effective/2, and the reported p_value is the larger of the
        # bootstrap and exact p-values (so BH, which ranks p_value, applies the
        # same guard). Detection is by value set, so it also covers recoded
        # yes/no data (e.g. {-1, 0} after negation) and stays mirror/affine
        # invariant.
        discordant = exact_p_improve = exact_p_regress = None
        exact_guard_applied = False
        notes = []
        values = np.unique(np.concatenate([before, after]))
        if len(values) <= 2:
            up, down = int(np.sum(after > before)), int(np.sum(after < before))
            discordant = up + down
            improving, regressing = (up, down) if self.goal == "increase" else (down, up)
            exact_p_improve = sign_test_upper_tail(improving, discordant)
            exact_p_regress = sign_test_upper_tail(regressing, discordant)
            p_value = max(p_value, exact_p_improve)
            threshold = alpha_effective / 2.0
            if (status == "GO" and exact_p_improve > threshold) or \
                    (status == "NO-GO" and exact_p_regress > threshold):
                exact_guard_applied = True
                status, executive_verdict = "REVIEW", "REVIEW / RETEST (NOT ENOUGH EVIDENCE)"
                exact_p = exact_p_improve if improving >= regressing else exact_p_regress
                risk_statement = (
                    f"The bootstrap interval excludes zero, but only {discordant} of {n} questions "
                    f"changed outcome between versions ({up} up, {down} down), and the exact sign "
                    f"test on those changes is not significant (one-sided p = {exact_p:.3f} > "
                    f"{threshold:.4f}). With this few changes the bootstrap is overconfident -- "
                    f"collect more data before deciding."
                )
            if discordant == 0:
                notes.append(f"No question changed outcome between versions (all {n} pairs tied), so the "
                             f"interval [0, 0] means 'no information about a difference', not "
                             f"'certainly no difference'.")
            elif discordant < MIN_DISCORDANT_PAIRS:
                notes.append(f"Only {discordant} of {n} questions changed outcome between versions; "
                             f"verdicts on yes/no metrics rest on these changes alone, so evidence is "
                             f"thin below {MIN_DISCORDANT_PAIRS}.")

        correction_note = (
            f" (Bonferroni-corrected for {self.n_comparisons} simultaneous comparisons: "
            f"effective threshold {alpha_effective:.4f} instead of {alpha:.4f})"
            if self.n_comparisons > 1 else ""
        )
        warnings = [w for w in [low_sample_warning(n)] + notes if w]

        return {
            "metric": self.metric_name,
            "goal": self.goal,
            "sample_size": n,
            "observed_before": observed_before_mean,
            "observed_after": observed_after_mean,
            "observed_lift": observed_lift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "alpha_effective": alpha_effective,
            "n_comparisons": self.n_comparisons,
            "status": status,
            "verdict": executive_verdict,
            "risk_statement": risk_statement + correction_note,
            # n-based band and discordant-pair notes, joined; None if neither applies
            "low_sample_warning": " ".join(warnings) if warnings else None,
            "discordant_pairs": discordant,              # None for non-dichotomous metrics
            "exact_sign_test_p": exact_p_improve,        # one-sided, improving direction
            "exact_guard_applied": exact_guard_applied,  # True if the exact test downgraded GO/NO-GO
        }

    @staticmethod
    def to_dataframe(results: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_executive_report().
        Returns a one-row DataFrame so a UI (Streamlit, etc.) can render a
        real table/traffic-light widget instead of parsing terminal text
        via stdout redirection.
        """
        return pd.DataFrame([{
            "Metric": results["metric"],
            "Verdict": results["verdict"],
            "Sample Size": results["sample_size"],
            "Baseline Mean": results["observed_before"],
            "After Mean": results["observed_after"],
            "Lift": results["observed_lift"],
            "CI Lower": results["ci_lower"],
            "CI Upper": results["ci_upper"],
            "p-value": results["p_value"],
            "q-value": results.get("q_value"),  # present only in BH-routed results
        }])

    def print_executive_report(self, results: dict):
        """
        Formats raw statistics into a clear non-technical business evaluation matrix.
        """
        print("=" * 65)
        print(f" EXECUTIVE EVALUATION REPORT: {results['metric'].upper()}")
        print("=" * 65)
        print(f"  VERDICT         : {results['verdict']}")
        print(f"  Tested Sample   : {results['sample_size']} paired queries")
        print(f"  Baseline Mean   : {results['observed_before']:.4f}")
        print(f"  New Mean        : {results['observed_after']:.4f}")
        print(f"  Estimated Change: {results['observed_lift']:+.4f}")
        if results["n_comparisons"] > 1:
            print(f"  95% CI (Bonferroni-adjusted for {results['n_comparisons']} tests): "
                  f"[{results['ci_lower']:+.4f} to {results['ci_upper']:+.4f}]")
        else:
            print(f"  95% Confidence  : [{results['ci_lower']:+.4f} to {results['ci_upper']:+.4f}]")
        print("-" * 65)
        print(f"  ANALYSIS: {results['risk_statement']}")
        if results.get("low_sample_warning"):
            print(f"  WARNING : {results['low_sample_warning']}")
        print("=" * 65 + "\n")


class AuditBatch:
    """
    Orchestrates multiple AIBootstrapAuditor runs as a single review cycle,
    automatically applying a multiple-comparisons correction to every
    metric in the batch.

    WHY THIS EXISTS: a per-metric correction (like Bonferroni) only works if
    every caller remembers to set n_comparisons correctly by hand -- which is
    exactly the kind of thing that gets silently forgotten in practice (a
    PM adds a 6th metric to this quarter's review and doesn't realize every
    existing AIBootstrapAuditor instantiation needs its n_comparisons bumped
    too). AuditBatch removes that failure mode: you register every metric
    being evaluated in this cycle, and it derives the correction from the
    batch automatically, once, in one place.

    correction_method:
      - "bonferroni" (default): controls the family-wise error rate. Simple,
        conservative -- guarantees the overall false-positive rate across the
        whole batch stays at or below alpha, but gets stricter (harder to
        reach significance) the more metrics you add, even for metrics with
        real effects. Correctly widens each metric's own confidence interval,
        so the reported CI stays simultaneously valid across the batch.
      - "benjamini_hochberg": controls the FALSE DISCOVERY RATE instead --
        the expected proportion of "GO" verdicts that are false positives,
        among all "GO" verdicts. Substantially less conservative than
        Bonferroni as batch size grows, at the cost of a weaker guarantee
        (it bounds the *rate* of false discoveries, not the probability of
        *any* false discovery). Recommended once batches regularly exceed
        ~10 metrics, where Bonferroni starts rejecting real effects it
        shouldn't. NOTE: in this mode, the reported confidence interval for
        each metric is the RAW, uncorrected interval at the nominal
        confidence level -- Benjamini-Hochberg adjusts significance
        decisions (via q-values), not confidence intervals, so the CIs shown
        are not simultaneously valid across the batch the way Bonferroni's
        are. This is flagged explicitly in the report rather than silently
        presenting an interval that looks corrected but isn't.
    """
    VALID_METHODS = ("bonferroni", "benjamini_hochberg")

    def __init__(self, confidence_level: float = 0.95, correction_method: str = "bonferroni"):
        if correction_method not in self.VALID_METHODS:
            raise ValueError(f"correction_method must be one of {self.VALID_METHODS}")
        self.confidence_level = confidence_level
        self.correction_method = correction_method
        self._metrics = []    # core value metrics: list of dicts
        self._guardrails = []  # guardrail metrics: list of dicts (separate family, own routing)

    def add_metric(self, metric_name: str, before_scores, after_scores, goal: str = "increase"):
        """
        Register one metric's paired before/after data into this review
        cycle. Does not run the audit yet -- just queues it, so the total
        batch size (and therefore the correction) is known before any
        individual audit runs.
        """
        if goal not in ["increase", "decrease"]:
            raise ValueError("Goal must be either 'increase' or 'decrease'")
        self._metrics.append({
            "metric_name": metric_name,
            "goal": goal,
            "before": np.array(before_scores),
            "after": np.array(after_scores),
        })
        return self  # allow chaining: batch.add_metric(...).add_metric(...)

    def add_guardrail(self, metric_name: str, before_scores, after_scores,
                       guard_direction: str = "not_increase", margin: float = 0.0,
                       alpha: float = 0.05):
        """
        Register a guardrail/cost metric (latency, tokens, $ per call) --
        a SEPARATE family from core value metrics, routed through
        GuardrailAuditor rather than the correction method used for
        add_metric(). This is deliberate: guardrails need high sensitivity
        to catch regressions, not the conservative correction applied to
        core value metrics. See GuardrailAuditor's docstring for why mixing
        the two families under one correction is wrong.

        alpha here is NOT touched by this batch's correction_method or
        n_comparisons -- guardrails are evaluated independently of how many
        core metrics are in the batch, and independently of each other,
        by default. Pass alpha explicitly per-guardrail if you want to
        tighten a specific one.
        """
        if guard_direction not in ("not_increase", "not_decrease"):
            raise ValueError("guard_direction must be 'not_increase' or 'not_decrease'")
        self._guardrails.append({
            "metric_name": metric_name,
            "guard_direction": guard_direction,
            "margin": margin,
            "alpha": alpha,
            "before": np.array(before_scores),
            "after": np.array(after_scores),
        })
        return self  # allow chaining

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float):
        """
        Standard BH step-up procedure. Returns (significant_mask, q_values),
        both in the ORIGINAL input order (not sorted).
        """
        m = len(p_values)
        order = np.argsort(p_values)
        sorted_p = p_values[order]

        # Largest k such that sorted_p[k-1] <= (k/m) * alpha  (1-indexed k)
        thresholds = (np.arange(1, m + 1) / m) * alpha
        passing = sorted_p <= thresholds
        if passing.any():
            k = np.max(np.where(passing)[0]) + 1  # 1-indexed count of rejections
        else:
            k = 0

        significant_sorted = np.zeros(m, dtype=bool)
        significant_sorted[:k] = True

        # q-values: step-up monotone adjustment, q_(i) = min_{j>=i} (p_(j) * m / j)
        raw_q = sorted_p * m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0, 1)

        # Map back to original order
        significant = np.empty(m, dtype=bool)
        q_values = np.empty(m, dtype=float)
        significant[order] = significant_sorted
        q_values[order] = q_sorted
        return significant, q_values

    def run(self, n_iterations: int = 10000, verbose: bool = True, seed: int = None) -> list:
        """
        Runs every registered core metric's audit, applying whichever
        correction_method was set at construction, AND every registered
        guardrail's audit via GuardrailAuditor (own routing, not affected
        by correction_method). Returns the list of CORE metric results
        (unchanged shape/behavior from before guardrails existed); guardrail
        results are stored internally and included automatically by
        print_batch_summary().

        seed: if provided, makes the whole batch reproducible run to run.
        Each metric/guardrail gets seed+i (a distinct but deterministic
        offset) so they aren't all resampled identically to each other.
        Leave as None for the original fresh-random-draw behavior.
        """
        if not self._metrics and not self._guardrails:
            raise ValueError("No metrics or guardrails registered. Call add_metric() "
                              "and/or add_guardrail() before run().")

        results = []
        if self._metrics:
            n_comparisons = len(self._metrics)
            alpha = 1.0 - self.confidence_level

            if self.correction_method == "bonferroni":
                for i, spec in enumerate(self._metrics):
                    auditor = AIBootstrapAuditor(
                        metric_name=spec["metric_name"],
                        goal=spec["goal"],
                        confidence_level=self.confidence_level,
                        n_comparisons=n_comparisons,
                    )
                    metric_seed = None if seed is None else seed + i
                    result = auditor.audit(spec["before"], spec["after"], n_iterations=n_iterations,
                                            seed=metric_seed)

                    # Preserve NO-GO through correction (same guard as the BH
                    # branch below). Correction exists to raise the bar for
                    # claiming a WIN; it must not demote a confident regression
                    # to REVIEW just because the batch grew and the widened CI
                    # now crosses zero. The TAE-2026 submission described this
                    # guard but the shipped code lacked it -- see
                    # tests/test_regressions.py::test_bonferroni_preserves_raw_no_go.
                    raw = AIBootstrapAuditor(
                        metric_name=spec["metric_name"],
                        goal=spec["goal"],
                        confidence_level=self.confidence_level,
                        n_comparisons=1,
                    ).audit(spec["before"], spec["after"], n_iterations=n_iterations, seed=metric_seed)
                    result["raw_status"] = raw["status"]
                    if raw["status"] == "NO-GO" and result["status"] != "NO-GO":
                        result["status"] = "NO-GO"
                        result["verdict"] = "NO-GO (REGRESSION DETECTED)"
                        result["risk_statement"] = (
                            "The uncorrected interval confidently shows this got worse; a regression "
                            "is kept as NO-GO regardless of multiple-comparisons correction, which only "
                            "raises the bar for declaring a win. "
                            f"(Bonferroni-corrected interval shown, {n_comparisons} comparisons.)"
                        )
                    results.append(result)
                    if verbose:
                        auditor.print_executive_report(result)

            else:  # benjamini_hochberg
                raw_results = []
                for i, spec in enumerate(self._metrics):
                    auditor = AIBootstrapAuditor(
                        metric_name=spec["metric_name"],
                        goal=spec["goal"],
                        confidence_level=self.confidence_level,
                        n_comparisons=1,
                    )
                    metric_seed = None if seed is None else seed + i
                    raw_results.append(auditor.audit(spec["before"], spec["after"], n_iterations=n_iterations,
                                                       seed=metric_seed))

                # SIDEDNESS: a single audit calls GO iff the lower bound of a
                # TWO-sided (1 - alpha) CI clears zero, i.e. a ONE-sided test
                # at alpha/2. p_value here is one-sided, so BH must run at
                # alpha/2 too. Running it at alpha (the TAE-2026 behavior)
                # made a batch of ONE metric twice as liberal as the plain
                # audit -- caught by the metamorphic check "a batch of size 1
                # must agree with a single audit" (tests/test_metamorphic.py).
                p_values = np.array([r["p_value"] for r in raw_results])
                significant, q_values = self._benjamini_hochberg(p_values, alpha / 2.0)

                for r, spec, is_sig, q in zip(raw_results, self._metrics, significant, q_values):
                    r = dict(r)
                    r["q_value"] = q
                    r["bh_significant"] = bool(is_sig)
                    r["n_comparisons"] = n_comparisons
                    direction_word = "IMPROVEMENT" if spec["goal"] == "increase" else "REDUCTION"

                    # Preserve three-state semantics through correction: a
                    # confident regression (raw CI fully on the bad side)
                    # stays a regression regardless of multiple-testing
                    # correction -- correction only raises the bar for
                    # calling something a WIN, it doesn't make a real
                    # regression less real. Otherwise, BH significance
                    # decides GO vs REVIEW: failing the corrected bar means
                    # "not enough evidence after correction," which is a
                    # REVIEW situation, not an overwritten two-state NO-GO.
                    # GO additionally requires the raw CI to clear zero, so BH
                    # can only ever narrow the set of uncorrected GOs, never
                    # add to it (BH-significant implies p <= alpha/2, which
                    # matches the raw rule up to percentile-interpolation edges).
                    if r["status"] == "NO-GO":
                        r["status"], r["verdict"] = "NO-GO", "NO-GO (REGRESSION DETECTED)"
                    elif is_sig and r["status"] == "GO":
                        r["status"], r["verdict"] = "GO", f"GO (CONFIRMED {direction_word})"
                    else:
                        r["status"] = "REVIEW"
                        r["verdict"] = "REVIEW / RETEST (NOT ENOUGH EVIDENCE)"

                    r["risk_statement"] = (
                        f"Benjamini-Hochberg adjusted p-value (q-value): {q:.4f} "
                        f"(raw one-sided p-value: {r['p_value']:.4f}, compared at FDR level "
                        f"{alpha / 2.0:.4f}; batch size: {n_comparisons}). "
                        f"NOTE: confidence interval below is the RAW nominal interval, "
                        f"not simultaneously corrected -- BH adjusts the significance "
                        f"decision, not the interval."
                    )
                    results.append(r)
                    if verbose:
                        auditor_for_print = AIBootstrapAuditor(spec["metric_name"], spec["goal"])
                        auditor_for_print.print_executive_report(r)

        # --- Guardrails: separate routing, always independent of correction_method ---
        guardrail_results = []
        for i, spec in enumerate(self._guardrails):
            guard = GuardrailAuditor(
                metric_name=spec["metric_name"],
                guard_direction=spec["guard_direction"],
                margin=spec["margin"],
                alpha=spec["alpha"],
            )
            guard_seed = None if seed is None else seed + 1000 + i  # offset so guardrails
            # don't share draws with core metrics at the same index
            g_result = guard.audit(spec["before"], spec["after"], n_iterations=n_iterations, seed=guard_seed)
            guardrail_results.append(g_result)
            if verbose:
                guard.print_executive_report(g_result)

        self._last_guardrail_results = guardrail_results
        return results

    def to_dataframe(self, results: list) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_batch_summary()'s
        core-metrics table. One row per metric.
        """
        rows = []
        for r in results:
            row = {
                "Metric": r["metric"],
                "Verdict": r["verdict"],
                "Lift": r["observed_lift"],
                "CI Lower": r["ci_lower"],
                "CI Upper": r["ci_upper"],
            }
            if self.correction_method == "benjamini_hochberg":
                row["q-value"] = r.get("q_value")
            else:
                row["p-value"] = r.get("p_value")
            rows.append(row)
        return pd.DataFrame(rows)

    def guardrails_to_dataframe(self) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_batch_summary()'s
        guardrail section. Must be called after run(); reads the results
        stored by the most recent run() call.
        """
        guardrail_results = getattr(self, "_last_guardrail_results", [])
        rows = [{
            "Metric": r["metric"],
            "Status": r["status"],
            "Lift": r["observed_lift"],
            "Margin": r["margin"],
            "CI Lower": r["lower_bound"],
            "CI Upper": r["upper_bound"],
        } for r in guardrail_results]
        return pd.DataFrame(rows)

    def print_batch_summary(self, results: list):
        """
        A single consolidated table across every metric in the batch --
        the actual "executive report" a PM would want to scan in one
        glance, rather than reading N separate full reports. Automatically
        includes a second section for guardrail results if any were
        registered via add_guardrail() and run() has been called.
        """
        if results:
            n_comparisons = len(results)
            print("#" * 65)
            print(f" CORE VALUE METRICS: {n_comparisons} metrics tested this cycle")
            print(f" Correction method: {self.correction_method}")
            if self.correction_method == "bonferroni":
                print(f" Bonferroni-corrected threshold applied to each: "
                      f"{(1 - self.confidence_level) / n_comparisons:.4f}")
            else:
                print(f" Benjamini-Hochberg: significance based on one-sided q-value <= "
                      f"{(1 - self.confidence_level) / 2:.4f} (false discovery rate control)")
            print("#" * 65)
            if self.correction_method == "benjamini_hochberg":
                print(f"  {'Metric':<28} {'Lift':>12} {'q-value':>10} {'Verdict':>26}")
                print("  " + "-" * 78)
                for r in results:
                    lift_str = f"{r['observed_lift']:+.4f}"
                    q_str = f"{r['q_value']:.4f}"
                    print(f"  {r['metric']:<28} {lift_str:>12} {q_str:>10} {r['verdict']:>26}")
            else:
                print(f"  {'Metric':<28} {'Lift':>12} {'Verdict':>28}")
                print("  " + "-" * 68)
                for r in results:
                    lift_str = f"{r['observed_lift']:+.4f}"
                    print(f"  {r['metric']:<28} {lift_str:>12} {r['verdict']:>28}")
            print("#" * 65 + "\n")

        guardrail_results = getattr(self, "_last_guardrail_results", [])
        if guardrail_results:
            print("#" * 65)
            print(f" GUARDRAIL / COST METRICS: {len(guardrail_results)} checked this cycle")
            print(" Routing: GuardrailAuditor (high-sensitivity, non-inferiority test --")
            print(" NOT the same correction as core metrics above; see GuardrailAuditor")
            print(" docstring for why applying a conservative correction here would be wrong)")
            print("#" * 65)
            print(f"  {'Metric':<38} {'Lift':>10} {'Status':>18}")
            print("  " + "-" * 70)
            for r in guardrail_results:
                lift_str = f"{r['observed_lift']:+.2f}"
                print(f"  {r['metric']:<38} {lift_str:>10} {r['status']:>18}")
            if any(r["status"] == "INCONCLUSIVE" for r in guardrail_results):
                print("\n  ⚠ One or more guardrails need REVIEW/RETEST -- decision pending, do not assume safe")
                print("    to ship until reviewed manually or re-tested with more data.")
            print("#" * 65 + "\n")


class GuardrailAuditor:
    """
    Purpose-built for guardrail/cost metrics (latency, tokens, $ per call),
    where the question is NOT "did this improve" but "did this regress
    beyond an acceptable margin" -- and where missing a real regression is
    the expensive mistake, not raising a false alarm.

    THIS IS DELIBERATELY NOT AIBootstrapAuditor WITH A STRICTER THRESHOLD.
    Applying a conservative correction (like Bonferroni) to a guardrail
    metric makes it HARDER to detect a regression, not easier -- a stricter
    significance bar means it takes a bigger, more obvious regression to
    trigger a flag. That is the opposite of what a guardrail should do.
    Guardrails should default to high sensitivity, not high conservatism.

    METHOD: non-inferiority testing via bootstrap, with an explicit
    THIRD outcome (INCONCLUSIVE) instead of collapsing "not enough
    evidence" into "assume it's fine." Treating an inconclusive result as
    a pass is the absence-of-evidence trap -- exactly what would let a
    real, moderate cost regression slip through silently. This auditor
    refuses to do that: it only reports PASS when there is actual
    confidence the regression, if any, is within the allowed margin.

    Parameters:
    - guard_direction: "not_increase" (e.g. latency, cost, token count --
      going up is bad) or "not_decrease" (e.g. a safety/accuracy floor --
      going down is bad).
    - margin: how much change is tolerated before it counts as a real
      regression. margin=0 means "must not get worse AT ALL" (strict).
      For token count you might set margin=0 (any real increase is a
      regression); for latency you might set margin=20 (allow up to 20ms
      of drift before flagging).
    - alpha: one-sided significance level for the bounds below. Default
      0.05 (95% one-sided confidence). Deliberately NOT tightened by a
      multiple-comparisons correction by default -- see class docstring.
      If you are genuinely running many guardrail checks at once and want
      to control false alarms across them, you may still pass a smaller
      alpha manually, but the default here favors catching regressions
      over avoiding false alarms, which is the right default for a
      guardrail.
    """
    def __init__(self, metric_name: str, guard_direction: str = "not_increase",
                 margin: float = 0.0, alpha: float = 0.05):
        if guard_direction not in ("not_increase", "not_decrease"):
            raise ValueError("guard_direction must be 'not_increase' or 'not_decrease'")
        if margin < 0:
            raise ValueError("margin must be >= 0 (it's a tolerance magnitude, sign is handled internally)")
        self.metric_name = metric_name
        self.guard_direction = guard_direction
        self.margin = margin
        self.alpha = alpha

    def audit(self, before_scores, after_scores, n_iterations: int = 10000, seed: int = None) -> dict:
        before = np.asarray(before_scores, dtype=float)
        after = np.asarray(after_scores, dtype=float)
        if len(before) != len(after):
            raise PairingError("Datasets must be paired! Length of 'before' and 'after' must match.")

        finite_mask = np.isfinite(before) & np.isfinite(after)
        has_bad_values = not finite_mask.all()
        n_clean = int(finite_mask.sum())

        # Below the minimum, or with unusable values: fail safe into
        # INCONCLUSIVE (already this class's designed "don't assume safe"
        # state) rather than letting a near-empty bootstrap produce a
        # falsely confident PASS.
        if has_bad_values or n_clean < MIN_PAIRED_OBSERVATIONS:
            reason = ("contains missing/non-numeric/infinite values" if has_bad_values
                       else f"has only {n_clean} usable paired observation(s), "
                            f"below the {MIN_PAIRED_OBSERVATIONS}-observation minimum")
            return {
                "metric": self.metric_name,
                "guard_direction": self.guard_direction,
                "margin": self.margin,
                "alpha": self.alpha,
                "sample_size": n_clean,
                "observed_before": float(np.nan),
                "observed_after": float(np.nan),
                "observed_lift": float(np.nan),
                "lower_bound": float(np.nan),
                "upper_bound": float(np.nan),
                "status": "INCONCLUSIVE",
                "verdict": "REVIEW / RETEST (NOT ENOUGH EVIDENCE)",
                "risk_statement": (
                    f"The data {reason}, so this guardrail cannot be evaluated. "
                    f"Decision pending -- review manually or collect more data before shipping."
                ),
            }

        before, after = before[finite_mask], after[finite_mask]
        n = len(before)
        observed_lift = np.mean(after) - np.mean(before)

        # Vectorized bootstrap (same approach as AIBootstrapAuditor's fix #1)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = rng.choice(n, size=(n_iterations, n), replace=True)
        bootstrap_lifts = after[indices].mean(axis=1) - before[indices].mean(axis=1)

        # One-sided bounds at the SAME alpha in both directions -- we need
        # both bounds regardless of guard_direction, because we use one to
        # test for PASS and the other to test for FAIL; whichever isn't the
        # "active" side just tells us how much headroom/margin exists.
        lower_bound = np.percentile(bootstrap_lifts, self.alpha * 100)
        upper_bound = np.percentile(bootstrap_lifts, (1 - self.alpha) * 100)

        if self.guard_direction == "not_increase":
            # Bad = lift > margin. PASS if we're confident lift <= margin
            # (upper bound doesn't exceed margin). FAIL if we're confident
            # lift > margin (lower bound already exceeds margin).
            if upper_bound <= self.margin:
                status = "PASS"
            elif lower_bound > self.margin:
                status = "FAIL"
            else:
                status = "INCONCLUSIVE"
            threshold_desc = f"lift must not confidently exceed +{self.margin}"
        else:  # not_decrease
            # Bad = lift < -margin. PASS if confident lift >= -margin
            # (lower bound doesn't go below -margin). FAIL if confident
            # lift < -margin (upper bound is still below -margin).
            if lower_bound >= -self.margin:
                status = "PASS"
            elif upper_bound < -self.margin:
                status = "FAIL"
            else:
                status = "INCONCLUSIVE"
            threshold_desc = f"lift must not confidently fall below -{self.margin}"

        verdict_map = {
            "PASS": "PASS (NO MEANINGFUL REGRESSION)",
            "FAIL": "FAIL (REGRESSION DETECTED)",
            "INCONCLUSIVE": "REVIEW / RETEST (NOT ENOUGH EVIDENCE)",
        }

        risk_statement = {
            "PASS": f"We are >= {(1 - self.alpha) * 100:.0f}% confident this metric's change stays within "
                    f"the allowed margin ({threshold_desc}).",
            "FAIL": f"We are >= {(1 - self.alpha) * 100:.0f}% confident this metric regressed beyond "
                    f"the allowed margin ({threshold_desc}).",
            "INCONCLUSIVE": f"The sample can't yet confirm the change stays within margin, but also can't "
                             f"confirm a regression. Decision pending -- collect more data or review "
                             f"manually before shipping.",
        }[status]

        return {
            "metric": self.metric_name,
            "guard_direction": self.guard_direction,
            "margin": self.margin,
            "alpha": self.alpha,
            "sample_size": n,
            "observed_before": np.mean(before),
            "observed_after": np.mean(after),
            "observed_lift": observed_lift,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "status": status,
            "verdict": verdict_map[status],
            "risk_statement": risk_statement,
            "low_sample_warning": low_sample_warning(n),
        }

    @staticmethod
    def to_dataframe(results: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_executive_report().
        """
        return pd.DataFrame([{
            "Metric": results["metric"],
            "Status": results["status"],
            "Sample Size": results["sample_size"],
            "Baseline Mean": results["observed_before"],
            "After Mean": results["observed_after"],
            "Lift": results["observed_lift"],
            "Margin": results["margin"],
            "CI Lower": results["lower_bound"],
            "CI Upper": results["upper_bound"],
        }])

    def print_executive_report(self, results: dict):
        print("=" * 65)
        print(f" GUARDRAIL REPORT: {results['metric'].upper()}")
        print("=" * 65)
        print(f"  VERDICT         : {results['verdict']}")
        print(f"  Tested Sample   : {results['sample_size']} paired queries")
        print(f"  Baseline Mean   : {results['observed_before']:.4f}")
        print(f"  New Mean        : {results['observed_after']:.4f}")
        print(f"  Estimated Change: {results['observed_lift']:+.4f}  (margin: {results['margin']})")
        print(f"  One-sided bounds: [{results['lower_bound']:+.4f} to {results['upper_bound']:+.4f}] "
              f"at {(1 - results['alpha']) * 100:.0f}% confidence")
        print("-" * 65)
        print(f"  ANALYSIS: {results['risk_statement']}")
        if results.get("low_sample_warning"):
            print(f"  WARNING : {results['low_sample_warning']}")
        print("=" * 65 + "\n")


def _binom_pmf_all(n: int, p: float) -> np.ndarray:
    """Exact Binomial(n, p) pmf over k = 0..n, via log-gamma (no scipy)."""
    k = np.arange(n + 1)
    if p <= 0.0:
        return (k == 0).astype(float)
    if p >= 1.0:
        return (k == n).astype(float)
    log_comb = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    return np.exp(log_comb + k * math.log(p) + (n - k) * math.log1p(-p))


def exact_binomial_test(k: int, n: int, p: float) -> float:
    """Two-sided exact binomial test p-value (minimum-likelihood method,
    same convention as scipy.stats.binomtest)."""
    pmf = _binom_pmf_all(n, p)
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()))


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Exact (Clopper-Pearson) interval for a binomial proportion, by
    bisection on the exact tail probabilities."""
    a = 1.0 - confidence

    def upper_tail(p):  # P(X >= k | p), increasing in p
        return _binom_pmf_all(n, p)[k:].sum()

    def lower_tail(p):  # P(X <= k | p), decreasing in p
        return _binom_pmf_all(n, p)[:k + 1].sum()

    def bisect(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else bisect(upper_tail, a / 2, increasing=True)
    upper = 1.0 if k == n else bisect(lower_tail, a / 2, increasing=False)
    return lower, upper


class HarnessCalibrationSuite:
    """
    Validates the STATISTICAL MACHINERY itself, not any one result. There is
    no trained rule here to overfit, so there is nothing to "hold out";
    individual NO-GO or REVIEW results are the honest output of a correctly
    functioning test given limited evidence. What can and should be checked
    is whether the test's long-run behavior matches what it claims:

      1. Null calibration (an A/A test): with NO real effect, how often does
         the auditor say GO? The verdict rule is "CI bound clears zero" on a
         two-sided (1 - alpha_eff) CI, i.e. a ONE-sided test at alpha_eff/2,
         so the target rate is alpha_eff / 2 (2.5% at 95%), not alpha_eff.
      2. Power curve: with a real effect, how often is it detected?
      3. CI coverage: does the nominal 95% CI contain the true lift ~95% of
         the time?

    HOW "well_calibrated" IS DECIDED: an exact two-sided binomial test of the
    observed count against the target rate, at level `test_level` (default
    0.01), with a Clopper-Pearson interval reported alongside. The TAE-2026
    version used `abs(observed - expected) < 0.05` against a 5% target --
    a band of [0%, 10%] that passes a test with half or double its nominal
    false-positive rate. Use n_trials >= 2000 if you want the check to have
    real power to detect ~1-point miscalibration.

    correlation: latent Gaussian-copula correlation between a query's
    before/after outcome (see _correlated_binomial_pair). 0.0 = independent,
    the previous (and default) behavior.
    """
    def __init__(self, confidence_level: float = 0.95, n_bootstrap_iterations: int = 2000,
                 test_level: float = 0.01):
        self.confidence_level = confidence_level
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.test_level = test_level

    def _rate_report(self, count: int, n_trials: int, target: float) -> dict:
        lo, hi = clopper_pearson(count, n_trials, confidence=1 - self.test_level)
        p = exact_binomial_test(count, n_trials, target)
        return {"ci_low": lo, "ci_high": hi, "binomial_p_value": p,
                "well_calibrated": p >= self.test_level}

    def null_calibration(self, true_p: float = 0.70, n_questions: int = 200,
                          n_trials: int = 200, n_comparisons: int = 1,
                          goal: str = "increase", seed: int = 0,
                          correlation: float = 0.0) -> dict:
        """
        Simulates n_trials before/after pairs where the TRUE rate is identical
        (no real effect) and counts how often the auditor says GO. Decisions
        are read from the structured `status` field, never the display string
        (see the "GO" in "NO-GO" defect in the paper, Section 5).
        """
        rng = np.random.RandomState(seed)
        alpha_effective = (1 - self.confidence_level) / n_comparisons
        target = alpha_effective / 2.0
        false_positives = 0

        for t in range(n_trials):
            before, after = _correlated_binomial_pair(rng, true_p, true_p, n_questions, correlation)
            auditor = AIBootstrapAuditor("calibration_check", goal=goal,
                                          confidence_level=self.confidence_level,
                                          n_comparisons=n_comparisons)
            result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations,
                                   seed=seed * 100003 + t)
            if result["status"] == "GO":
                false_positives += 1

        observed_rate = false_positives / n_trials
        return {
            "n_trials": n_trials,
            "n_questions": n_questions,
            "n_bootstrap_iterations": self.n_bootstrap_iterations,
            "correlation": correlation,
            "expected_false_positive_rate": target,
            "observed_false_positive_rate": observed_rate,
            "n_false_positives": false_positives,
            **self._rate_report(false_positives, n_trials, target),
        }

    def power_curve(self, true_p_before: float = 0.70, effect_sizes=(0.02, 0.05, 0.08, 0.10),
                     sample_sizes=(100, 200, 500, 1000), n_trials: int = 100,
                     n_comparisons: int = 1, seed: int = 1, correlation: float = 0.0) -> list:
        """
        For each (effect_size, sample_size), simulates n_trials audits with
        that TRUE effect present and reports the fraction detected as GO,
        with a 95% Clopper-Pearson interval on that fraction.
        """
        rng = np.random.RandomState(seed)
        rows = []
        for n in sample_sizes:
            for effect in effect_sizes:
                true_p_after = min(true_p_before + effect, 1.0)
                detections = 0
                for _ in range(n_trials):
                    before, after = _correlated_binomial_pair(rng, true_p_before, true_p_after, n,
                                                              correlation)
                    auditor = AIBootstrapAuditor("power_check", goal="increase",
                                                  confidence_level=self.confidence_level,
                                                  n_comparisons=n_comparisons)
                    result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations,
                                           seed=int(rng.randint(0, 2**31 - 1)))
                    if result["status"] == "GO":
                        detections += 1
                lo, hi = clopper_pearson(detections, n_trials, confidence=0.95)
                rows.append({
                    "sample_size": n,
                    "true_effect_size": effect,
                    "n_trials": n_trials,
                    "detection_rate": detections / n_trials,
                    "ci_low": lo,
                    "ci_high": hi,
                })
        return rows

    def ci_coverage(self, true_p_before: float = 0.70, true_effect: float = 0.05,
                     n_questions: int = 200, n_trials: int = 200, seed: int = 2,
                     correlation: float = 0.0) -> dict:
        """
        Simulates n_trials audits with a KNOWN true lift and checks what
        fraction of the reported CIs contain it, tested exactly against the
        nominal confidence level.

        KNOWN LIMITATION: the auditor uses a PERCENTILE bootstrap interval,
        which is known to under-cover at small-to-moderate n (e.g. ~88% at
        n=10 and ~93-94% at n=50-200 for binary metrics in our runs). With
        enough trials this check WILL flag that -- correctly. That is the
        check working, not a bug in the check.
        """
        rng = np.random.RandomState(seed)
        true_p_after = min(true_p_before + true_effect, 1.0)
        true_lift = true_p_after - true_p_before
        covered = 0

        for t in range(n_trials):
            before, after = _correlated_binomial_pair(rng, true_p_before, true_p_after, n_questions,
                                                      correlation)
            auditor = AIBootstrapAuditor("coverage_check", goal="increase",
                                          confidence_level=self.confidence_level,
                                          n_comparisons=1)
            result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations,
                                   seed=seed * 100003 + t)
            if result["ci_lower"] <= true_lift <= result["ci_upper"]:
                covered += 1

        observed_coverage = covered / n_trials
        return {
            "n_trials": n_trials,
            "n_questions": n_questions,
            "n_bootstrap_iterations": self.n_bootstrap_iterations,
            "correlation": correlation,
            "true_effect": true_lift,
            "nominal_coverage": self.confidence_level,
            "observed_coverage": observed_coverage,
            **self._rate_report(covered, n_trials, self.confidence_level),
        }

    def print_full_report(self, null_result: dict, power_rows: list, coverage_result: dict):
        print("=" * 70)
        print(" HARNESS CALIBRATION REPORT")
        print(" (validates the TEST ITSELF, not any single audit result)")
        print("=" * 70)

        print("\n[1] Null calibration -- one-sided GO rate when NO real effect exists")
        status = "OK" if null_result["well_calibrated"] else "MISCALIBRATED"
        print(f"    Target: {null_result['expected_false_positive_rate']:.2%}   "
              f"Observed: {null_result['observed_false_positive_rate']:.2%} "
              f"[{null_result['ci_low']:.2%}, {null_result['ci_high']:.2%}]   "
              f"({null_result['n_false_positives']}/{null_result['n_trials']} trials, "
              f"exact binomial p={null_result['binomial_p_value']:.3f})   [{status}]")

        print("\n[2] Power curve -- detection rate when a REAL effect of this size exists")
        print(f"    {'n':>6} {'true effect':>12} {'detection rate':>16} {'95% CI':>16}")
        for row in power_rows:
            print(f"    {row['sample_size']:>6} {row['true_effect_size']:>12.0%} "
                  f"{row['detection_rate']:>16.0%}   [{row['ci_low']:.0%}, {row['ci_high']:.0%}]")
        print("    -> Read this BEFORE trusting a REVIEW as 'no effect exists.'")

        print("\n[3] CI coverage -- does the nominal CI contain the true lift at the stated rate?")
        status = "OK" if coverage_result["well_calibrated"] else "MISCALIBRATED"
        print(f"    Nominal: {coverage_result['nominal_coverage']:.1%}   "
              f"Observed: {coverage_result['observed_coverage']:.1%} "
              f"[{coverage_result['ci_low']:.1%}, {coverage_result['ci_high']:.1%}]   "
              f"(exact binomial p={coverage_result['binomial_p_value']:.3f})   [{status}]")
        print("=" * 70 + "\n")


def _abramowitz_stegun_erf(x: np.ndarray) -> np.ndarray:
    """
    Vectorized erf approximation (Abramowitz & Stegun 7.1.26, max abs error
    ~1.5e-7) -- used below to build correlated Bernoulli pairs without
    adding a scipy dependency to a project whose README explicitly advertises
    being dependency-light.
    """
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _abramowitz_stegun_erf(z / np.sqrt(2.0)))


def _correlated_binomial_pair(rng: np.random.RandomState, p_before: float, p_after: float,
                               n: int, correlation: float) -> tuple:
    """
    Generates n paired (before_i, after_i) Bernoulli draws with marginal
    rates p_before/p_after and a specified per-query correlation, via a
    Gaussian copula: draw (z_before, z_after) jointly normal with the
    requested correlation, then threshold each through the standard normal
    CDF against its own marginal rate. correlation=0 reduces exactly to two
    independent binomial draws (the harness's original, still-default
    behavior) -- this function is purely additive and does not change any
    previously reported result computed at the default correlation=0.

    RATIONALE: real paired evaluation data (the same benchmark queries run
    through two model versions) is typically positively correlated -- a
    query that is hard for one version tends to be hard for the other. That
    correlation is precisely what makes a *paired* bootstrap more powerful
    than an unpaired test on the same data, since it cancels out per-query
    difficulty rather than treating it as noise. Simulating before/after as
    independent (the prior, and still-default, behavior) is a legitimate
    worst-case assumption, but it is a conservative one: required-sample-
    size figures computed at correlation=0 are upper bounds on what a real,
    positively-correlated paired evaluation would need, not a general-
    purpose estimate. This was previously true of every SampleSizeOptimizer
    result but was not stated as an assumption anywhere in code or docs.
    """
    if correlation == 0.0:
        before = rng.binomial(1, p=p_before, size=n)
        after = rng.binomial(1, p=p_after, size=n)
        return before, after

    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    before = (_standard_normal_cdf(z[:, 0]) < p_before).astype(int)
    after = (_standard_normal_cdf(z[:, 1]) < p_after).astype(int)
    return before, after


class SampleSizeOptimizer:
    """
    Inverts the power curve: instead of "given n, what's my power?", answers
    "given a target power, what's the minimum n?" -- computed via the SAME
    simulation approach as HarnessCalibrationSuite, not a closed-form formula
    and not a hardcoded lookup table. Every number this class reports is
    something it actually simulated, not an assumption or extrapolation.

    WHY NOT JUST HARDCODE A TABLE: a plausible-sounding table like "n=150-200
    for a 15% effect, n=1500+ for a 2-3% effect" is exactly the kind of
    unverified claim this whole project has been built to avoid trusting
    without evidence. Effect size, true baseline rate, and n_comparisons all
    interact -- a table computed for one true_p_before or one n_comparisons
    value will not automatically hold for another. This class computes the
    actual answer for YOUR specific scenario, every time.
    """
    def __init__(self, confidence_level: float = 0.95, n_bootstrap_iterations: int = 1000):
        self.suite = HarnessCalibrationSuite(confidence_level=confidence_level,
                                              n_bootstrap_iterations=n_bootstrap_iterations)

    def _detection_rate_at_n(self, n: int, true_p_before: float, effect_size: float,
                              n_comparisons: int, n_trials: int, rng: np.random.RandomState,
                              correlation: float = 0.0) -> float:
        """
        Shared simulation core: runs n_trials audits at sample size n with
        the given true effect actually present, returns fraction detected
        as GO. Used by both find_required_n() (fixed grid) and
        find_required_n_adaptive() (exponential search + refinement), so
        the two methods can't silently drift out of sync with each other.

        correlation (default 0.0, backward-compatible with every previously
        reported result): the assumed per-query correlation between a
        query's before/after outcome, passed to _correlated_binomial_pair().
        0.0 simulates before/after as fully independent -- a conservative
        worst case that does NOT reflect typical paired evaluation data,
        where the same query tends to be hard (or easy) for both model
        versions. Required-n figures at correlation=0 should be read as
        upper bounds, not general-purpose estimates; pass a positive value
        here if you have an empirical estimate of your own data's per-query
        before/after correlation (e.g. from a pilot run) and want a
        realistic rather than worst-case required-n.
        """
        true_p_after = min(true_p_before + effect_size, 1.0)
        detections = 0
        for _ in range(n_trials):
            before, after = _correlated_binomial_pair(rng, true_p_before, true_p_after, n, correlation)
            auditor = AIBootstrapAuditor("n_search", goal="increase",
                                          confidence_level=self.suite.confidence_level,
                                          n_comparisons=n_comparisons)
            result = auditor.audit(before, after, n_iterations=self.suite.n_bootstrap_iterations)
            if result["status"] == "GO":  # structured field, never the display string
                detections += 1
        return detections / n_trials

    def find_required_n(self, true_p_before: float, effect_size: float,
                         target_power: float = 0.80, n_comparisons: int = 1,
                         candidate_sizes=(50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000),
                         n_trials: int = 50, seed: int = 3, correlation: float = 0.0) -> dict:
        """
        Simulates detection rate at each candidate sample size for a
        SPECIFIC true_p_before/effect_size/n_comparisons combination, and
        returns the smallest sample size that empirically reaches
        target_power. Returns the full curve too, so you can see the
        tradeoff, not just the cutoff.

        LIMITATION: only searches the exact sizes you give it. If the true
        required n is above your largest candidate, this returns
        required_n=None with no indication of how much further to look --
        see find_required_n_adaptive() for a version that keeps searching
        automatically.

        NOTE: n_trials controls how precise this estimate is. n_trials=50
        is fast enough to explore interactively but has real Monte Carlo
        noise (+/- ~10-15 percentage points on the detection rate estimate
        at that count) -- rerun with a higher n_trials (200+) before using
        the result to justify a real budget decision.

        correlation: see _detection_rate_at_n(). Default 0.0 is the
        conservative independent-samples assumption used by every
        previously reported figure from this class.
        """
        rng_seed = seed
        curve = []
        required_n = None

        for n in sorted(candidate_sizes):
            rng = np.random.RandomState(rng_seed)
            rng_seed += 1
            detection_rate = self._detection_rate_at_n(n, true_p_before, effect_size,
                                                         n_comparisons, n_trials, rng,
                                                         correlation=correlation)
            curve.append({"sample_size": n, "detection_rate": detection_rate})
            if required_n is None and detection_rate >= target_power:
                required_n = n

        return {
            "true_p_before": true_p_before,
            "effect_size": effect_size,
            "n_comparisons": n_comparisons,
            "target_power": target_power,
            "correlation": correlation,
            "required_n": required_n,  # None if no candidate size reached target_power
            "curve": curve,
            "n_trials_per_point": n_trials,
        }

    def find_required_n_adaptive(self, true_p_before: float, effect_size: float,
                                  target_power: float = 0.80, n_comparisons: int = 1,
                                  start_n: int = 50, max_n: int = 50000,
                                  growth_factor: float = 2.0, n_trials: int = 50,
                                  refine: bool = True, refine_steps: int = 6,
                                  diminishing_returns_threshold: float = 0.03,
                                  seed: int = 5, correlation: float = 0.0) -> dict:
        """
        Doesn't require you to guess a candidate range up front. Starts at
        start_n and keeps doubling (by growth_factor) until either:
          (a) detection rate reaches target_power -- search stops, or
          (b) n exceeds max_n -- search stops, reports hit_max_n=True,
              meaning this effect may not be economically detectable, or a
              higher max_n is needed, or
          (c) diminishing returns -- doubling the sample size is barely
              moving the detection rate (delta < diminishing_returns_threshold
              between consecutive doublings) while still well below target
              (< 70% of target_power). This usually means the effect is too
              small relative to the noise floor to be worth chasing with
              more data at all -- search stops early, reports
              diminishing_returns=True, instead of walking all the way to
              max_n for a result that was already predictable. Set
              diminishing_returns_threshold=0 to disable this check.

        If refine=True and a passing n was found, runs a binary search
        between the last failing size and the first passing size to
        tighten the estimate, rather than reporting the coarse doubled
        value (e.g. if 1500 fails and 3000 passes, refine narrows toward
        the actual crossover instead of stopping at 3000).

        Returns the same shape as find_required_n(), plus:
          - "search_path": the exponential-phase (n, detection_rate) pairs
          - "hit_max_n": True if max_n was reached without finding target_power
          - "diminishing_returns": True if the search stopped early because
            more data stopped meaningfully helping
        """
        rng_seed = seed
        search_path = []
        n = start_n
        last_failing_n = None
        first_passing_n = None
        first_passing_rate = None
        hit_max_n = False
        diminishing_returns = False
        previous_detection_rate = None

        # --- Phase 1: exponential search ---
        while True:
            rng = np.random.RandomState(rng_seed)
            rng_seed += 1
            detection_rate = self._detection_rate_at_n(n, true_p_before, effect_size,
                                                         n_comparisons, n_trials, rng,
                                                         correlation=correlation)
            search_path.append({"sample_size": n, "detection_rate": detection_rate})

            if detection_rate >= target_power:
                first_passing_n = n
                first_passing_rate = detection_rate
                break

            # Diminishing-returns check: only meaningful once we have at
            # least two points to compare, and only worth stopping early if
            # we're still clearly far from the target (otherwise a small
            # delta near the target is expected and fine -- that's normal
            # convergence, not a sign the effect is undetectable).
            if (previous_detection_rate is not None
                    and diminishing_returns_threshold > 0
                    and detection_rate < target_power * 0.70):
                delta = detection_rate - previous_detection_rate
                if delta < diminishing_returns_threshold:
                    diminishing_returns = True
                    last_failing_n = n
                    break

            previous_detection_rate = detection_rate
            last_failing_n = n
            next_n = int(n * growth_factor)
            if next_n > max_n:
                hit_max_n = True
                break
            n = next_n

        curve = list(search_path)
        required_n = first_passing_n

        # --- Phase 2: optional binary-search refinement between bounds ---
        # (skipped automatically if diminishing_returns/hit_max_n stopped us
        # before ever finding a passing n -- nothing to refine toward)
        if refine and first_passing_n is not None and last_failing_n is not None:
            lo, hi = last_failing_n, first_passing_n
            for _ in range(refine_steps):
                mid = (lo + hi) // 2
                if mid <= lo or mid >= hi:
                    break  # bounds have converged as tightly as integers allow
                rng = np.random.RandomState(rng_seed)
                rng_seed += 1
                mid_rate = self._detection_rate_at_n(mid, true_p_before, effect_size,
                                                       n_comparisons, n_trials, rng,
                                                       correlation=correlation)
                curve.append({"sample_size": mid, "detection_rate": mid_rate})
                if mid_rate >= target_power:
                    hi = mid
                    required_n = mid
                else:
                    lo = mid

        curve.sort(key=lambda row: row["sample_size"])

        return {
            "true_p_before": true_p_before,
            "effect_size": effect_size,
            "n_comparisons": n_comparisons,
            "target_power": target_power,
            "correlation": correlation,
            "required_n": required_n,
            "hit_max_n": hit_max_n,
            "diminishing_returns": diminishing_returns,
            "search_path": search_path,
            "curve": curve,
            "n_trials_per_point": n_trials,
        }

    def cost_optimal_plan(self, true_p_before: float, effect_size: float,
                           cost_per_query: float, target_power: float = 0.80,
                           n_comparisons: int = 1, **kwargs) -> dict:
        """
        Wraps find_required_n() with a direct cost translation, and flags
        the concrete wasted spend if you tested at a larger n than needed.
        """
        result = self.find_required_n(true_p_before, effect_size, target_power,
                                       n_comparisons, **kwargs)
        if result["required_n"] is not None:
            result["required_cost"] = result["required_n"] * cost_per_query
        else:
            result["required_cost"] = None
        for row in result["curve"]:
            row["cost"] = row["sample_size"] * cost_per_query
        return result

    @staticmethod
    def to_dataframe(result: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_search_report()'s
        curve table. One row per sample size tested during the search.
        """
        df = pd.DataFrame(result["curve"]).sort_values("sample_size").reset_index(drop=True)
        if result.get("required_n") is not None:
            df["is_required_n"] = df["sample_size"] == result["required_n"]
        return df

    def print_search_report(self, result: dict):
        print("=" * 70)
        print(f" SAMPLE SIZE SEARCH")
        print(f" True baseline: {result['true_p_before']:.0%}  |  "
              f"Target effect: +{result['effect_size']:.0%}  |  "
              f"n_comparisons: {result['n_comparisons']}  |  "
              f"Target power: {result['target_power']:.0%}")
        print("=" * 70)
        has_cost = "cost" in result["curve"][0] if result["curve"] else False
        header = f"    {'n':>6} {'detection rate':>16}"
        if has_cost:
            header += f" {'cost':>12}"
        print(header)
        for row in result["curve"]:
            marker = "  <-- required n" if row["sample_size"] == result["required_n"] else ""
            line = f"    {row['sample_size']:>6} {row['detection_rate']:>16.0%}"
            if has_cost:
                line += f" {row['cost']:>12,.2f}"
            print(line + marker)
        print("-" * 70)
        if result["required_n"] is not None:
            msg = f"  Minimum n for {result['target_power']:.0%} power: {result['required_n']}"
            if result.get("required_cost") is not None:
                msg += f"  (estimated cost: ${result['required_cost']:,.2f})"
            print(msg)
            print(f"  Testing at any n above this wastes budget without meaningfully")
            print(f"  improving your ability to detect this specific effect size.")
        else:
            if result.get("diminishing_returns"):
                last_n = result["search_path"][-1]["sample_size"]
                last_rate = result["search_path"][-1]["detection_rate"]
                print(f"  Stopped early at n={last_n} ({last_rate:.0%} power) -- doubling the sample")
                print(f"  size stopped meaningfully improving detection. This effect is likely too")
                print(f"  small relative to measurement noise to be worth chasing with more data.")
                print(f"  (Set diminishing_returns_threshold=0 to force the search to keep going to max_n.)")
            elif result.get("hit_max_n"):
                print(f"  Searched up to n={result['search_path'][-1]['sample_size']} and never reached "
                      f"{result['target_power']:.0%} power.")
                print(f"  This effect size may not be economically detectable at a sane sample size --")
                print(f"  consider a larger effect target, a lower target_power, or accept the cost of")
                print(f"  raising max_n further.")
            else:
                print(f"  None of the tested sample sizes reached {result['target_power']:.0%} power.")
                print(f"  Either this effect is too small to detect economically, or you")
                print(f"  need to test larger candidate sizes than were tried here.")
        print(f"\n  (Based on {result['n_trials_per_point']} simulated trials per sample size --")
        print(f"   increase n_trials for a more precise estimate before committing budget.)")
        print("=" * 70 + "\n")


# =====================================================================
# "AUDIT THE AUDITOR" -- live self-validation benchmarks
# =====================================================================
# These three functions are what a Validation Suite / "Audit the Auditor"
# view calls. Every number they return is computed fresh on each call --
# nothing here is a stored or hardcoded result. That's deliberate: a
# dashboard that prints a canned "22.6% -> 4.9%" string isn't a demo of
# anything, it's a prop. These functions exist so the printed numbers are
# always an honest, reproducible-on-demand measurement of this exact
# codebase, not a claim made once and then repeated.

def run_fwer_stress_test(n_comparisons: int = 5, n_trials: int = 100, n_per_metric: int = 200,
                          base_rate: float = 0.70, n_iterations: int = 500, seed: int = None) -> dict:
    """
    Simulates n_trials batches of n_comparisons TRUE-NULL metrics (before
    and after are drawn from the identical distribution -- no real effect
    exists) and measures the empirical false-positive rate of uncorrected
    vs. Bonferroni-corrected testing. A "false positive" here means at
    least one of the n_comparisons metrics in a batch was (incorrectly)
    called GO.
    """
    rng = np.random.RandomState(seed)
    uncorrected_false_batches = 0
    corrected_false_batches = 0

    for _ in range(n_trials):
        any_uncorrected_fp = False
        any_corrected_fp = False
        for _ in range(n_comparisons):
            trial_seed = rng.randint(0, 1_000_000)
            r = np.random.RandomState(trial_seed)
            before = r.binomial(1, base_rate, n_per_metric).astype(float)
            after = r.binomial(1, base_rate, n_per_metric).astype(float)  # true null

            r_uncorrected = AIBootstrapAuditor("m", "increase", n_comparisons=1).audit(
                before, after, n_iterations=n_iterations, seed=1)
            if r_uncorrected["status"] == "GO":
                any_uncorrected_fp = True

            r_corrected = AIBootstrapAuditor("m", "increase", n_comparisons=n_comparisons).audit(
                before, after, n_iterations=n_iterations, seed=1)
            if r_corrected["status"] == "GO":
                any_corrected_fp = True

        uncorrected_false_batches += int(any_uncorrected_fp)
        corrected_false_batches += int(any_corrected_fp)

    return {
        "n_trials": n_trials,
        "n_comparisons": n_comparisons,
        "uncorrected_false_batches": uncorrected_false_batches,
        "corrected_false_batches": corrected_false_batches,
        "uncorrected_fpr": uncorrected_false_batches / n_trials,
        "corrected_fpr": corrected_false_batches / n_trials,
        # GO is a one-sided test at alpha/2 = 0.025 per metric (see
        # HarnessCalibrationSuite), so the independent-metrics FWER is
        # 1 - (1 - 0.025)^k, not 1 - (1 - 0.05)^k.
        "theoretical_uncorrected_fwer": 1 - (1 - 0.025) ** n_comparisons,
    }


def run_edge_case_validation_suite() -> list:
    """
    Runs a fixed suite of known-bad inputs through AIBootstrapAuditor.audit()
    and confirms each is correctly rejected with a typed validation error
    (not a raw traceback, not a silent garbage verdict). This is a live
    re-run of the exact failure modes discovered and fixed during this
    project's development -- see MIN_PAIRED_OBSERVATIONS and the typed
    exception hierarchy above.
    """
    cases = [
        ("Empty input (n=0)", np.array([]), np.array([])),
        ("Single data point (n=1)", np.array([0.7]), np.array([0.8])),
        ("Tiny sample (n=5)", np.array([0.70, 0.71, 0.69, 0.72, 0.70]),
         np.array([0.80, 0.78, 0.81, 0.79, 0.82])),
        ("Missing values (NaN present)", np.array([0.7, np.nan, 0.72] * 4),
         np.array([0.8, 0.79, 0.81] * 4)),
    ]
    results = []
    for name, before, after in cases:
        try:
            AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=500)
            results.append({"case": name, "blocked": False, "exception_type": None, "message": None})
        except ValidationError as e:
            results.append({"case": name, "blocked": True, "exception_type": type(e).__name__, "message": str(e)})
    return results


def run_efficiency_benchmark(n_iterations_list=(1000, 5000, 10000), n: int = 500,
                              batch_metrics: int = 5, batch_n_iterations: int = 5000,
                              seed: int = None) -> dict:
    """Times the vectorized bootstrap at a few iteration counts on a
    fixed-size single-metric dataset, plus one realistic multi-metric
    batch, on whatever hardware this is actually running on."""
    rng = np.random.RandomState(seed)
    single_metric_timings = []
    for n_iter in n_iterations_list:
        before, after = rng.binomial(1, 0.7, n).astype(float), rng.binomial(1, 0.78, n).astype(float)
        t0 = time.perf_counter()
        AIBootstrapAuditor("m", "increase").audit(before, after, n_iterations=n_iter, seed=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        single_metric_timings.append({"n_iterations": n_iter, "n": n, "elapsed_ms": elapsed_ms})

    t0 = time.perf_counter()
    batch = AuditBatch(correction_method="benjamini_hochberg")
    for i in range(batch_metrics):
        b = rng.binomial(1, 0.7, 300).astype(float)
        a = rng.binomial(1, 0.75, 300).astype(float)
        batch.add_metric(f"m{i}", b, a, goal="increase")
    batch.run(n_iterations=batch_n_iterations, verbose=False, seed=1)
    batch_elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "single_metric_timings": single_metric_timings,
        "batch_metrics": batch_metrics,
        "batch_n_iterations": batch_n_iterations,
        "batch_elapsed_ms": batch_elapsed_ms,
    }


def run_detection_power_check(n_trials: int = 300, n_min: int = 3, n_max: int = 9,
                               base_rate: float = 0.70, seed: int = None) -> dict:
    """
    Honest, deliberately-not-headlined check: at very small n (below
    MIN_PAIRED_OBSERVATIONS), does an UNGATED bootstrap check (no sample
    floor) produce false positives on pure noise more often than nominal?
    Included for completeness and transparency, not because it's a strong
    differentiator -- see the docstring note in the caller. This function
    measures the naive rate only; this tool's own rate in this regime is
    always exactly 0%, by construction, since MIN_PAIRED_OBSERVATIONS
    refuses every case here categorically rather than probabilistically.
    """
    rng = np.random.RandomState(seed)
    naive_false_positives = 0
    for _ in range(n_trials):
        n = rng.randint(n_min, n_max + 1)
        trial_seed = rng.randint(0, 1_000_000)
        r = np.random.RandomState(trial_seed)
        before = r.binomial(1, base_rate, n).astype(float)
        after = r.binomial(1, base_rate, n).astype(float)  # true null
        idx = r.choice(n, size=(2000, n), replace=True)
        boot_lifts = after[idx].mean(axis=1) - before[idx].mean(axis=1)
        ci_lower = np.percentile(boot_lifts, 2.5)
        if ci_lower > 0:
            naive_false_positives += 1
    return {
        "n_trials": n_trials, "n_range": (n_min, n_max),
        "naive_false_positive_rate": naive_false_positives / n_trials,
        "naive_false_positives": naive_false_positives,
        "our_tool_false_positive_rate": 0.0,  # by construction, always refuses in this range
    }


if __name__ == "__main__":
    np.random.seed(42)

    # --- TEST 1: TEAM A (Accuracy - "Higher is Better"), single metric, no correction ---
    before_acc = np.random.binomial(1, p=0.70, size=500)
    after_acc  = np.random.binomial(1, p=0.78, size=500)

    accuracy_auditor = AIBootstrapAuditor(metric_name="RAG LLM Accuracy", goal="increase")
    accuracy_report = accuracy_auditor.audit(before_acc, after_acc)
    accuracy_auditor.print_executive_report(accuracy_report)

    # --- TEST 2: TEAM B (Latency - "Lower is Better"), single metric, no correction ---
    before_lat = np.random.normal(loc=1200, scale=300, size=150)
    before_lat = np.clip(before_lat, 100, None)
    after_lat  = np.random.normal(loc=950, scale=250, size=150)
    after_lat  = np.clip(after_lat, 100, None)

    latency_auditor = AIBootstrapAuditor(metric_name="API Latency (ms)", goal="decrease")
    latency_report = latency_auditor.audit(before_lat, after_lat)
    latency_auditor.print_executive_report(latency_report)

    # --- TEST 3: Demonstrating the multiple-comparisons correction ---
    # Simulate a review cycle where 5 teams each test their own metric at once.
    # Same accuracy data as Team A, but now explicitly flagged as 1-of-5
    # simultaneous comparisons.
    print("### Same Team A result, but run as 1 of 5 simultaneous metrics in one review cycle: ###\n")
    accuracy_auditor_corrected = AIBootstrapAuditor(
        metric_name="RAG LLM Accuracy", goal="increase", n_comparisons=5
    )
    accuracy_report_corrected = accuracy_auditor_corrected.audit(before_acc, after_acc)
    accuracy_auditor_corrected.print_executive_report(accuracy_report_corrected)

    # --- TEST 4: AuditBatch -- the intended real-world usage ---
    # A review cycle with 5 metrics: the two above, plus 3 more where one is
    # deliberately borderline, to show the correction actually change a
    # verdict rather than just widening a CI that was never close to zero.
    print("### AuditBatch: full review cycle, correction applied automatically ###\n")

    np.random.seed(7)
    # A borderline metric: small true effect, small sample -- exactly the
    # kind of result that can look "significant" uncorrected but shouldn't
    # survive correction when it's one of several simultaneous tests.
    before_borderline = np.random.binomial(1, p=0.70, size=80)
    after_borderline = np.random.binomial(1, p=0.74, size=80)

    before_c = np.random.normal(loc=500, scale=80, size=200)
    after_c = np.random.normal(loc=470, scale=80, size=200)

    before_d = np.random.binomial(1, p=0.60, size=300)
    after_d = np.random.binomial(1, p=0.60, size=300)  # genuinely no effect

    batch = AuditBatch(confidence_level=0.95)
    batch.add_metric("RAG LLM Accuracy", before_acc, after_acc, goal="increase")
    batch.add_metric("API Latency (ms)", before_lat, after_lat, goal="decrease")
    batch.add_metric("Borderline Feature Flag", before_borderline, after_borderline, goal="increase")
    batch.add_metric("Support Response Time (s)", before_c, after_c, goal="decrease")
    batch.add_metric("Unrelated Control Metric", before_d, after_d, goal="increase")

    batch_results = batch.run(verbose=False)  # suppress individual reports, just show summary
    batch.print_batch_summary(batch_results)

    # --- TEST 5: GuardrailAuditor -- the "prompt got wordier" scenario ---
    # Prompt change makes accuracy genuinely better (core value metric),
    # but also makes responses genuinely longer and slightly slower
    # (guardrail metrics) -- exactly the tradeoff described in the prompt.
    print("### GuardrailAuditor: catching a real, moderate regression that a ###")
    print("### Bonferroni-style stricter test would likely have missed      ###\n")

    np.random.seed(11)
    before_tokens = np.random.normal(loc=180, scale=40, size=250)
    before_tokens = np.clip(before_tokens, 10, None)
    # A real, moderate increase (not huge) -- the "wordier answers" effect
    after_tokens = np.random.normal(loc=216, scale=45, size=250)
    after_tokens = np.clip(after_tokens, 10, None)

    token_guard = GuardrailAuditor(
        metric_name="Output Token Count",
        guard_direction="not_increase",
        margin=0,       # any real increase counts as a regression here
        alpha=0.05,
    )
    token_result = token_guard.audit(before_tokens, after_tokens)
    token_guard.print_executive_report(token_result)

    # A borderline/small-sample case, deliberately, to show INCONCLUSIVE
    # rather than a silent PASS when evidence is genuinely insufficient.
    np.random.seed(22)
    before_latency_small = np.random.normal(loc=800, scale=150, size=25)
    after_latency_small = np.random.normal(loc=830, scale=150, size=25)

    latency_guard = GuardrailAuditor(
        metric_name="Generation Latency p95 (ms, small sample)",
        guard_direction="not_increase",
        margin=20,
        alpha=0.05,
    )
    latency_result = latency_guard.audit(before_latency_small, after_latency_small)
    latency_guard.print_executive_report(latency_result)

    # --- TEST 6: The full diagram, wired together in one batch ---
    # Core value metrics -> Benjamini-Hochberg (find real quality wins)
    # Guardrail/cost metrics -> GuardrailAuditor (catch stealth regressions)
    # One unified report, two different statistical treatments underneath.
    print("### Full integration: core metrics (BH) + guardrails (GuardrailAuditor), one report ###\n")

    np.random.seed(99)
    before_int_acc = np.random.binomial(1, p=0.72, size=400)
    after_int_acc  = np.random.binomial(1, p=0.80, size=400)   # real accuracy win

    before_int_rel = np.random.normal(loc=0.70, scale=0.15, size=400)
    after_int_rel  = np.random.normal(loc=0.76, scale=0.15, size=400)  # real relevance win

    before_int_tokens = np.random.normal(loc=180, scale=40, size=400)
    before_int_tokens = np.clip(before_int_tokens, 10, None)
    after_int_tokens  = np.random.normal(loc=214, scale=42, size=400)  # real token regression
    after_int_tokens  = np.clip(after_int_tokens, 10, None)

    before_int_latency = np.random.normal(loc=900, scale=180, size=400)
    after_int_latency  = np.random.normal(loc=915, scale=180, size=400)  # small, tolerable drift

    integrated_batch = AuditBatch(confidence_level=0.95, correction_method="benjamini_hochberg")
    integrated_batch.add_metric("intent_classification_accuracy", before_int_acc, after_int_acc, goal="increase")
    integrated_batch.add_metric("response_relevance_llm_judge", before_int_rel, after_int_rel, goal="increase")
    integrated_batch.add_guardrail("output_token_count", before_int_tokens, after_int_tokens,
                                    guard_direction="not_increase", margin=0)
    integrated_batch.add_guardrail("generation_latency_p95", before_int_latency, after_int_latency,
                                    guard_direction="not_increase", margin=50)  # allow up to 50ms drift

    integrated_results = integrated_batch.run(verbose=False)
    integrated_batch.print_batch_summary(integrated_results)

    # --- TEST 7: Calibrating the harness itself ---
    print("### HarnessCalibrationSuite: validating the test's own long-run behavior ###\n")

    suite = HarnessCalibrationSuite(confidence_level=0.95, n_bootstrap_iterations=1000)

    null_result = suite.null_calibration(true_p=0.70, n_questions=200, n_trials=100, n_comparisons=1)
    power_rows = suite.power_curve(
        true_p_before=0.70,
        effect_sizes=(0.05, 0.08),
        sample_sizes=(200, 500),
        n_trials=40,
        n_comparisons=1,
    )
    coverage_result = suite.ci_coverage(true_p_before=0.70, true_effect=0.05, n_questions=200, n_trials=100)

    suite.print_full_report(null_result, power_rows, coverage_result)

    # --- TEST 8: Sample size optimizer -- checking the earlier unverified claims ---
    print("### SampleSizeOptimizer: computing (not assuming) required sample sizes ###\n")

    optimizer = SampleSizeOptimizer(confidence_level=0.95, n_bootstrap_iterations=800)

    # Claim to check: "n=150-200 is enough for a 15% swing-for-the-fences effect"
    result_big_effect = optimizer.find_required_n(
        true_p_before=0.70, effect_size=0.15, target_power=0.80,
        candidate_sizes=(50, 100, 150, 200, 300), n_trials=200,
    )
    optimizer.print_search_report(result_big_effect)

    # Claim to check: "n=1500+ needed for a 2-3% micro-optimization"
    # Previously (fixed grid up to 3000) this came back "None of the tested
    # sizes reached target power" -- unresolved. Adaptive search removes the
    # need to guess the range: it keeps doubling until it finds an answer
    # or hits max_n.
    result_small_effect = optimizer.find_required_n_adaptive(
        true_p_before=0.70, effect_size=0.03,
        target_power=0.80, start_n=500, max_n=50000, n_trials=200,
    )
    optimizer.print_search_report(result_small_effect)
    if result_small_effect["required_n"] is not None:
        cost = result_small_effect["required_n"] * 0.02  # 2 cents/LLM-judge call, example
        print(f"  At $0.02/query, that's an estimated ${cost:,.2f} to reliably detect this effect.\n")
