"""
Real-data case study: Mistral-7B-Instruct v0.1 -> v0.2 on the Open LLM
Leaderboard v1 tasks (per-question results; see scripts/fetch_real_data.py and
scripts/build_paired_dataset.py). Answers reviewer xSFC-W2.

    python scripts/real_data_case_study.py [--fast]

  A. Release review: one batch, 5 core accuracy metrics + 2 guardrails.
  B. Measured per-question before/after correlation, and what it does to the
     data needed (closed-form paired sample size, 80% power, one-sided 2.5%).
  C. Small evals: each of the 57 MMLU subjects (n = 100-1,534) audited as its
     own metric -- naive "number went up" vs. the harness, batch-corrected.
  D. Subsampling: treat each full task as the population, draw many small
     evals, and compare how often a naive read and the harness get it wrong.
"""
import argparse
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ai_bootstrap_auditor import AIBootstrapAuditor, AuditBatch, clopper_pearson  # noqa: E402

PAIRED = os.path.join(ROOT, "data", "mistral_v01_v02_paired.csv")
CORE = [("arc_challenge", "acc_norm", "ARC-Challenge"), ("hellaswag", "acc_norm", "HellaSwag"),
        ("mmlu", "acc", "MMLU"), ("winogrande", "acc", "Winogrande"), ("gsm8k", "acc", "GSM8K")]
Z_975, Z_80 = 1.959964, 0.841621


def series(df, group, metric):
    rows = df[(df.group == group) & (df.metric == metric)]
    return rows.before.to_numpy(float), rows.after.to_numpy(float)


def header(title):
    print("\n" + "=" * 86 + f"\n {title}\n" + "=" * 86)


def section_a(df, B):
    header(f"A. Release review: Mistral-7B-Instruct v0.1 -> v0.2 (B={B}, BH, seed=1)")
    batch = AuditBatch(correction_method="benjamini_hochberg")
    for group, metric, label in CORE:
        batch.add_metric(label, *series(df, group, metric))
    tq_b, tq_a = series(df, "truthfulqa_mc", "mc2")
    batch.add_guardrail("TruthfulQA mc2 (must not drop > 0.02)", tq_b, tq_a,
                        guard_direction="not_decrease", margin=0.02)
    len_b, len_a = series(df, "gsm8k", "response_words")
    len_margin = round(0.10 * len_b.mean(), 1)
    batch.add_guardrail(f"GSM8K answer length, words (must not grow > {len_margin})", len_b, len_a,
                        guard_direction="not_increase", margin=len_margin)
    results = batch.run(n_iterations=B, verbose=False, seed=1)
    print(f"  {'metric':<16}{'n':>7}{'v0.1':>8}{'v0.2':>8}{'lift':>8}   95% CI              verdict")
    for r in results:
        print(f"  {r['metric']:<16}{r['sample_size']:>7}{r['observed_before']:>8.3f}{r['observed_after']:>8.3f}"
              f"{r['observed_lift']:>+8.3f}   [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]   {r['status']}")
    for g in batch._last_guardrail_results:
        print(f"  GUARDRAIL {g['metric']}: v0.1 {g['observed_before']:.2f} -> v0.2 {g['observed_after']:.2f} "
              f"(lift {g['observed_lift']:+.2f}, one-sided bounds [{g['lower_bound']:+.2f}, "
              f"{g['upper_bound']:+.2f}]) -> {g['status']}")


def section_b(df):
    header("B. Measured per-question correlation (v0.1 vs v0.2) and data needed for a +3pt effect")
    print(f"  {'task':<16}{'n':>7}{'Pearson r':>11}{'n needed, r=0':>16}{'n needed, measured r':>23}{'saving':>9}")
    for group, metric, label in CORE + [("truthfulqa_mc", "mc2", "TruthfulQA mc2")]:
        b, a = series(df, group, metric)
        r = float(np.corrcoef(b, a)[0, 1])
        p1 = b.mean()
        sd_b, sd_a = b.std(ddof=1), a.std(ddof=1)
        delta = 0.03

        def n_needed(rho):
            return (Z_975 + Z_80) ** 2 * (sd_b ** 2 + sd_a ** 2 - 2 * rho * sd_b * sd_a) / delta ** 2

        n0, nr = n_needed(0.0), n_needed(r)
        print(f"  {label:<16}{len(b):>7}{r:>11.2f}{n0:>16,.0f}{nr:>23,.0f}{1 - nr / n0:>9.0%}")
    print("  (closed-form paired-difference sample size using each task's observed per-question\n"
          "   variances; 'r=0' is the independence assumption the TAE-2026 planner used)")


def section_c(df, B):
    header(f"C. 57 MMLU subjects as separate small evals (B={B}, seed=1)")
    subjects = df[df.group == "mmlu"].groupby("task")
    sizes = subjects.size()
    print(f"  subject sizes: min {sizes.min()}, median {int(sizes.median())}, max {sizes.max()}")
    naive_up = int((subjects.after.mean() > subjects.before.mean()).sum())
    naive_down = int((subjects.after.mean() < subjects.before.mean()).sum())
    print(f"  naive read ('number went up'): {naive_up} up, {naive_down} down, "
          f"{len(sizes) - naive_up - naive_down} unchanged")
    single = {"GO": 0, "REVIEW": 0, "NO-GO": 0}
    for task, rows in subjects:
        s = AIBootstrapAuditor(task).audit(rows.before.to_numpy(float), rows.after.to_numpy(float),
                                           n_iterations=B, seed=1)["status"]
        single[s] += 1
    print(f"  harness, each subject alone (no correction): {single}")
    for method in ("benjamini_hochberg", "bonferroni"):
        batch = AuditBatch(correction_method=method)
        for task, rows in subjects:
            batch.add_metric(task, rows.before.to_numpy(float), rows.after.to_numpy(float))
        counts = pd.Series([r["status"] for r in batch.run(n_iterations=B, verbose=False, seed=1)]).value_counts()
        print(f"  harness, all 57 as one batch ({method}): {counts.to_dict()}")


def section_d(df, B, reps):
    header(f"D. Small evals drawn from the full task ({reps} draws per size, B={B})")
    print("  wrong direction = verdict/read says v0.2 is WORSE, although on the full task it is better\n")
    print(f"  {'task':<12}{'full lift':>10}{'n':>6}{'naive: up':>11}{'naive: wrong dir':>18}"
          f"{'harness GO':>12}{'REVIEW':>9}{'NO-GO (wrong)':>15}")
    rng = np.random.RandomState(2026)
    for group, metric, label in [("winogrande", "acc", "Winogrande"), ("mmlu", "acc", "MMLU")]:
        b_all, a_all = series(df, group, metric)
        for n in (50, 100, 200, 500):
            naive_up = naive_wrong = go = review = nogo = 0
            for i in range(reps):
                idx = rng.choice(len(b_all), size=n, replace=False)
                b, a = b_all[idx], a_all[idx]
                lift = a.mean() - b.mean()
                naive_up += lift > 0
                naive_wrong += lift < 0
                status = AIBootstrapAuditor(label).audit(b, a, n_iterations=B, seed=i)["status"]
                go += status == "GO"
                review += status == "REVIEW"
                nogo += status == "NO-GO"
            pct = lambda k: f"{k / reps:.0%}"
            print(f"  {label:<12}{a_all.mean() - b_all.mean():>+10.3f}{n:>6}{pct(naive_up):>11}"
                  f"{pct(naive_wrong):>18}{pct(go):>12}{pct(review):>9}{pct(nogo):>15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    B, reps = (1000, 200) if args.fast else (4000, 1000)
    data = pd.read_csv(PAIRED)
    n_questions = (data.metric != "response_words").sum()  # answer length re-measures GSM8K items
    print(f"data: {PAIRED} ({n_questions:,} paired questions; {len(data):,} rows incl. GSM8K answer length)")
    section_a(data, B)
    section_b(data)
    section_c(data, B)
    section_d(data, min(B, 1000), reps)
