# AI Bootstrap Auditor


**Is this AI metric change a real win, or just noise?**

**Live app:** https://ai-bootstrap-auditor.streamlit.app (no install needed)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23524033.svg)](https://doi.org/10.5281/zenodo.23524033)

A statistical auditing tool for AI/ML evaluation results. Paste before/after numbers in, get a plain-English verdict — backed by paired bootstrap resampling implemented from scratch, not a black-box stats library — instead of a raw p-value nobody on a cross-functional team can interpret.

---

## The problem this solves

Teams ship AI changes on evidence that doesn't hold up. Run an eval on 50 queries, see a small lift, and ship — without realizing that lift could be nothing more than random sampling noise. This tool exists to catch that, before it costs real money.

While building it, that exact failure mode was found *inside the tool itself*: a single data point could produce a confident "GO" verdict with "0% chance of noise." That bug — how it was found, fixed, and verified — is as much a part of this project as the statistics are. See [`ai_bootstrap_auditor.py`](./ai_bootstrap_auditor.py) for the full validation logic and typed exception hierarchy that now prevents it.

## What's in the app

| Tab | What it does |
|---|---|
| **Single Metric Audit** | Is one metric's before/after change real or noise? Live GO / REVIEW-RETEST / NO-GO verdict. |
| **Guardrail Check** | Did a cost/risk metric (latency, error rate) regress beyond an acceptable margin? PASS / FAIL / REVIEW-RETEST, tuned to be more sensitive than a standard win/loss check — missing a real regression is the expensive mistake here. |
| **Batch Review** | Audit several core metrics and guardrails together in one pass, with automatic Bonferroni or Benjamini-Hochberg correction for multiple comparisons. Includes an overall GO/HOLD/NO-GO recommendation banner synthesizing every result. |
| **Sample Size Planner** | How much data do you actually need before testing? Adaptive required-*n* search, plus a "Translate to Budget" step that turns that into a real dollar figure. |
| **Harness Calibration** *(Advanced)* | Validates the test's own long-run behavior — false-positive rate under the null, and confidence-interval coverage — rather than any single result. |
| **Validation Suite** | *"Audit the Auditor."* Re-runs this tool's own statistical claims live, on demand: a multi-metric false-positive stress test with fresh randomized data every click, a re-run of the exact edge cases that caused the original bug, and a live compute-efficiency benchmark. Nothing here is a stored or hardcoded result. |

## Why this is more than a p-value calculator

- **Refuses to answer below a sample floor.** Fewer than 10 paired observations, or any missing/non-numeric/infinite values, and the tool raises a typed exception (`MinimumSampleFloorError`, `DataIntegrityError`) instead of guessing. This is a hard guarantee, not a soft warning.
- **Three-state verdicts, not two.** A confidence interval that straddles zero means *not enough evidence* — a genuinely different situation from one that's confidently on the bad side. Collapsing both into one "NO-GO" bucket hides that distinction; this tool doesn't.
- **Guardrails and core metrics are routed differently, on purpose.** Applying stricter multiple-comparisons correction to guardrails makes real regressions *harder* to catch — the opposite of what you want from a safety check. Guardrails are evaluated independently.
- **Reproducible by default.** Every audit accepts an optional `seed`; the app defaults to a fixed seed with an explicit "re-roll" control, so a result can be reproduced on demand rather than drifting between clicks.
- **Self-validating.** The Validation Suite tab re-tests the tool's own false-positive rate and edge-case handling live, rather than asking anyone to trust a claim made once in a README.

## Project structure

```
ai_bootstrap_auditor.py   Core statistics engine — zero Streamlit dependency, fully testable/importable on its own.
                          AIBootstrapAuditor, GuardrailAuditor, AuditBatch, SampleSizeOptimizer,
                          HarnessCalibrationSuite, typed exception hierarchy, and the Validation
                          Suite's live self-test functions (run_fwer_stress_test, etc.)

audit_toolkit_app.py      Streamlit UI — fast-path tabs (Single Metric Audit, Guardrail Check,
                          Batch Review), shared input validation, and the app entry point.

slow_path_cache.py        Streamlit UI — slow/cached tabs (Sample Size Planner, Harness
                          Calibration, Validation Suite). Separated from the fast-path tabs
                          because these are button-triggered and st.cache_data-backed rather
                          than recomputed on every keystroke.
```

## Running it locally

```bash
pip install -r requirements.txt
streamlit run audit_toolkit_app.py
```

## Tests and reproducing the paper

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests            # regression, metamorphic, and calibration tests (~1 min)
python reproduce_paper.py         # every reported number, fast settings
python reproduce_paper.py --full  # paper-grade trial counts (slow)
```

`tests/test_regressions.py` checks each defect reported in the paper against both the fixed code and the
original TAE-2026 submission (`legacy/ai_bootstrap_auditor_tae2026.py`), so each test is shown to catch the
defect, not only to pass on the fix. `tests/test_metamorphic.py` holds invariants that need no ground truth
(a batch of one equals a single audit; correction never demotes a NO-GO or adds a GO; goal-mirror and
unit-change invariance).

### Use as a CI release gate

`examples/test_release_gate.py` is a pytest test that fails the build unless a batch of metrics and
guardrails is a clean GO; `AuditBatch.decision()` returns `GO` / `HOLD` / `NO-GO` plus the reasons
(the app's summary banner uses the same rule). The example itself is tested in
`tests/test_ci_example.py`: on the real Mistral GSM8K data it blocks the release because answers grew
far beyond the 2.9-word guardrail.

### Real-data case study (Mistral-7B-Instruct v0.1 → v0.2)

```bash
python scripts/fetch_real_data.py        # ~190 MB of public per-question results -> data/raw/ (git-ignored)
python scripts/build_paired_dataset.py   # -> data/mistral_v01_v02_paired.csv (hashes + scores, no benchmark text)
python scripts/real_data_case_study.py   # release review, measured correlations, small-eval analyses
```

The paired CSV ships with the repo, so the app's "REAL:" examples and the *Load a real release review*
batch work without downloading anything. Source: archived Open LLM Leaderboard v1 details on Hugging Face
(`open-llm-leaderboard-old/details_mistralai__Mistral-7B-Instruct-v0.{1,2}`).

Correction defaults: the library's `AuditBatch` defaults to Bonferroni; the app's Batch Review tab defaults
to Benjamini-Hochberg. Both treat GO as a one-sided test at alpha/2 (2.5% at 95% confidence).

## A note on rigor

This project was built, then deliberately stress-tested against its own claims — small/empty/malformed inputs, repeated correction-method comparisons, and a live false-positive-rate simulation — rather than trusted on first pass. Several real bugs were found and fixed this way, most notably the one described above. The goal was to hold this tool to the same standard it holds the AI systems it's meant to evaluate.
