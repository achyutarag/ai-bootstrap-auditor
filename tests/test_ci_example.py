"""The CI release gate printed in the paper (examples/test_release_gate.py) must run as
printed. On the real Mistral v0.1 -> v0.2 GSM8K data it must FAIL the build, for the
right reason: accuracy improved but answers grew far beyond the 2.9-word margin."""
import os
import runpy

import numpy as np
import pandas as pd
import pytest

from ai_bootstrap_auditor import AuditBatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "test_release_gate.py")
PAIRED = os.path.join(ROOT, "data", "mistral_v01_v02_paired.csv")


def write_eval_csv(directory, acc_old, acc_new, words_old, words_new):
    pd.DataFrame({"acc_old": acc_old, "acc_new": acc_new,
                  "words_old": words_old, "words_new": words_new}).to_csv(
        os.path.join(directory, "eval_results.csv"), index=False)


@pytest.mark.skipif(not os.path.exists(PAIRED), reason="needs data/mistral_v01_v02_paired.csv")
def test_paper_snippet_blocks_the_real_mistral_release(tmp_path, monkeypatch):
    data = pd.read_csv(PAIRED)
    acc = data[(data.group == "gsm8k") & (data.metric == "acc")]
    words = data[(data.group == "gsm8k") & (data.metric == "response_words")]
    rows = acc.merge(words, on="question_hash", suffixes=("_acc", "_words"))  # same question per row
    assert len(rows) == 1319
    write_eval_csv(tmp_path, rows.before_acc, rows.after_acc, rows.before_words, rows.after_words)
    monkeypatch.chdir(tmp_path)
    gate_test = runpy.run_path(EXAMPLE)["test_release_gate"]
    with pytest.raises(AssertionError, match="guardrail failed: answer_words"):
        gate_test()


def test_paper_snippet_passes_a_clean_release(tmp_path, monkeypatch):
    r = np.random.RandomState(0)
    acc_old = r.binomial(1, 0.60, 2000)
    acc_new = np.where(r.rand(2000) < 0.15, 1, acc_old)    # real gain, many discordant pairs
    words_old = r.normal(30, 5, 2000)
    words_new = words_old + r.normal(0, 1, 2000)            # length unchanged
    write_eval_csv(tmp_path, acc_old, acc_new, words_old, words_new)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(EXAMPLE)["test_release_gate"]()          # must not raise


def test_decision_rule_order():
    batch = AuditBatch()
    batch.add_metric("flat", np.r_[np.ones(50), np.zeros(50)], np.r_[np.ones(50), np.zeros(50)])
    batch.run(n_iterations=500, seed=0)
    assert batch.decision() == {"decision": "HOLD", "reasons": ["not enough evidence: flat"]}
    with pytest.raises(RuntimeError):
        AuditBatch().decision()
