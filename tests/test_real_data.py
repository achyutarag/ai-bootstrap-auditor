"""Sanity checks on the real-data case study input (data/mistral_v01_v02_paired.csv):
the per-question pairing must reproduce the scores the Open LLM Leaderboard (v1)
published for Mistral-7B-Instruct v0.1 and v0.2."""
import os

import pandas as pd
import pytest

PAIRED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "mistral_v01_v02_paired.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(PAIRED), reason="run scripts/build_paired_dataset.py first")

# Published leaderboard scores (%), v0.1 and v0.2. MMLU is excluded: the
# leaderboard macro-averages subjects, our pooled per-question mean does not.
PUBLISHED = {
    ("arc_challenge", "acc_norm"): (54.52, 63.14),
    ("hellaswag", "acc_norm"): (75.63, 84.88),
    ("truthfulqa_mc", "mc2"): (56.28, 68.26),
    ("winogrande", "acc"): (73.72, 77.19),
    ("gsm8k", "acc"): (14.25, 40.03),
}


@pytest.fixture(scope="module")
def paired():
    return pd.read_csv(PAIRED)


@pytest.mark.parametrize("key, published", PUBLISHED.items())
def test_pairing_reproduces_published_scores(paired, key, published):
    rows = paired[(paired.group == key[0]) & (paired.metric == key[1])]
    # every question is kept (repeated texts are distinct items), so the means
    # match the published 2-decimal scores up to rounding
    assert rows.before.mean() * 100 == pytest.approx(published[0], abs=0.006)
    assert rows.after.mean() * 100 == pytest.approx(published[1], abs=0.006)


def test_every_question_is_paired_once(paired):
    assert not paired.duplicated(["task", "metric", "question_hash"]).any()
    assert paired[["before", "after"]].notna().all().all()
    assert paired[paired.group == "mmlu"].task.nunique() == 57
