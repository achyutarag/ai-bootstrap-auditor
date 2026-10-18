"""
Builds data/mistral_v01_v02_paired.csv from the raw leaderboard details
fetched by scripts/fetch_real_data.py. One row per (task, question), with the
v0.1 ("before") and v0.2 ("after") per-question score, matched across versions
by a hash of the question text.

    python scripts/build_paired_dataset.py

Per-question scores follow the Open LLM Leaderboard v1 conventions:
ARC / HellaSwag acc_norm, MMLU / Winogrande / GSM8K acc, TruthfulQA mc2.
Plus one cost proxy: GSM8K response length in whitespace-separated words.
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "mistral_v01_v02_paired.csv")

SCORE_COLUMN = {"arc_challenge": "acc_norm", "hellaswag": "acc_norm", "winogrande": "acc",
                "gsm8k": "acc", "truthfulqa_mc": "mc2"}


def per_question(version: str, task: str, column: str) -> pd.Series:
    df = pd.read_parquet(os.path.join(RAW, f"mistral-{version}", f"{task}.parquet"))
    # v0.2 files lack the harness's 'hashes' column, so match on the question
    # text itself ('example'), shortened to a SHA-1 prefix for the CSV.
    # Some texts repeat within a task but are DIFFERENT items (different
    # choices/gold, e.g. 22 HellaSwag contexts), so the key also carries the
    # occurrence number: "<hash>#0", "<hash>#1". That is only a valid match if
    # both versions list items in the same order, which pair() asserts.
    text_hash = df["example"].map(lambda text: hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16])
    key = text_hash + "#" + text_hash.groupby(text_hash).cumcount().astype(str)
    if column == "response_words":
        values = df["predictions"].map(lambda p: len(str(p[0]).split()))
    elif column in df.columns:  # v0.1 layout: one column per metric
        values = df[column].astype(float)
    else:                       # v0.2 layout: metrics nested in a dict column
        values = df["metrics"].map(lambda m: float(m[column]))
    s = pd.Series(values.to_numpy(), index=key.to_numpy())
    assert not s.index.has_duplicates, f"{version}/{task}: question keys are not unique"
    return s


def pair(task: str, column: str, metric: str, group: str) -> pd.DataFrame:
    before, after = per_question("v0.1", task, column), per_question("v0.2", task, column)
    # Alignment check: the two runs must list the same questions in the same
    # order (verified for every task); otherwise occurrence-numbered keys
    # could pair different items and silently destroy the paired correlation.
    assert list(before.index) == list(after.index), f"{task}: question order differs between versions"
    common = before.index.intersection(after.index)
    unmatched = len(before) + len(after) - 2 * len(common)
    if unmatched:
        print(f"  {task}: {unmatched} question(s) not present in both versions, dropped")
    return pd.DataFrame({"group": group, "task": task, "metric": metric, "question_hash": common,
                         "before": before[common].to_numpy(), "after": after[common].to_numpy()})


if __name__ == "__main__":
    frames = [pair(task, col, col, task) for task, col in SCORE_COLUMN.items()]
    frames.append(pair("gsm8k", "response_words", "response_words", "gsm8k"))
    for path in sorted(glob.glob(os.path.join(RAW, "mistral-v0.1", "hendrycksTest-*.parquet"))):
        task = os.path.basename(path)[:-len(".parquet")]
        frames.append(pair(task, "acc", "acc", "mmlu"))
    paired = pd.concat(frames, ignore_index=True)
    paired.to_csv(OUT, index=False)

    summary = (paired.groupby(["group", "metric"])
               .agg(n=("before", "size"), before=("before", "mean"), after=("after", "mean")))
    summary["lift"] = summary["after"] - summary["before"]
    print(summary.round(4).to_string())
    print(f"\nwrote {OUT} ({len(paired):,} rows)")
