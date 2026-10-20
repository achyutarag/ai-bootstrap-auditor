import pandas as pd
from ai_bootstrap_auditor import AuditBatch

# CI release gate: pytest fails the build unless GO
def test_release_gate():
    df = pd.read_csv("eval_results.csv")  # 1 row/item
    batch = AuditBatch(
        correction_method="benjamini_hochberg")
    batch.add_metric(
        "accuracy", df.acc_old, df.acc_new)
    batch.add_guardrail(
        "answer_words", df.words_old, df.words_new,
        guard_direction="not_increase", margin=2.9)
    batch.run(seed=0)
    gate = batch.decision()  # GO / HOLD / NO-GO
    assert gate["decision"] == "GO", gate["reasons"]
