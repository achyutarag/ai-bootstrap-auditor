import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session")
def legacy():
    """The exact ai_bootstrap_auditor.py submitted to TAE @ NeurIPS 2026, kept
    so regression tests can show they exercise the historical defect, not
    only the fix."""
    path = os.path.join(REPO_ROOT, "legacy", "ai_bootstrap_auditor_tae2026.py")
    spec = importlib.util.spec_from_file_location("ai_bootstrap_auditor_tae2026", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def regression_pair():
    """n=180 paired binary data with a real regression (true 0.72 -> 0.60).
    Uncorrected audit (seed=1, B=4000): lift -0.106, CI [-0.206, -0.011],
    NO-GO."""
    r = np.random.RandomState(0)
    return r.binomial(1, 0.72, 180), r.binomial(1, 0.60, 180)


def null_metrics(k, n=180, base_seed=10000):
    """k true-null metrics (identical before/after rate) to pad a batch."""
    out = []
    for j in range(k):
        r = np.random.RandomState(base_seed + j)
        out.append((r.binomial(1, 0.7, n), r.binomial(1, 0.7, n)))
    return out
