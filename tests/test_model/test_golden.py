"""Golden-output regression test for BHAD.decision_function.

The reference outputs in ``tests/golden_decision_function.npz`` were
generated from the pre-refactor implementation. After internal changes
(e.g. switching ``df_one`` to a sparse representation) the scores must
remain numerically equivalent up to floating-point round-off.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bhad.model import BHAD

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden_decision_function.npz"


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden file missing")
def test_decision_function_matches_golden() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((500, 8)), columns=[f"f{i}" for i in range(8)]
    )
    X["cat"] = rng.choice(["a", "b", "c", "d"], size=500)

    model = BHAD(contamination=0.05, nbins=7, verbose=False)
    model.fit(X)
    train_scores = model.decision_function(X)

    rng2 = np.random.default_rng(1)
    Xt = pd.DataFrame(
        rng2.standard_normal((200, 8)), columns=[f"f{i}" for i in range(8)]
    )
    Xt["cat"] = rng2.choice(["a", "b", "c", "d", "UNSEEN"], size=200)
    test_scores = model.decision_function(Xt)

    golden = np.load(GOLDEN_PATH)
    np.testing.assert_allclose(train_scores, golden["train"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(test_scores, golden["test"], rtol=1e-10, atol=1e-12)
