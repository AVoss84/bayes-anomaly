"""Benchmark script for BHAD latency.

Times the main hot paths so before/after comparisons are reproducible.
Run with the repo's venv active:

    python scripts/bench.py

Output is a markdown table printed to stdout (and appended to docs/perf.md
when invoked with ``--write``).
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bhad.explainer import Explainer  # noqa: E402
from bhad.model import BHAD  # noqa: E402


def _time(fn: Callable[[], None], repeats: int = 5) -> tuple[float, float]:
    """Return (median, min) seconds across ``repeats`` runs."""
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def _make_numeric(n: int, p: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"f{i}" for i in range(p)],
    )


def _make_mixed(n: int, p_num: int, p_cat: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        f"num{i}": rng.standard_normal(n) for i in range(p_num)
    }
    for j in range(p_cat):
        data[f"cat{j}"] = rng.choice(list("ABCDE"), size=n)
    return pd.DataFrame(data)


def bench_fit_predict(n: int, p: int, nbins: int = 10) -> dict:
    X = _make_numeric(n, p)

    def run() -> None:
        model = BHAD(contamination=0.05, nbins=nbins, verbose=False)
        model.fit_predict(X)

    median, best = _time(run, repeats=3 if n >= 100_000 else 5)
    return {"name": f"fit_predict n={n} p={p} nbins={nbins}", "median": median, "min": best}


def bench_predict_new(n_train: int, n_test: int, p: int, nbins: int = 10) -> dict:
    """Time predict on unseen data (exercises the sparse score_samples path)."""
    X_train = _make_numeric(n_train, p, seed=0)
    X_test = _make_numeric(n_test, p, seed=1)
    model = BHAD(contamination=0.05, nbins=nbins, verbose=False)
    model.fit(X_train)

    def run() -> None:
        model.predict(X_test)

    median, best = _time(run, repeats=5)
    return {
        "name": f"predict (new data) n_train={n_train} n_test={n_test} p={p}",
        "median": median,
        "min": best,
    }


def bench_map_bins(n: int = 10_000, p: int = 10) -> dict:
    X = _make_numeric(n, p)

    def run() -> None:
        model = BHAD(contamination=0.05, nbins=None, verbose=False)
        model.fit(X)

    median, best = _time(run, repeats=3)
    return {"name": f"fit (MAP nbins) n={n} p={p}", "median": median, "min": best}


def bench_explainer(n: int = 10_000, p_num: int = 5, p_cat: int = 5) -> dict:
    X = _make_mixed(n, p_num, p_cat)
    num_cols = [c for c in X.columns if c.startswith("num")]
    cat_cols = [c for c in X.columns if c.startswith("cat")]

    def run() -> None:
        model = BHAD(
            contamination=0.05,
            nbins=10,
            num_features=num_cols,
            cat_features=cat_cols,
            verbose=False,
        )
        model.fit(X)
        # Need a fitted discretizer for the explainer; it lives on model._discretizer.
        expl = Explainer(model, model._discretizer, verbose=False)
        expl.fit()
        expl.get_explanation()

    median, best = _time(run, repeats=2)
    return {"name": f"explainer n={n} p_num={p_num} p_cat={p_cat}", "median": median, "min": best}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip the largest cases")
    args = parser.parse_args()

    results: list[dict] = []
    results.append(bench_fit_predict(n=1_000, p=10))
    results.append(bench_fit_predict(n=10_000, p=20))
    if not args.quick:
        results.append(bench_fit_predict(n=100_000, p=50))
    results.append(bench_map_bins(n=10_000, p=10))
    results.append(bench_predict_new(n_train=10_000, n_test=5_000, p=20))
    results.append(bench_explainer(n=10_000, p_num=5, p_cat=5))

    print()
    print("| benchmark | median (s) | min (s) |")
    print("|---|---:|---:|")
    for r in results:
        print(f"| {r['name']} | {r['median']:.3f} | {r['min']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
