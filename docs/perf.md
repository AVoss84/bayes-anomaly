# BHAD performance baselines

Numbers are wall-clock seconds on the dev machine; reproduce with:

```bash
python scripts/bench.py
```

Each benchmark runs multiple repeats; we report the median and best (min)
runtimes. All tests in `tests/` are green for every recorded column,
including a golden-output regression test that pins `decision_function`
values pre- and post-refactor (`tests/test_model/test_golden.py`).

## Results

| benchmark                                          | before (median, s) | after (median, s) | speedup |
|----------------------------------------------------|-------------------:|------------------:|--------:|
| `fit_predict` n=1_000   p=10  nbins=10              |              0.025 |             0.025 |  ~1.0x  |
| `fit_predict` n=10_000  p=20  nbins=10              |              0.408 |             0.404 |  ~1.0x  |
| `fit_predict` n=100_000 p=50  nbins=10              |              9.275 |             9.200 |  ~1.0x  |
| `fit` (MAP nbins) n=10_000 p=10                     |              0.954 |             0.926 |  ~1.03x |
| `predict` (new data) n_train=10_000 n_test=5_000 p=20 |          n/a (1)   |             0.121 |   n/a   |
| `explainer` n=10_000 p_num=5 p_cat=5                |              1.036 |             0.622 |  ~1.7x  |

(1) `predict (new data)` was added after the refactor; baseline timing was
not collected pre-refactor.

## Where the wins (and the lack thereof) come from

- The big visible win is in the **Explainer**: replacing the per-row
  `df.loc[obs, "explanation"] = ...` write pattern with a single column
  assignment removes the dominant cost of the row loop. ~1.7x faster on
  10k rows; the gap grows with `n`.
- The `BHAD` fit/score path is now operating on sparse CSR throughout:
  `df_one` and `f_mat` are sparse, the score is a single
  `df_one @ log_pred` matvec, and no `O(n * n_cats)` `np.tile` allocations
  are made. Wall-clock improvement is small in the listed benchmarks
  because the remaining bottleneck on numeric data is `pd.cut` /
  `Interval -> object` conversions inside `Discretize.fit` (about half
  the runtime of the 100k case). Memory consumption, however, drops
  substantially -- the previous code allocated two dense `(n, n_cats)`
  float64 matrices on every fit and every score call.
- `onehot_encoder.fit` no longer has the quadratic in-loop `pd.concat`;
  the gain is invisible at p=50 but becomes large for wide frames
  (hundreds of columns).
- `onehot_encoder.transform` no longer uses Python `list.extend` to build
  COO indices; it constructs the CSR in a single numpy operation.
- `Discretize.fit` no longer recomputes the data-independent prior
  integration once per feature -- it is cached on the instance after the
  first call.

## Future work (out of scope here)

- Remove the redundant `astype(object)` / Categorical conversions between
  `Discretize.fit`, `_fast_bhad`, and `onehot_encoder.fit`. Profiling shows
  these are now the single biggest cost in `fit` on large numeric data.
- Vectorise `log_marglike_nbins` across all `M` for a feature (re-use one
  big `np.histogram` and aggregate via cumulative sums).
- Tighten dtypes to float32 in the score arithmetic where safe.
