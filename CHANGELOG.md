# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.3]
### Changed
- complement type hints
- add python package structure
- add verbose
- make scoring more efficient


## [0.0.4]
### Changed
- add local model explainer module
- reset_index dataframe in discretize() fct in case input data was shuffeled, e.g. from train_test_split. Maight case troubles in explainer 

## [0.0.5]
### Bug fix in explainer
- correct get_explanation() method of Explainer class (line 153)
- df_orig had different column order than nz_freq. This lead to the expaliner assigning the wrong values to the variable names in expalnation column (output of Explainer)
- correct mask in get_feature_names_out() method of onehot_encoder class in utils.py. String search did not work correctly and lead to wrong indexing in pmf's of explainer.py -> make string 
- change logical negation from '~' to 'not' in _make_explanation_string() of explainer.py. 'if ~any(comp):' leads to unexpected outcome.
- Capitalize discretize class
- add global model explanations


## [0.0.7]
### Change threshold logic in explainer
- once most relevant features are determined for local explanations, compute univariate ECDFs for each continuous feature (based on org. scales). 
Then compute the empirical (1-p)% confidence interval of the observations. 
If an observation is not an element of that interval consider it as relevant (w.r.t. anomaly score expl.)     
- Readme.md change 

## [0.0.9]
### explainer module: Change maximum number of bins logic
- If user does not specify a maximum number of bins in the explainer, use a square root of sampe size as a default rule

## [0.1.0]
### Change one_hot_encoder for speed improvements
- Use vectorization in transform method of one_hot_encoder of utils.py. Yields favorable run time improvement

## [0.2.0]
### Update to Python 3.12
- Change installation process from setup.py to suing pyproject.toml
- Update Python version and related package dependencies

## [0.2.7]
### Performance optimizations
- Replace slow `np.apply_along_axis()` with vectorized `sum(axis=1)` in BHAD model scoring
- Replace expensive `DataFrame.equals()` checks with fast shape/index comparison in `_is_same_data()` helper
- Vectorize `Discretize.transform()` method: replaced slow row-by-row `itertuples()` loop with vectorized `pd.cut()` for dramatic speedup on large datasets
- Optimize `onehot_encoder.transform()`: use sparse matrix construction directly instead of dense array allocation
- Store fitted one-hot matrix as `csr_matrix` for consistency between fit and transform
- Add `_is_same_data()` helper methods to `BHAD`, `Discretize`, and `onehot_encoder` classes for efficient data caching checks
- Update README to use uv for setup
- Add more documentation

## [Unreleased]
### Performance

- `BHAD._fast_bhad` and `BHAD.score_samples` now operate on a sparse CSR
  one-hot matrix end-to-end. The previous code densified the matrix and
  built two `O(n_samples * n_categories)` dense tiles via `np.tile`; the
  score is now computed as a single sparse-dense matvec
  (`df_one @ log_pred`), and `BHAD.f_mat` is stored as a sparse matrix.
  Memory usage on `fit` and `score_samples` drops substantially.
- `BHAD.f_mat_bayes` / `f_mat_bayes_` are now lazy `@property` accessors
  computed on demand from the stored sparse matrices and log probabilities;
  no eager allocation. Public API is unchanged.
- `onehot_encoder.fit` no longer accumulates dummies with an in-loop
  `pd.concat` (a quadratic anti-pattern). A single `pd.concat` after the
  loop turns it linear in the number of input columns.
- `onehot_encoder.transform` is now fully vectorised: it uses
  `pd.Categorical.codes` to map values to global column indices and builds
  the CSR matrix in one shot, removing the per-level `np.where` scans and
  the Python `list.extend` over `range(n_rows)`.
- `Discretize.log_post_pmf_nof_bins` now caches the data-independent prior
  integration on the instance; previously the Simpson integration over the
  gamma grid was recomputed for every numeric feature.
- The `simpson(...)` call in `Discretize` now uses keyword arguments,
  silencing the SciPy DeprecationWarning that was logged on every fit.
- `Explainer.get_explanation` builds the explanation column in a Python
  list and assigns it once at the end, instead of writing every row via
  `df.loc[obs, "explanation"] = ...` inside the loop.

### Tests / benchmarks

- Added `tests/test_model/test_golden.py` and a small
  `tests/golden_decision_function.npz` fixture so any future internal
  refactor of the scoring path is checked against pre-refactor outputs
  to within `rtol=1e-10`.
- Added `scripts/bench.py` and `docs/perf.md` with before/after wall-clock
  numbers for the main hot paths.

## [0.2.9]
### Integrated Discretization into BHAD
- **New simplified API**: BHAD now includes built-in discretization, eliminating the need for sklearn Pipeline wrapper
- Added new parameters to BHAD constructor: `nbins`, `discretize`, `lower`, `k`, `round_intervals`, `eps`, `make_labels`, `prior_gamma`, `prior_max_M`
- When `discretize=True` (default), continuous features are automatically discretized during `fit()` and `transform()` calls
- Internal `_discretizer` attribute holds the fitted `Discretize` instance for use with the Explainer
- Updated `score_samples()`, `decision_function()`, and `predict()` to handle automatic discretization of new data
- **Backward compatible**: Set `discretize=False` to use the old Pipeline-based workflow
- Added new example notebook `Titanic_Example_NewAPI.ipynb` demonstrating the simplified API

#### Migration Guide
**Old API (still supported with `discretize=False`):**
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[
    ('discrete', Discretize(nbins=None)),
    ('model', BHAD(contamination=0.01, discretize=False))
])
y_pred = pipe.fit_predict(X_train)
```

**New API (recommended):**
```python
model = BHAD(contamination=0.01, nbins=None)
y_pred = model.fit_predict(X_train)
```