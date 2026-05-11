import numpy as np
import pandas as pd
import bhad.utils as utils
from bhad.utils import (
    paste,
    jitter,
    freedman_diaconis,
    log_marglike_nbins,
    exp_normalize,
    geometric_prior,
    onehot_encoder,
    Discretize,
    timer,
)


def test_exp_normalize():
    """Test Exp-normalize trick"""
    x = np.random.normal(size=(1000, 1))
    assert (
        np.round(utils.exp_normalize(x).sum(), 5) == 1
    ), "Input vector does not sum to 1!"


def test_log_post_pmf_nof_bins():
    """Test log posterior prob. measure of number of bins"""
    disc = utils.Discretize(nbins=None, prior_max_M=100, verbose=False)
    x = np.random.normal(size=1000)
    lpost = disc.log_post_pmf_nof_bins(x)
    for p in lpost.values():
        assert np.isfinite(p), f"log post. {p} is nan!"


def test_paste_basic():
    assert paste(["a", "b"], ["c", "d"]) == ["a c", "b d"]


def test_paste_with_separator():
    assert paste(["a", "b"], ["c", "d"], sep="-") == ["a-c", "b-d"]


def test_paste_with_collapse():
    assert paste(["a", "b"], ["c", "d"], collapse=",") == "a c,b d"


def test_paste_with_separator_and_collapse():
    assert paste(["a", "b"], ["c", "d"], sep="-", collapse=",") == "a-c,b-d"


def test_paste_different_lengths():
    assert paste(["a", "b"], ["c"]) == ["a c"]


def test_paste_empty_lists():
    assert paste([], []) == []


def test_paste_single_list():
    assert paste(["a", "b"]) == ["a", "b"]


def test_paste_multiple_lists():
    assert paste(["a", "b"], ["c", "d"], ["e", "f"]) == ["a c e", "b d f"]


# ---------- Additional tests for coverage ----------


def test_jitter_shape() -> None:
    """Test that jitter returns the correct number of values."""
    result = jitter(M=100, seed=42)
    assert len(result) == 100
    assert all(np.isfinite(result))


def test_jitter_small_magnitude() -> None:
    """Jitter values should be very small (divided by noise_scale)."""
    result = jitter(M=50, noise_scale=10**5, seed=1)
    assert np.max(np.abs(result)) < 1e-4


def test_freedman_diaconis_nbins() -> None:
    """Test Freedman-Diaconis returns a positive integer number of bins."""
    np.random.seed(42)
    data = np.random.randn(500)
    nbins = freedman_diaconis(data)
    assert isinstance(nbins, int)
    assert nbins > 0


def test_freedman_diaconis_width() -> None:
    """Test Freedman-Diaconis returns a positive bin width."""
    np.random.seed(42)
    data = np.random.randn(500)
    width = freedman_diaconis(data, return_width=True)
    assert width > 0


def test_log_marglike_nbins_finite() -> None:
    """Log marginal likelihood should return finite values."""
    np.random.seed(42)
    y = np.random.randn(200)
    result = log_marglike_nbins(M=5, y=y)
    assert np.isfinite(result)


def test_geometric_prior_sums_roughly_to_one() -> None:
    """Geometric prior over grid should sum approximately to 1."""
    max_M = 100
    total = sum(geometric_prior(m, gamma=0.9, max_M=max_M) for m in range(1, max_M))
    assert 0.8 < total < 1.1  # approximate due to truncation


def test_geometric_prior_log() -> None:
    """Log geometric prior should be finite."""
    result = geometric_prior(M=5, gamma=0.7, max_M=100, log=True)
    assert np.isfinite(result)


def test_geometric_prior_invalid_gamma() -> None:
    """Gamma outside (0,1) should return 0."""
    result = geometric_prior(M=5, gamma=1.5, max_M=100, log=False)
    assert result == 0.0


def test_onehot_encoder_fit_transform() -> None:
    """Test one-hot encoder on categorical data."""
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "size": ["S", "M", "L", "S"],
        }
    )
    enc = onehot_encoder(prefix_sep="__", verbose=False)
    enc.fit(df)
    result = enc.transform(df)
    # 3 colors + 1 OTHERS + 3 sizes + 1 OTHERS = 8 columns
    assert result.shape == (4, 8)
    # Each row should sum to number of features (2)
    assert all(result.toarray().sum(axis=1) == 2)


def test_onehot_encoder_unseen_levels() -> None:
    """Unseen category levels should map to OTHERS bucket."""
    df_train = pd.DataFrame({"color": ["red", "blue", "green"]})
    df_test = pd.DataFrame({"color": ["red", "purple"]})

    enc = onehot_encoder(prefix_sep="__", verbose=False)
    enc.fit(df_train)
    result = enc.transform(df_test)
    # Each row should still sum to 1 (one feature)
    assert all(result.toarray().sum(axis=1) == 1)


def test_onehot_encoder_get_feature_names() -> None:
    """Feature names should be retrievable after fit."""
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    enc = onehot_encoder(prefix_sep="__", verbose=False)
    enc.fit(df)
    names = enc.get_feature_names_out()
    assert len(names) > 0


def test_discretize_transform_new_data() -> None:
    """Transform should apply fitted bins to new data."""
    np.random.seed(42)
    train = pd.DataFrame({"x": np.random.randn(100)})
    test = pd.DataFrame({"x": np.random.randn(20)})

    disc = Discretize(columns=["x"], nbins=5, verbose=False)
    disc.fit(train)
    result = disc.transform(test)
    assert result.shape == test.shape
    # All values should be binned (interval or object dtype, not raw numeric)
    assert result["x"].dtype != np.float64


def test_discretize_map_estimate() -> None:
    """Discretize with nbins=None should estimate bins via MAP."""
    np.random.seed(42)
    data = pd.DataFrame({"x": np.random.randn(200)})
    disc = Discretize(columns=["x"], nbins=None, verbose=False)
    disc.fit(data)
    assert disc.nbins_ is not None
    assert disc.nbins_ > 0


def test_discretize_zero_variance() -> None:
    """Discretize should handle zero-variance features with jitter."""
    data = pd.DataFrame({"x": np.ones(100)})
    disc = Discretize(columns=["x"], nbins=5, verbose=False)
    disc.fit(data)
    result = disc.transform(data)
    assert result.shape == data.shape


def test_timer_decorator() -> None:
    """Timer decorator should not alter function return value."""

    @timer
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


# pytest -v tests
