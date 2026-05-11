import pytest
import numpy as np
import pandas as pd
from bhad.model import BHAD


@pytest.fixture
def numeric_data() -> pd.DataFrame:
    """Create a simple numeric dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "f1": np.random.randn(200),
            "f2": np.random.randn(200),
            "f3": np.random.randn(200),
        }
    )


@pytest.fixture
def categorical_data() -> pd.DataFrame:
    """Create a categorical dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "color": np.random.choice(["red", "blue", "green"], size=200),
            "size": np.random.choice(["S", "M", "L", "XL"], size=200),
        }
    )


@pytest.fixture
def mixed_data() -> pd.DataFrame:
    """Create a mixed numeric + categorical dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "amount": np.random.randn(200),
            "category": np.random.choice(["A", "B", "C"], size=200),
        }
    )


class TestBHADInit:
    """Test BHAD initialization."""

    def test_default_parameters(self) -> None:
        model = BHAD(verbose=False)
        assert model.contamination == 0.01
        assert model.alpha == 0.5
        assert model.verbose is False
        assert model.discretize is True

    def test_custom_parameters(self) -> None:
        model = BHAD(
            contamination=0.05,
            alpha=1.0,
            nbins=10,
            verbose=False,
        )
        assert model.contamination == 0.05
        assert model.alpha == 1.0
        assert model.nbins == 10

    def test_repr(self) -> None:
        model = BHAD(verbose=False)
        repr_str = repr(model)
        assert "BHAD" in repr_str
        assert "contamination" in repr_str


class TestBHADFit:
    """Test BHAD fit method."""

    def test_fit_numeric(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        result = model.fit(numeric_data)
        assert result is model  # fit returns self
        assert hasattr(model, "threshold_")
        assert hasattr(model, "scores_")
        assert hasattr(model, "X_")

    def test_fit_categorical(self, categorical_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, discretize=False, verbose=False)
        result = model.fit(categorical_data)
        assert result is model
        assert hasattr(model, "threshold_")

    def test_fit_with_auto_bins(self, numeric_data: pd.DataFrame) -> None:
        """Test fit with MAP bin estimation (nbins=None)."""
        model = BHAD(contamination=0.05, nbins=None, verbose=False)
        model.fit(numeric_data)
        assert hasattr(model, "threshold_")
        assert model._discretizer is not None


class TestBHADPredict:
    """Test BHAD predict and scoring methods."""

    def test_predict_returns_labels(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        model.fit(numeric_data)
        labels = model.predict(numeric_data)
        assert len(labels) == len(numeric_data)
        assert set(np.unique(labels)).issubset({-1, 1})

    def test_fit_predict(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        labels = model.fit_predict(numeric_data)
        assert len(labels) == len(numeric_data)
        assert set(np.unique(labels)).issubset({-1, 1})

    def test_decision_function(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        model.fit(numeric_data)
        scores = model.decision_function(numeric_data)
        assert len(scores) == len(numeric_data)
        # Scores should be centered around threshold: negative = outlier
        assert np.any(scores > 0)  # at least some inliers

    def test_score_samples(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        model.fit(numeric_data)
        scores = model.score_samples(numeric_data)
        assert len(scores) == len(numeric_data)

    def test_predict_on_new_data(self, numeric_data: pd.DataFrame) -> None:
        """Test prediction on unseen data (not training data)."""
        np.random.seed(99)
        new_data = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "f3": np.random.randn(50),
            }
        )
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        model.fit(numeric_data)
        labels = model.predict(new_data)
        assert len(labels) == 50
        assert set(np.unique(labels)).issubset({-1, 1})

    def test_score_samples_new_data(self, numeric_data: pd.DataFrame) -> None:
        """Test score_samples on unseen data."""
        np.random.seed(99)
        new_data = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "f3": np.random.randn(50),
            }
        )
        model = BHAD(contamination=0.05, nbins=5, verbose=False)
        model.fit(numeric_data)
        scores = model.score_samples(new_data)
        assert len(scores) == 50


class TestBHADContamination:
    """Test that contamination parameter affects predictions."""

    def test_higher_contamination_more_outliers(
        self, numeric_data: pd.DataFrame
    ) -> None:
        model_low = BHAD(contamination=0.01, nbins=5, verbose=False)
        model_high = BHAD(contamination=0.10, nbins=5, verbose=False)

        model_low.fit(numeric_data)
        model_high.fit(numeric_data)

        outliers_low = np.sum(model_low.predict(numeric_data) == -1)
        outliers_high = np.sum(model_high.predict(numeric_data) == -1)

        assert outliers_high >= outliers_low


class TestBHADAppendScore:
    """Test append_score functionality."""

    def test_append_score(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, append_score=True, verbose=False)
        model.fit(numeric_data)
        assert "outlier_score" in model.scores_.columns


class TestBHADExcludeCol:
    """Test column exclusion."""

    def test_exclude_columns(self, numeric_data: pd.DataFrame) -> None:
        model = BHAD(contamination=0.05, nbins=5, exclude_col=["f3"], verbose=False)
        model.fit(numeric_data)
        assert "f3" not in model.df_.columns
