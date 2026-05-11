import pytest
import numpy as np
import pandas as pd
from bhad.model import BHAD
from bhad.utils import Discretize
from bhad.explainer import Explainer


@pytest.fixture
def fitted_numeric_model() -> tuple:
    """Create a fitted BHAD model with numeric data and explicit feature lists."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "amount": np.random.randn(200),
            "value": np.random.randn(200),
        }
    )
    # Fit discretizer separately (as Explainer expects)
    disc = Discretize(columns=["amount", "value"], nbins=5, verbose=False)
    data_disc = disc.fit_transform(data)

    model = BHAD(
        contamination=0.05,
        num_features=["amount", "value"],
        cat_features=[],
        discretize=False,
        verbose=False,
    )
    model.fit(data_disc)
    return model, disc, data


@pytest.fixture
def fitted_categorical_model() -> tuple:
    """Create a fitted BHAD model with categorical data and explicit feature lists."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "color": np.random.choice(["red", "blue", "green"], size=200),
            "size": np.random.choice(["S", "M", "L"], size=200),
        }
    )
    disc = Discretize(columns=[], nbins=5, verbose=False)
    data_disc = disc.fit_transform(data)

    model = BHAD(
        contamination=0.05,
        num_features=[],
        cat_features=["color", "size"],
        discretize=False,
        verbose=False,
    )
    model.fit(data_disc)
    return model, disc, data


class TestExplainerInit:
    """Test Explainer initialization."""

    def test_init_with_fitted_objects(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        explainer = Explainer(model, disc, verbose=False)
        assert explainer.avf is model
        assert explainer.disc is disc

    def test_init_requires_fitted_model(self) -> None:
        """Unfitted model should raise an error."""
        model = BHAD(verbose=False)
        disc = Discretize(verbose=False)
        with pytest.raises(Exception):
            Explainer(model, disc, verbose=False)

    def test_repr(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        explainer = Explainer(model, disc, verbose=False)
        repr_str = repr(explainer)
        assert "Explainer" in repr_str


class TestExplainerFit:
    """Test Explainer fit method."""

    def test_fit_numeric(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        explainer = Explainer(model, disc, verbose=False)
        result = explainer.fit()
        assert result is explainer
        assert hasattr(explainer, "feature_distr_")
        assert hasattr(explainer, "modes_")
        assert hasattr(explainer, "cdfs_")

    def test_fit_categorical(self, fitted_categorical_model: tuple) -> None:
        model, disc, _ = fitted_categorical_model
        explainer = Explainer(model, disc, verbose=False)
        result = explainer.fit()
        assert result is explainer
        assert hasattr(explainer, "modes_")


class TestExplainerGetExplanation:
    """Test Explainer get_explanation method."""

    def test_get_explanation_numeric(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        # Need to call predict/score to populate f_mat
        explainer = Explainer(model, disc, verbose=False)
        explainer.fit()
        result = explainer.get_explanation(nof_feat_expl=2, append=True)
        assert "explanation" in result.columns
        assert len(result) == 200

    def test_get_explanation_append_false(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        explainer = Explainer(model, disc, verbose=False)
        explainer.fit()
        result = explainer.get_explanation(nof_feat_expl=2, append=False)
        assert isinstance(result, pd.Series)
        assert result.name == "explanation"

    def test_global_feature_importance(self, fitted_numeric_model: tuple) -> None:
        model, disc, _ = fitted_numeric_model
        explainer = Explainer(model, disc, verbose=False)
        explainer.fit()
        explainer.get_explanation(nof_feat_expl=2)
        assert hasattr(explainer, "global_feat_imp")
        assert len(explainer.global_feat_imp) > 0
        # Importances should be in [0, 1]
        assert explainer.global_feat_imp.values.max() <= 1.0
        assert explainer.global_feat_imp.values.min() >= 0.0
