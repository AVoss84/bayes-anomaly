import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin

import bhad.utils as utils
from bhad.utils import Discretize


class BHAD(BaseEstimator, OutlierMixin):

    def __init__(
        self,
        contamination: float = 0.01,
        alpha: float = 1 / 2,
        exclude_col: List[str] = [],
        num_features: List[str] = [],
        cat_features: List[str] = [],
        append_score: bool = False,
        verbose: bool = True,
        # Discretize parameters (optional - set to enable automatic discretization)
        nbins: Optional[int] = None,
        discretize: bool = True,
        lower: Optional[float] = None,
        k: int = 1,
        round_intervals: int = 5,
        eps: float = 0.001,
        make_labels: bool = False,
        prior_gamma: float = 0.9,
        prior_max_M: Optional[int] = None,
    ):
        """
        Bayesian Histogram-based Anomaly Detector (BHAD), see [1] for details.

        Args:
            contamination (float, optional): The amount (fraction) of contamination of the data set, i.e. the proportion
                                             of outliers in the data set. Used when fitting to define the threshold
                                             on the decision function. Defaults to 0.01.
            alpha (float, optional): Hyperparamter ('pseudo counts') of the Dirirchlet prior. Defaults to 1/2.
            exclude_col (Optional[List[str]], optional): List of column names that should be excluded from the model training. Defaults to [].
            num_features (Optional[List[str]], optional): List of numeric feature names (will be discretized). Defaults to [].
            cat_features (Optional[List[str]], optional): List of categorical feature names. Defaults to [].
            append_score (bool, optional): Output input dataset with model scores appended in extra column. Defaults to False.
            verbose (bool, optional): Show user information. Defaults to True.
            nbins (int, optional): Number of bins for discretization. If None, optimal bins are estimated via MAP. Defaults to None.
            discretize (bool, optional): Whether to automatically discretize continuous features. Defaults to True.
            lower (float, optional): Lower bound for binning (e.g., 0 for amounts). Defaults to None.
            k (int, optional): Number of standard deviations for bin intervals. Defaults to 1.
            round_intervals (int, optional): Number of digits to round intervals. Defaults to 5.
            eps (float, optional): Minimum variance threshold for zero-variance check. Defaults to 0.001.
            make_labels (bool, optional): Use integer labels instead of interval labels. Defaults to False.
            prior_gamma (float, optional): Prior hyperparameter for geometric distribution. Defaults to 0.9.
            prior_max_M (int, optional): Maximum number of bins to consider. Defaults to None (auto).

        References:
        -----------
        [1] Vosseler, A. (2022): Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles, in Risks, 10, (132)
        [2] Vosseler, A. (2023): BHAD: Fast unsupervised anomaly detection using Bayesian histograms, Working paper
        """
        super(BHAD, self).__init__()
        self.contamination = contamination  # outlier proportion in the dataset
        self.alpha = alpha  # uniform Dirichlet prior concentration parameter used for each feature
        self.verbose = verbose
        self.append_score = append_score
        self.numeric_features = num_features
        self.cat_features = cat_features
        self.exclude_col = exclude_col  # list with column names in X of columns to exclude for computation of the score

        # Discretize parameters
        self.nbins = nbins
        self.discretize = discretize
        self.lower = lower
        self.k = k
        self.round_intervals = round_intervals
        self.eps = eps
        self.make_labels = make_labels
        self.prior_gamma = prior_gamma
        self.prior_max_M = prior_max_M
        self._discretizer = None  # will hold the fitted Discretize instance

        if self.verbose:
            print("\n-- Bayesian Histogram-based Anomaly Detector (BHAD) --\n")

    def __del__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"BHAD(contamination = {self.contamination}, alpha = {self.alpha}, exclude_col = {self.exclude_col}, numeric_features = {self.numeric_features}, cat_features = {self.cat_features}, append_score = {self.append_score}, verbose = {self.verbose}, nbins = {self.nbins}, discretize = {self.discretize})"

    def _is_same_data(self, X1: pd.DataFrame, X2: pd.DataFrame) -> bool:
        """Fast check if two DataFrames are the same (by shape and index)."""
        if X1.shape != X2.shape:
            return False
        if not X1.index.equals(X2.index):
            return False
        return True

    def _fast_bhad(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Input:
        X:            design matrix as pandas df with all features (must all be categorical,
                        since one-hot enc. will be applied! Otherwise run discretize() first.)
        append_score: Should anomaly score be appended to X?
        return: scores
        """
        assert isinstance(X, pd.DataFrame), "X must be of type pd.DataFrame"
        selected_col = X.columns[~X.columns.isin(self.exclude_col)]
        if len(self.exclude_col) > 0:
            print("Features", self.exclude_col, "excluded.")

        df = X[selected_col].astype(object).copy()
        self.df_shape = df.shape
        self.columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()  # use only categorical (including discretized numerical)
        if len(self.columns) != self.df_shape[1]:
            warnings.warn("Not all features in X are categorical!!")
        self.df = df

        # Apply one-hot encoder to categorical -> sparse dummy matrix
        # -------------------------------------------------------------
        # unique_categories_ = [df[var].unique().tolist() + ['infrequent'] for var in df.columns]
        # self.enc = OneHotEncoder(handle_unknown='infrequent_if_exist', dtype = int, categories = unique_categories_)
        self.enc = utils.onehot_encoder(
            prefix_sep="__", verbose=self.verbose
        )  # more flexible but much slower
        self.df_one = self.enc.fit_transform(df).toarray()
        assert all(
            np.sum(self.df_one, axis=1) == df.shape[1]
        ), "Row sums must be equal to number of features!!"
        if self.verbose:
            print("Matrix dimension after one-hot encoding:", self.df_one.shape)
        self.columns_onehot_ = self.enc.get_feature_names_out()

        # Prior parameters and sufficient statistics:
        # ---------------------------------------------
        self.alphas = np.array(
            [self.alpha] * self.df_one.shape[1]
        )  # Dirichlet concentration parameters; aka pseudo counts
        self.freq = self.df_one.sum(
            axis=0
        )  # suff. statistics of multinomial likelihood
        self.log_pred = np.log(
            (self.alphas + self.freq) / np.sum(self.alphas + self.freq)
        )  # log posterior predictive probabilities for single trial / multinoulli

        # Duplicate list of marg. freq. in an array for elementwise multiplication
        # i.e. Repeat counts for each obs. i =1...n
        # ---------------------------------------------------------------------------
        # Keep frequencies for explanation later
        a = np.tile(
            self.freq, (self.df_shape[0], 1)
        )  # Repeat counts for each obs. i =1...n
        a_bayes = np.tile(self.log_pred, (self.df_shape[0], 1))

        # Keep only nonzero matrix entries
        # (via one-hot encoding matrix), i.e. freq. for respective entries
        # Assign each obs. the overall category count
        # -------------------------------------------------------------------
        self.f_mat = self.df_one * np.array(a)  # keep only nonzero matrix entries
        self.f_mat_bayes = self.df_one * np.array(a_bayes)

        # Calculate outlier score for each row (observation),
        # see equation (5) in [1]
        # -----------------------------------------------------
        out = pd.Series(self.f_mat_bayes.sum(axis=1), index=df.index)
        if self.append_score:
            out = pd.concat([df, pd.DataFrame(out, columns=["outlier_score"])], axis=1)
        return out

    def _discretize_data(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply discretization to the input data if enabled.

        Parameters
        ----------
        X : pandas.DataFrame
            The input samples.
        fit : bool
            If True, fit the discretizer on X. If False, use already fitted discretizer.

        Returns
        -------
        X_discretized : pandas.DataFrame
            The discretized data.
        """
        if not self.discretize:
            return X

        # Check if there are any numeric columns to discretize
        numeric_cols = X.select_dtypes(
            include=[np.number, "float", "int"]
        ).columns.tolist()
        if not numeric_cols:
            if self.verbose:
                print("No numeric features found - skipping discretization.")
            return X

        if fit:
            if self.verbose:
                print("Discretizing continuous features...")
            self._discretizer = Discretize(
                columns=numeric_cols,
                nbins=self.nbins,
                lower=self.lower,
                k=self.k,
                round_intervals=self.round_intervals,
                eps=self.eps,
                make_labels=self.make_labels,
                verbose=self.verbose,
                prior_gamma=self.prior_gamma,
                prior_max_M=self.prior_max_M,
            )
            X_discretized = self._discretizer.fit_transform(X)
        else:
            if self._discretizer is None:
                raise ValueError("Discretizer not fitted. Call fit() first.")
            X_discretized = self._discretizer.transform(X)

        return X_discretized

    def fit(self, X: pd.DataFrame, y: Union[np.array, pd.Series] = None) -> "BHAD":
        """
        Apply the BHAD and calculate the outlier threshold value.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or castable to
            str (e.g. catagorical). If discretize=True (default), continuous
            features will be automatically discretized.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BHAD object
        """
        # Store original input for reference
        self.X_orig_ = X.copy()

        # Apply discretization if enabled
        X_disc = self._discretize_data(X, fit=True)

        # Fit model:
        # ------------
        if self.verbose:
            print("Fit BHAD on discretized data.")
            print(f"Input shape: {X_disc.shape}")
        self.scores = self._fast_bhad(X_disc)

        if self.append_score:
            self.threshold_ = np.nanpercentile(
                self.scores["outlier_score"].tolist(), q=100 * self.contamination
            )
        else:
            self.threshold_ = np.nanpercentile(
                self.scores.tolist(), q=100 * self.contamination
            )
        if self.verbose:
            print("Finished training.")

        # Tag as fitted for sklearn compatibility:
        # https://scikit-learn.org/stable/developers/develop.html#estimated-attributes
        self.X_ = X_disc
        self.xindex_fitted_, self.df_, self.scores_, self.freq_ = (
            self.X_.index,
            self.df,
            self.scores,
            self.freq,
        )
        self.enc_, self.df_one_, self.f_mat_, self.f_mat_bayes_ = (
            self.enc,
            self.df_one,
            self.f_mat,
            self.f_mat_bayes,
        )
        self.numeric_features_, self.cat_features_ = (
            self.numeric_features,
            self.cat_features,
        )
        return self

    def score_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Outlier score calculated by summing the counts
        of each feature level in the dataset.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable
            to str (e.g. categorical). If discretize=True, continuous features
            will be automatically discretized using the fitted discretizer.

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold
            value.
        """
        # Check if this is the same data as training (before discretization)
        is_training_data = hasattr(self, "X_orig_") and self._is_same_data(
            X, self.X_orig_
        )

        if is_training_data:
            # Return cached scores for training data
            self.f_mat = self.f_mat_.copy()
            return self.scores_

        # Apply discretization if enabled
        X_disc = self._discretize_data(X, fit=False)

        if self.verbose:
            print("\nScore input data.")
            print("Apply fitted one-hot encoder.")
        df = X_disc.copy()
        self.df_one = self.enc_.transform(
            df
        ).toarray()  # apply fitted one-hot encoder to categorical -> sparse dummy matrix
        assert all(
            np.sum(self.df_one, axis=1) == df.shape[1]
        ), "Row sums must be equal to number of features!!"

        # Update suff. stat with abs. freq. of new data points/levels
        self.freq_updated_ = self.freq_ + self.df_one.sum(axis=0)
        # freq_updated = np.log(np.exp(self.freq) + self.df_one + alpha)    # multinomial-dirichlet

        # Log posterior predictive probabilities for single trial / multinoulli
        self.log_pred = np.log(
            (self.alphas + self.freq_updated_)
            / np.sum(self.alphas + self.freq_updated_)
        )
        self.f_mat = (
            self.freq_updated_ * self.df_one
        )  # get level specific counts for X, e.g. test set
        f_mat_bayes = self.log_pred * self.df_one
        self.scores = pd.Series(f_mat_bayes.sum(axis=1), index=X.index)

        return self.scores

    def decision_function(self, X: pd.DataFrame) -> np.array:
        """
        Outlier score centered around the threshold value. Outliers are scored
        negatively (<= 0) and inliers are scored positively (> 0).

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable
            to str (e.g. categorical). If discretize=True, continuous features
            will be automatically discretized using the fitted discretizer.

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold
            value.
        """
        # Check if this is the same data as training (before discretization)
        is_training_data = hasattr(self, "X_orig_") and self._is_same_data(
            X, self.X_orig_
        )

        # Center scores; divide into outlier and inlier (-/+)
        if is_training_data:
            if self.verbose:
                print("Score input data.")
            self.anomaly_scores = self.scores_.to_numpy() - self.threshold_
        else:
            self.anomaly_scores = self.score_samples(X).to_numpy() - self.threshold_
        return self.anomaly_scores

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable
            to str (e.g. categorical). If discretize=True, continuous features
            will be automatically discretized using the fitted discretizer.

        Returns
        -------
        scores : array, shape (n_samples,)
            The outlier labels of the input samples.
            -1 means an outlier, 1 means an inlier.
        """
        anomaly_scores = self.decision_function(X)  # get centered anomaly scores
        outliers = np.asarray(
            -1 * np.asarray(anomaly_scores <= 0).astype(int)
        )  # for sklearn compatibility
        inliers = np.asarray(np.asarray(anomaly_scores > 0).astype(int))
        return outliers + inliers
