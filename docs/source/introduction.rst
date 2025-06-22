Quick start
============

.. contents::
   :local:

Basic model usage
------------------

1. Preprocess the input data: discretize continuous features and conduct Bayesian model selection (*optional*).
2. Train the model using discrete data.

For convenience these two steps can be wrapped up via a scikit-learn pipeline (*optional*).

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from bhad.model import BHAD
   from bhad.utils import Discretize

   num_cols = [....]   # names of numeric features
   cat_cols = [....]   # categorical features

   pipe = Pipeline(steps=[
      ('discrete', Discretize(nbins = None)),   
      ('model', BHAD(contamination = 0.01, num_features = num_cols, cat_features = cat_cols))
   ])

Setting *nbins* to *None* infers the Bayes-optimal number of bins (=only parameter) using the MAP estimate.

For a given dataset get binary model decisions and anomaly scores:

.. code-block:: python

   y_pred = pipe.fit_predict(X = dataset)        

   anomaly_scores = pipe.decision_function(dataset)

Get *global* model explanation as well as for *individual* observations:

.. code-block:: python

   from bhad.explainer import Explainer

   local_expl = Explainer(bhad_obj = pipe.named_steps['model'], discretize_obj = pipe.named_steps['discrete']).fit()

   local_expl.get_explanation(nof_feat_expl = 5, append = False)          # individual explanations

   print(local_expl.global_feat_imp)                                      # global explanation

More examples
--------------

In the following notebooks you can find examples of how to use the *bhad* package on toy and real-world datasets:

.. toctree::
   :maxdepth: 1

   notebooks/Toy_Example
   notebooks/Titanic_Example
