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

   from bhad.model import BHAD

   num_cols = [....]   # names of numeric features
   cat_cols = [....]   # categorical features

   model = BHAD(contamination=0.01, 
                num_features = num_cols, cat_features = cat_cols
                nbins=None, verbose=False
               )

Setting *nbins* to *None* infers the Bayes-optimal number of bins (=only parameter) using the MAP estimate.

For a given dataset get binary model decisions and anomaly scores:

.. code-block:: python

   y_pred = model.fit_predict(X = dataset)        

   anomaly_scores = model.decision_function(dataset)

Get *global* model explanation as well as for *individual* observations:

.. code-block:: python

   from bhad.explainer import Explainer

   local_expl = explainer.Explainer(bhad_obj=model, discretize_obj=model._discretizer).fit()

   local_expl.get_explanation(nof_feat_expl = 5, append = False)          # individual explanations

   print(local_expl.global_feat_imp)                                      # global explanation

More examples
--------------

In the following notebooks you can find examples of how to use the *bhad* package on toy and real-world datasets:

- `Synthetic dataset - Jupyter Notebook <notebooks/Toy_Example.ipynb>`__
- `Titanic dataset - Jupyter Notebook <notebooks/Titanic_Example.ipynb>`__
