# Bayesian Histogram Anomaly Detection (BHAD)

[![PyPI version](https://badge.fury.io/py/bhad.svg)](https://badge.fury.io/py/bhad)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the **Bayesian Histogram-based Anomaly Detection (BHAD)** algorithm for unsupervised anomaly detection with explainability features.

## Overview

BHAD is an explainable anomaly detection method that leverages Bayesian inference and histogram-based modeling to identify outliers in high-dimensional datasets. The algorithm provides both global and local explainability due to its linear structure, making it particularly valuable for applications requiring interpretable results.

### Key Features

- **Explainable AI**: Provides both global and local explanations for anomaly predictions
- **Bayesian Approach**: Uses Bayesian inference for robust uncertainty quantification
- **High-Dimensional Data**: Handles high-dimensional datasets effectively
- **Unsupervised Learning**: No labeled data required for training
- **Linear Structure**: Interpretable model architecture

## Installation

### Using uv

Install package via [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12
uv add bhad
```

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install bhad
```

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from bhad.model import BHAD
from bhad.utils import Discretize

# Load your data
X = pd.DataFrame(np.random.randn(1000, 10), 
                 columns=[f'feature_{i}' for i in range(10)])

# Create pipeline with discretization and BHAD model
pipe = Pipeline(steps=[
    ('discrete', Discretize(nbins=None, verbose=False)),  # Discretize continuous features
    ('model', BHAD(contamination=0.01))                   # BHAD model
])

# Fit the pipeline and predict anomalies
anomaly_labels = pipe.fit_predict(X)        # Returns -1 for outliers, 1 for inliers
anomaly_scores = pipe.decision_function(X)
```

## Documentation

For detailed usage examples, API reference, and tutorials, visit our [documentation](https://avoss84.github.io/bayes-anomaly/).

## Examples

The package includes Jupyter notebooks with practical examples:
- `Toy_Example.ipynb`: Simulated data demonstration
- `Titanic_Example.ipynb`: Real-world dataset application

## Research & Publications

This implementation is based on the following research papers:

1. **Vosseler, A. (2022)**: [Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles](https://www.researchgate.net/publication/361463552_Unsupervised_Insurance_Fraud_Prediction_Based_on_Anomaly_Detector_Ensembles)

2. **Vosseler, A. (2023)**: [BHAD: Explainable anomaly detection using Bayesian histograms](https://www.researchgate.net/publication/364265660_BHAD_Explainable_anomaly_detection_using_Bayesian_histograms)

## Conference Presentations

- **PyCon DE & PyData Berlin 2023**: [Watch the presentation](https://www.youtube.com/watch?v=_8zfgPTD-d8&list=PLGVZCDnMOq0peDguAzds7kVmBr8avp46K&index=8)
- **MaxEnt 2023**: [42nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering](https://www.mdpi.com/2673-9984/9/1/1), Max-Planck-Institute for Plasma Physics, Garching, Germany

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Alexander Vosseler**

## Citation

If you use BHAD in your research, please cite:

```bibtex
@article{vosseler2022unsupervised,
  title={Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles},
  author={Vosseler, Alexander},
  journal={Risks},
  volume={10},
  number={7},
  year={2022},
  month={June}
}
```
