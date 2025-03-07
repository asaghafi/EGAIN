# EGAIN
Abolfazl Saghafi, Soodeh Moallemian, Miray Budak, and Rutvik Deshpande, EGAIN: Enhanced Generative Adversarial Networks for Imputing Missing Values, under review by MPI BDCC, 2025.

Missing values pose a challenge in predictive analysis specially in big data because most models depend on complete datasets to estimate functional relationships between variables. Generative Adversarial Imputation Networks are among the most reliable methods to impute missing values with plausible numbers from the dataset. This research introduces Enhanced Generative Adversarial Networks (EGAIN), which address the GAIN convergence issue, introduce new functionality to the GAIN process, and significantly improve its performance.

# Requirements
The EGAIN package uses `TensorFlow 2`. 

# Install
To install EGAIN using pip, 

```console
pip install EGAIN
```
Imputing a dataset:
After installing the package, are ready to use EGAIN fucction to impute missing values in a dataset. The EGAIN function requires the following inputs: 

```python
## Import requirements
##-------------------
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, MaxPooling1D, Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

## Load data with missing values and store it as numpy ndarray
##-------------------
data_x = pd.read_csv('/content/EGAIN/data/example.csv').to_numpy(dtype=float)

## Set hyperparameters
##-------------------
# batch_size: 64 (default)
# hint_rate: 0.90 (default)
# alpha: 80 (default), can be adjusted after a test run
# iterations: 1000 (default)
egain_parameters = {'batch_size': 64, 'hint_rate': 0.90, 'alpha': 80, 'iterations': 1000}

## Use EGAIN to impute missing values in data_x
##-------------------
# Options: default is False
# plots: True/False plots the generator, discriminator loss functions
# retrain: True/False: whether to use the weights from previous run to retrain
imputed_data = EGAIN(data_x, egain_parameters, retrain=False, plots=True)
```

It is best to set `plots=True` for the first run. A chart like the following will be generated, after the training iterations complete. On the right you have the $\mathcal{L}_G$ Loss that is only applied to imputed missing $(m_i=0)$ and penalizes G if D performs well by correctly outputting low chances. $\alpha \cdot \mathcal{L}_M$ Loss is only applied to observed values $(m_i=1)$ encouraging the generator to produce realistic values that deceive the discriminator.
[<img src="sample.png">](sample.png)

Change value of hyperparameter `alpha` so that the Generator Loss starts around the value of the Discriminator Loss, similar to the following chart where `alpha=100` is chosen. 
[<img src="sample.png">](sample.png)

