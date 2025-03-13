#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

## Import Packages
##-------------------
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


## Utility Functions
##-------------------
'''Utility functions for EGAIN.
(1) mcar_missing: Randomly delete from arr_cols (ncols=0: from the entire data, ncols>0: from a random selection of cols) to create data with MCAR.
(2) mar_missing: Randomly delete from col_miss based on col_ctrl ranks to create data with MAR.
(3) mnar_missing: Randomly delete from df_col to create data with MNAR.
(4) rmse_loss: Calculate RMSE between full_data and imputed_data for missing values in data_x.
(5) plot_losses: Plot the loss functions over iterations.
(6) rounding: Round imputed data based on input data decimals.
'''

def mcar_missing(arr_cols: np.ndarray, ncols: np.ndarray, miss_rate: float, random_seed=None):
  '''Randomly delete from arr_cols (ncols=0: from the entire data, ncols>0: from a random selection of cols) to create data with MCAR.'''
  if random_seed is not None:
    np.random.seed(random_seed)
  data = arr_cols.copy()
  if ncols == 0:
    # uniform random vector
    u = np.random.uniform(size=data.shape)
    # missing values where u <= miss_rate
    mask = (u <= miss_rate)
    data[mask] = np.nan
  else:
    # Randomly select ncols columns
    selected_cols = np.random.choice(arr_cols.shape[1], ncols, replace=False)
    for col in selected_cols:
      # uniform random vector
      u = np.random.uniform(size=data.shape[0])
      # missing values where u <= miss_rate
      mask = (u <= miss_rate)
      data[mask, col] = np.nan
  return data


def mar_missing(col_miss: np.ndarray, col_ctrl: np.ndarray, miss_rate: float, random_seed=None):
  '''Randomly delete from col_miss based on col_ctrl ranks to create data with MAR.'''
  if random_seed is not None:
    np.random.seed(random_seed)
  data = col_miss.copy()
  # Compute the percentile ranks of the ctrl column
  ranks = np.argsort(col_ctrl, axis=0).argsort(axis=0) + 1
  ranks = ranks / col_ctrl.shape[0]  # Normalize ranks
  # Calculate probabilities based on the ranks to achieve the desired missing rate
  if miss_rate <= 0.5:
    probs = 2 * miss_rate * ranks
  else:
    probs = 1 - 2 * (1 - miss_rate) * (1 - ranks)
  # uniform random vector
  u = np.random.uniform(size=data.shape)
  # missing values where u <= miss_rate
  mask = (u <= probs)
  data[mask] = np.nan
  return data


def mnar_missing(col_miss: np.ndarray, miss_rate: float, missing_on='high', random_seed=None):
  '''Randomly delete from col_miss to create data with MNAR.'''
  if random_seed is not None:
    np.random.seed(random_seed)
  data = df_col.copy()
  # Compute the percentile ranks
  sorted_indices = np.argsort(col_miss, axis=0)
  ranks = np.empty_like(sorted_indices, dtype=float)
  ranks[sorted_indices] = (np.arange(1, col_miss.shape[0] + 1) / col_miss.shape[0])
  # Invert ranks for missing_on='low' direction
  if missing_on == 'low':
    ranks = 1 - ranks
  # Calculate probabilities based on the ranks to achieve the desired missing rate
  if miss_rate <= 0.5:
    probs = 2 * miss_rate * ranks
  else:
    probs = 1 - 2 * (1 - miss_rate) * (1 - ranks)
  # Clip probabilities to ensure they are between 0 and 1
  probs = np.clip(probs, 0, 1)
  # uniform random vector
  u = np.random.uniform(size=data.shape)
  # missing values where u <= miss_rate
  mask = (u <= probs)
  data[mask] = np.nan
  return data


def rmse_loss (full_data: np.ndarray, data_x: np.ndarray, imputed_data: np.ndarray):
  '''Calculate RMSE between full_data and imputed_data for missing values in data_x.'''
  scaler = MinMaxScaler()
  norm_parameters = scaler.fit(full_data)
  full_data = scaler.transform(full_data)
  imputed_data = scaler.transform(imputed_data)
  data_m = 1 - np.isnan(data_x)
  # Only for missing values:  valid_indices = (1-data_m)
  rmse = root_mean_squared_error(full_data*(1-data_m), imputed_data*(1-data_m))
  return rmse


def plot_losses(d_losses, g_losses, g_temp_losses, mse_losses):
  '''Plot the loss functions over iterations.'''
  plt.figure(figsize=(12, 6))
  # Plot Discriminator and Generator Losses
  plt.subplot(1, 2, 1)
  plt.plot(d_losses, label='Discriminator Loss', color='red')
  plt.plot(g_losses, label='Generator Loss', color='blue')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  #plt.title('Discriminator and Generator Losses')
  plt.legend()
  # Plot MSE and ENT Losses
  plt.subplot(1, 2, 2)
  plt.plot(g_temp_losses, label=r'$\mathcal{L}_G$ Loss', color='purple')
  plt.plot(mse_losses, label=r'$\alpha \cdot \mathcal{L}_M$ Loss', color='green')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  #plt.title('GTemp and MSE Losses')
  plt.legend()
  plt.tight_layout()
  plt.show()


def rounding (data_x, imputed_data):
  '''Round imputed data based on data_x decimals.'''
  rounded_data = imputed_data.copy()
  max_places = np.zeros(data_x.shape[1], dtype=int)
  for col in range(data_x.shape[1]):
      for row in data_x[:, col]:
          if not np.isfinite(row):
              continue
          val_str = str(row).split('.')
          if len(val_str) > 1:
              max_places[col] = max(max_places[col], len(val_str[1]))
      rounded_data[:, col] = np.round(rounded_data[:, col], decimals=max_places[col])
  return rounded_data
