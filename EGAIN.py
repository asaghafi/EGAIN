#
# GNU GENERAL PUBLIC LICENSE
# Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc.
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#

## Import Packages
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

sys.path.append('/content/EGAIN')

from utils import mcar_missing, mar_missing, mnar_missing
from utils import rmse_loss
from utils import plot_losses
from utils import rounding


## EGAIN Function
##-------------------
def EGAIN (data_x, egain_parameters, retrain=False, plots=False):
  '''Imputes missing values in data_x.
  data_x: data with missing values
  egain_parameters: input dictionary of EGAIN parameters
  retrain: True/False, whether to retrain the generator or not
  plots: True/False, whether to plot the losses or not
  '''
  print('THE CODES WILL BE RELEASED AFTER PUBLICATION OF THE PAPER')
  imputed_data = (data_x)

  return imputed_data
