#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

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
  # System params
  batch_size = egain_parameters['batch_size']
  hint_rate = egain_parameters['hint_rate']
  alpha = egain_parameters['alpha']
  iterations = egain_parameters['iterations']

  # Other params
  no, dim = data_x.shape
  h_dim = int(dim)

  # Mask matrix
  data_m = 1 - np.isnan(data_x)

  # Normalization
  scaler = MinMaxScaler()
  norm_parameters = scaler.fit(data_x)
  norm_data = scaler.transform(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)

  ## Architecture
  ##--------------------
  # Define Generator Model
  def build_generator():
    x_input = Input(shape=(dim, 2))  # 3D Input
    h1 = Conv1D(filters=dim, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal')(x_input)
    h2 = MaxPooling1D(pool_size=dim, data_format="channels_first", padding='same')(h1)
    h3 = Flatten()(h2)
    h4 = Dense(h_dim, activation='relu', kernel_initializer='glorot_normal')(h3)
    h5 = Dense(h_dim, activation='relu', kernel_initializer='glorot_normal')(h4)
    output = Dense(dim, activation='sigmoid')(h5)
    return Model(x_input, output)

  # Define Discriminator Model
  def build_discriminator():
    x_input = Input(shape=(dim, 2))  # 3D Input
    h1 = Conv1D(filters=dim, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal')(x_input)
    h2 = MaxPooling1D(pool_size=dim, data_format="channels_first", padding='same')(h1)
    h3 = Flatten()(h2)
    h4 = Dense(h_dim, activation='relu', kernel_initializer='glorot_normal')(h3)
    h5 = Dense(h_dim, activation='relu', kernel_initializer='glorot_normal')(h4)
    output = Dense(dim, activation='sigmoid')(h5)
    return Model(x_input, output)

  generator = build_generator()
  discriminator = build_discriminator()

  ## Structure
  ##--------------------
  @tf.function
  def train_step(X_mb, M_mb, H_mb):
    # Convert to float tensors
    X_mb = tf.cast(X_mb, tf.float32)
    M_mb = tf.cast(M_mb, tf.float32)
    H_mb = tf.cast(H_mb, tf.float32)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator
        G_sample = generator(tf.stack([X_mb, M_mb], axis=-1))
        # Combine with observed data
        Hat_X = (M_mb * X_mb) + ((1 - M_mb) * G_sample)
        # Discriminator
        D_prob = discriminator(tf.stack([Hat_X, H_mb], axis=-1))
        ## GAIN loss
        D_loss = 10*-tf.reduce_mean(M_mb * tf.math.log(D_prob + 1e-8) + (1 - M_mb) * tf.math.log(1. - D_prob + 1e-8))
        G_loss_temp = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean(tf.square(M_mb * X_mb - M_mb * G_sample))
        G_loss = G_loss_temp + (alpha * MSE_loss)

    gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)

    optimizer_D.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    optimizer_G.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return D_loss, G_loss_temp, MSE_loss, G_loss

  optimizer_D = tf.keras.optimizers.Adam()
  optimizer_G = tf.keras.optimizers.Adam()

  ## Iterations
  ##--------------------
  # Initialize
  checkpoint_path = "best_generator.weights.h5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, monitor='loss', mode='min')
  best_loss = float('inf')

  if retrain:
    generator.load_weights(checkpoint_path)

  if plots:
    d_losses, g_temp_losses, mse_losses, g_losses = [], [], [], []

  for it in tqdm(range(iterations)):
    # Sample batch
    batch_idx = np.random.permutation(no)[:batch_size]
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]
    # Sample random noise
    Z_mb = np.random.uniform(low=0, high=1., size=[batch_size, dim])
    # Combine random noise with observed vectors
    X_mb = (M_mb * X_mb) + ((1 - M_mb) * Z_mb)
    # Sample hint vectors
    B_mb = np.random.binomial(n=1, p=hint_rate, size=[batch_size, dim])
    H_mb = (B_mb * M_mb) + (0.5 * (1 - B_mb))
    # Calculate loss
    D_loss_curr, G_loss_temp_curr, MSE_loss_curr, G_loss_curr = train_step(X_mb, M_mb, H_mb)

    if plots:
      d_losses.append(D_loss_curr)
      g_temp_losses.append(G_loss_temp_curr)
      mse_losses.append(alpha*MSE_loss_curr)
      g_losses.append(G_loss_curr)

    if G_loss_curr < best_loss:
        best_loss = G_loss_curr
        generator.save_weights(checkpoint_path)

  generator.load_weights(checkpoint_path)

  # Plot the loss curves after training
  if plots:
    plot_losses(d_losses, g_losses, g_temp_losses, mse_losses)

  ## Final Imputation
  ##--------------------
  X_mb = norm_data_x
  M_mb = data_m
  Z_mb = np.random.uniform(low=0, high=1., size=[no, dim])
  X_mb = (M_mb * X_mb) + ((1 - M_mb) * Z_mb)

  ## Return imputed data
  imputed_data = generator(tf.stack([X_mb, M_mb], axis=-1)).numpy()
  imputed_data = (data_m * norm_data_x) + ((1 - data_m) * imputed_data)

  # Renormalization
  imputed_data = scaler.inverse_transform(imputed_data)
  imputed_data = rounding(data_x, imputed_data)

  return imputed_data