# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras import regularizers


# set random state for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# define early stopping callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=50, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# define the neural network model
input_shape = [X_train.shape[1]]

nn_model = keras.Sequential([
    layers.Dense(units=512, activation='relu', kernel_regularizer=regularizers.l1(0.0003), input_shape=input_shape),
    #layers.Dropout(rate=0.3, seed=0),
    layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    #layers.Dropout(rate=0.3, seed=0),
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=1)
])

# define optimizer
nn_model.compile(
    optimizer="adam",
    loss="mean_squared_logarithmic_error"
)

# train neural network
history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=10000,
    callbacks=[early_stopping],
    verbose=0 # turn off training log
)
