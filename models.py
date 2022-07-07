import tensorflow as tf
import numpy as np

from tensorflow import keras

def lstm(train_x:np.array,train_y:np.array):
    input_shape = np.shape(train_x[1])
    model = keras.models.Sequential([
        keras.Input(shape = input_shape),
        keras.layers.LSTM(50,activation = 'relu', return_sequences = True),
        keras.layers.LSTM(50,activation = 'relu', return_sequences = False),
        keras.layers.Dense(25),
        keras.layers.Dense(1)
    ])
    
    model.summary()

    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam()
    
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

    model.fit(train_x, train_y, batch_size = 1, epochs = 1)

    return model
