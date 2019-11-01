"""
For taking the training data from create_training_data.py 
and train a model with it
"""

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
from numpy import array
import pickle


TENSORBOARD_NAME = "DONKEY_FROGGER_FROM_VIDEO"

tensorboard = TensorBoard(log_dir = "logs/{}".format(TENSORBOARD_NAME))

pickle_in = open("training_sets/train_X_from_video1.pickle","rb")
X = pickle.load(pickle_in)
print(len(X))

pickle_in = open("training_sets/train_y_from_video1.pickle","rb")
y = array(pickle.load(pickle_in))
print(len(y))

# Down scaling our training features.
# This is possible because all pixels
# in our images range between 0 and 255
X = X/255.0


model = Sequential()

model.add(Flatten())

model.add(Dense(128, activation = tf.nn.relu))

model.add(Dense(128, activation = tf.nn.relu))

#Remember the last layer needs to math the number of labels
model.add(Dense(4, activation = tf.nn.softmax))

model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
)

model.fit(X,y, epochs = 34)

model.save('{}.model'.format("models/frogger_donkey_from_video"))