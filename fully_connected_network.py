
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from numpy import array
import cv2

import pickle

TENSORBOARD_NAME = "Donkey or Frogger"

tensorboard = TensorBoard(log_dir = "logs/{}".format(TENSORBOARD_NAME))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

# I had a problem with tensorflow saying:
# Failed to find data adapter that can handle input: <class 'numpy.ndarray'>, (<class 'list'> containing values of types {"<class 'int'>"})  
# i found in a comment on a youtube video that it was because.
# X was a numpy array and y wasn't
# therfore i convert y to a numpy array
pickle_in = open("y.pickle","rb")
y = array(pickle.load(pickle_in))

# Down scaling our training features.
# This is possible because all pixels
# in our images range between 0 and 255
X = X/255.0


model2 = Sequential()

model2.add(Flatten())

model2.add(Dense(128, activation = tf.nn.relu))

model2.add(Dense(128, activation = tf.nn.relu))

model2.add(Dense(2, activation = tf.nn.softmax))

model2.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
)

model2.fit(X,y, epochs = 100)

model2.save('games2.model')

