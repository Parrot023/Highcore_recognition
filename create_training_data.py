
import pickle
import cv2
import numpy as np
import os
import random

"""
For taking the frames from outputed by load_frames_from_video.py
and turning them into a training set
"""

TRAINING_DATA = []

PATH_TO_DATA_FOLDER = 'data'
IMG_SIZE = 100

dirs = os.listdir(PATH_TO_DATA_FOLDER)
CATEGORIES = []

for i in dirs:
    CATEGORIES.append(i)

    class_num = dirs.index(i)
    print(dirs.index(i))

    for j in os.listdir(PATH_TO_DATA_FOLDER + "/" + i):

        try:
            print(PATH_TO_DATA_FOLDER + "/" + i + "/"+ j)
            array = cv2.imread(PATH_TO_DATA_FOLDER + "/" + i + "/"+ j)
            new_array = cv2.resize(array, (IMG_SIZE, IMG_SIZE))

            TRAINING_DATA.append([new_array, class_num])

        except Exception as e:
            print(e)


random.shuffle(TRAINING_DATA)


train_X = []

train_y = []

for feature, label in TRAINING_DATA:
    train_X.append(feature)
    train_y.append(label)

print(len(train_X))
print(len(train_y))

train_X = np.array(train_X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print("reshape " + str(len(train_X)))

pickle_out = open("training_sets/train_X_from_video1.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close    

pickle_out = open("training_sets/train_y_from_video1.pickle", "wb")
pickle.dump(train_y, pickle_out)
pickle_out.close

print("CATEGORIES: {CATEGORIES}".format(CATEGORIES=CATEGORIES))