# Part 2 in sentdex tutorial

import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle

DATADIR = "images"

CATEGORIES = ["donkey_kong", "frogger", "amidar"]

IMG_SIZE = 100

training_data = []


for catagory in CATEGORIES:
    path = os.path.join(DATADIR, catagory)

    class_num = CATEGORIES.index(catagory)

    for img in tqdm(os.listdir(path)):

        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            # plt.imshow(img_array)
            # plt.show()

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            plt.imshow(new_array, cmap="gray")
            plt.show()
            #print(img)

            training_data.append([new_array, class_num])
        except Exception as e:
            print(e)
            
        break
    break

# creating training data
def create_training_data():

    # looping through cats and dogs
    for category in CATEGORIES:  

        # makes path based on cat or dog
        path = os.path.join(DATADIR,category)  

        # class number is based on the index of the category
        # a dog is a 0 and a cat is a 1
        class_num = CATEGORIES.index(category)  

        # looping through all images in the corresponding dir (directory)
        # Dog for dog category
        # Cat for cat category
        for img in tqdm(os.listdir(path)): 

            # some files are currupt this is the reason for using a try except statement
            # for some stuppid reason costing me at least 30 minutes of trail and error
            # was that the files were curropt and even using a try except statement didn't work
            # anyhow i deleted all the currupt files so now it works
            try:
                # printing file name
                print(img)
                # converting image to array
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
                    
                # resizing the image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # adds image to training data
                # new_array is the feature
                # class_num is the label
                # read more in notes.txt
                training_data.append([new_array, class_num])

            except Exception as e:
                # deletes file if exception
                print("general exception", e, os.path.join(path,img))
                os.remove(os.path.join(path,img))


create_training_data()

random.shuffle(training_data)

# this will be our features
X = []
#this will be our labels
y = []

# Splitting our training data in to the two variables we need
for feature, label in training_data:
    X.append(feature)
    y.append(label)

# Converting our features in to an np array
# This has something to do with the with tensorflow and keras work
# Reshape is for the same reason
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close    

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close

print(len(training_data))