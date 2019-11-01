"""
For using the model from:
create_model.py to predict which game the webcam is seing
"""

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ["Burger time", "Amidar", "Donkey kong", "Frogger"]
#CATEGORIES = ["BOOK","ALARM"]

CAM = cv2.VideoCapture(0)
cv2.namedWindow('test')

#Function to prepare image for prediction
def prepare_frame(frame):

    """
    Function to prepare webcam frame to be feed through our neural network

    Frame: image array (numpy array)
    """

    IMG_SIZE = 100

    new_array = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    #returning the prepared image
    return new_array

# Loads in our trained model
model = tf.keras.models.load_model("models/frogger_donkey_from_video.model")

while True:

    # Checks if the escape key is pressed 
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # Getting the frame from the webcam
    ret, frame = CAM.read()

    # The image comming from our webcam i mirroed.
    # therefore we flip the image.
    # 1 is a vertical flip 0 is a horizontal flip
    flipped_frame = cv2.flip(frame, 1)

    new_frame = cv2.resize(frame, (100,100)).reshape(-1, 100,100, 3)

    # Makes a prediction based on the frame
    prediction = model.predict(new_frame.astype(float))

    # Defines a font for the text to be written on our image
    font = cv2.FONT_HERSHEY_DUPLEX

    # Draws text on our image.
    # The text draw shows the prediction of our neural network
    cv2.putText(flipped_frame,  
            CATEGORIES[np.argmax(prediction)],  
            (20, 40),  
            fontFace=font,  
            fontScale=1,  
            color=(0, 0, 0)) 
 
    # Shown our image in the window 'test'
    cv2.imshow('test', flipped_frame,)

    # A little delay.
    # this is neccescary for our window to update
    cv2.waitKey(1)

    

    # Prints the prediction
    # print(CATEGORIES[np.argmax(prediction)])

    # for i in x_test:
    #     cv2.imshow('test', i)
    #     cv2.waitKey(1)
    #     prediction = model.predict(i.reshape(-1, 100, 100, 1))
    #     print(CATEGORIES[np.argmax(prediction)])


