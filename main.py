import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import requests
from bs4 import BeautifulSoup

CATEGORIES = ["Donkey Kong", "Frogger"]

CAM = cv2.VideoCapture(0)
cv2.namedWindow('test')

#Function to prepare image for prediction
def prepare_file(filepath, inspection = 0):

    """
    Function to prepare local file for being feed
    through our neural network

    filepath: path to file(a string)
    inspection: whether or not to show the image after preperation (a boolean)
    """

    IMG_SIZE = 100
    #reads the image
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #downscales the image
    img_array = img_array/255
    #resizing the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).reshape(-1, 100, 100, 1)
    #showing the image for inspection
    if inspection:
        plt.imshow(new_array, cmap = "gray")
        plt.show()

    #returning the prepared image
    return new_array

#Function to prepare image for prediction
def prepare_frame(frame):

    """
    Function to prepare webcam frame to be feed through our neural network

    Frame: image array (numpy array)
    """

    IMG_SIZE = 100
    #converting the frame from our webcam to grayscale
    gray_array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #downscales the image
    gray_array = gray_array/255
    #resizing the image
    new_array = cv2.resize(gray_array, (IMG_SIZE, IMG_SIZE)).reshape(-1, 100, 100, 1)
    #returning the prepared image
    return new_array

# Loads in our trained model
model = tf.keras.models.load_model("games2.model")

# files to test our neural network
x_test = [
        prepare_file('images/testing/frogger.png'),
        prepare_file('images/testing/donkey_kong.png'),
        prepare_file('images/testing/frogger2.jpeg')
        ]

# Loops through our test files
# for i in x_test:
#     prediction = model.predict(i)
#     print(CATEGORIES[np.argmax(prediction)])

retrieve_highscores()

while True:

    # Delay
    time.sleep(0.01)

    # Checks if the escape key is pressed 
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # Getting the frame from the webcam
    ret, frame = CAM.read()
    # Converting the frame to grayscale to match the frame we give our neural network
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The image comming from our webcam i mirroed.
    # therefore we flip the image.
    # 1 is a vertical flip 0 is a horizontal flip
    flipped_frame = cv2.flip(gray_frame, 1)

    # Makes a prediction based on the frame
    prediction = model.predict(prepare_frame(frame))

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


