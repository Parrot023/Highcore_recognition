import cv2
import numpy as np
import random

"""
For importing a gameplay video of a given game,
put some noise rotation and cropping on the images to make them as real life as possible
and outputting them to then later be turned into to training data by.
create_training_data.py
"""

#Imports video
cap = cv2.VideoCapture('videos/burger_time.mp4')

count = 0

#Loops through every frame
while(cap.isOpened()):
    
    count += 1

    #reads frame from video
    ret, frame = cap.read()

    if count%60 == 0:

        #Checks for q key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #gets width, height and channels from frame 
        rows, cols, channels = frame.shape

        #Picks a random rotation value
        rotation_angle = random.randint(-15,15)

        #Gets rotation matrix
        #I havent fully understood this one yet
        #You can read about it here under rotation
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html?highlight=rotate
        Matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
        
        #Rotates image
        new_frame = cv2.warpAffine(frame, Matrix, (cols,rows))

        #Create noise img
        noise_img = np.zeros((rows,cols, channels), np.uint8)
        #Genrating noise
        noise = cv2.randu(noise_img, (0), (150))

        #Adding noise to rotated image
        new_frame = new_frame + noise_img

        #Create blank image
        #This is for adding sidebars
        blank_image = np.zeros((rows + 50 ,cols + 50, channels), np.uint8)

        #Picks a random color for the side bars
        side_bar_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

        #Colors the blank image the random color
        blank_image[:] = side_bar_color

        size_divider = random.randint(1, 2)

        new_frame = cv2.resize(new_frame, (int(cols/size_divider), int(rows/size_divider)))

        #Putting the image with noise and rotation and the blank image with sidebars
        
        while True:
            x_offset = 25 + random.randint(0, cols)
            y_offset = 25 + random.randint(0, cols)
            try:
                blank_image[y_offset:y_offset+int(rows/size_divider), x_offset:x_offset+int(cols/size_divider)] = new_frame
                break
            except Exception as e:
                print("Position not possible")

        #Showing the image
        cv2.imshow('frame', blank_image)
        cv2.imwrite('data/burger_time/burger_time{}.png'.format(count), blank_image)
        print("saved image")


cap.release()
cv2.destroyAllWindows()