# Arcade game highscore recognition
A neural network based program to detect whether or not a highscores has been reached


I will try to be building a python program to detect the current game played on an arcade machine,web scrape highcores from opfinderklubben.dk and check if a highscore has been reached


**MAIN FILE:** main.py

## LOG

### Friday November 1st 2019
- **Predictions are now more accurate** thanks to noise when creating the training data. this reduces the perfectness of the images. making it easier to predict real life images
- **New files**
    - load_frames_from_video.py
        - Extracts frame from a gameplay video of a given game and apllies noise, rotation and cropping. to make it less perfect
    - create_training_data.py
        - Takes the outputted images from load_frames_from_video and turns them into training sets (pickle files)
    - create_model.py
        - Takes the training data from create_training_data and trains a fully connected neural network
    - test_model_with_webcam.py
        - Opens the webcam and makes prediction based on the frames
- **New folders**
    - models
    - training_sets
    - files_not_in_use
        - for files not in use

### Saturday October 19th 2019
- added model for:
    - Frogger
    - Donkey Kong
    - Amidar
- creted tflite model for rasperry pi

### Friday October 18th 2019
- i have now trained a neural network using python and is able to classify frogger and donkey kon with an acurracy of 100% during training

