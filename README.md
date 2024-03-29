# Word Search Solver

![Preview](https://ashayp.com/images/projects/wordsearchsolver.PNG)

Word Search Solver was my submission to [Congressional App Challenge 2019-2020](https://www.congressionalappchallenge.us/) for Illinois District 8. The project is a website that uses deep learning to solve word search puzzles. After the user has to take a picture of their puzzle, the image is sent to a server that uses image processing algorithms and a Convolutional Neural Network to read the image. A word-search solving algorithm is then applied to the converted puzzle and the results are sent back to the user, with the original image having highlighted words. The user also recieves definitions on every word they are looking for. This app was designed for students who struggle with reading and writing. The project tied for 2nd place at the District Level.

## Word Search Solving Process

1. Obtain colored picture of word search puzzle
2. Convert to black and white
3. Obtain the number of rows and columns
4. Crop out the word search to get images of individual letters
5. Run the CNN on each image to obtain a letter
```
def larger_model():
# create model
    model = Sequential()
    model.add(Conv2D(45, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(25, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```
7. Create a 2D array of letters, which represents the word search puzzle, and search for the given words using a Brute-Force algorithm
8. Highlight the words found

## Getting Started

These instructions will get you a copy of the website running on your local machine for development and testing purposes.

### Prerequisites

Your machine needs to be compatible for running Node.js and Python. These are the dependencies and modules needed.

Node.js
```
"dependencies": {
    "body-parser": "^1.19.0",
    "childprocess": "^2.0.2",
    "ejs": "^2.7.1",
    "express": "^4.17.1",
    "formidable": "^1.2.1",
    "fs": "0.0.1-security",
    "helmet": "^3.21.2",
    "jimp": "^0.8.5",
    "jsdom": "^15.2.0",
    "mv": "^2.1.1",
    "path": "^0.12.7",
    "png-to-jpeg": "^1.0.1",
    "word-definition": "^2.1.6"
  }
```
Python 3.6
```
from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import model_from_yaml
import cv2
import argparse
import tkinter as tk
import json
from PIL import Image
from PyDictionary import PyDictionary
from nltk.corpus import wordnet
```

### Installing

A step by step series of examples that tell you how to get a development env running

Download the zipped version of this repository and unzip the folder.

Next, navigate to the directory through command prompt or terminal and type the following:
```
node app.js
```
A local version of the website should now be running on port 3000.

## Authors

* **Ashay Parikh** - [more details](https://ashayp.com/)

## License

This project is licensed under the Gnu General Public License - see the [LICENSE.md](https://github.com/ashayp22/WordSearchSolver/blob/master/LICENSE) file for details


