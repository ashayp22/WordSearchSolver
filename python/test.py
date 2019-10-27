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
from PIL import Image

# load YAML and create model
yaml_file = open('models/newmodel3.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("models/newmodel3.h5")
print("Loaded model from disk")


img = img = cv2.imread('dataset2/e.jpg',0)

img = cv2.bitwise_not(img)

image = cv2.resize(img, (28, 28))

cv2.imshow("image", image)
cv2.waitKey(0)

image = np.copy(image).reshape((1, 1, 28, 28)).astype("float32") / 255

result = loaded_model.predict(np.copy(image));
result = result[0]

# print(result)

#find whats the highest probability
max = np.where(result == np.amax(result))

predicted_letter = chr(max[0] + 64)
print(predicted_letter.lower())
