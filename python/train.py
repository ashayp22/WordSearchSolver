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
import os
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th') #sets depth, input_depth, rows, columns for the convolutional neural network


X = []
y = []

folder_path = "dataset/English/Fnt"

#load the data

#method for cropping the image

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

#loads image from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    return images

for i in range(26):
    data = load_images_from_folder(folder_path + "/Sample0" + str(i + 11))
    letter = chr(i + 65)
    for sample in data:
        #now, we make white black and black white

        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) #makes greyscale

        new_sample = cv2.bitwise_not(sample) #inverses image

        new_sample = cv2.resize(new_sample, (28, 28)) #resizes image to 128 * 128


        #adds the data
        X.append(new_sample)
        y.append([i+1])

print("done loading data")

#format the x and y train datasets
X = np.array(X)
y = np.array(y)

print(len(X))
print(len(y))

X = X.reshape(X.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X = X / 255
# one hot encode outputs
y = np_utils.to_categorical(y)
num_classes = y.shape[1]


#create a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


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


# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=100,
verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# serialize model to YAML
model_yaml = model.to_yaml()
with open("models/newmodel3.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("models/newmodel3.h5")

print("Saved model to disk")
