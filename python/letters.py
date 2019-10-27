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



K.set_image_dim_ordering('th')

#load the data

X_train, y_train = extract_training_samples('letters')
X_test, y_test = extract_test_samples('letters')


print(X_train.shape)
print(X_test.shape)


print("train size: ", len(X_train))
print("test size: ", len(X_test))

#trim the data to remove black border around

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]



new_train = []
new_test = []

for z in range(len(X_train)):
    new_train.append(cv2.resize(crop_image(X_train[z]), (28, 28)))

print("done cropping train")

for z in range(len(X_test)):
    new_test.append(cv2.resize(crop_image(X_test[z]), (28, 28)))

print("done cropping test")

X_train = np.array(new_train)
X_test = np.array(new_test)

print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))

print(X_train.shape)
print(X_test.shape)


#format the data and normalize

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

# for i in range(0, 4):
# #     print("model ", i)
# #     # build the model
# #     model = larger_model(i)
# #     # Fit the model
# #     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200,
# #     verbose=0)
# #     # Final evaluation of the model
# #     scores = model.evaluate(X_test, y_test, verbose=0)
# #     print("CNN Error: %.2f%%" % (100-scores[1]*100))


# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=200,
verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# serialize model to YAML
model_yaml = model.to_yaml()
with open("models/model3.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("models/model3.h5")

print("Saved model to disk")
