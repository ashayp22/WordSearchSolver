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

#word search solver


X_test, y_test = extract_test_samples('letters')

# cv2.imshow("image", X_test[0])
# cv2.waitKey(0)


X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_test = X_test / 255

y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

# print(y_test)

# load YAML and create model
yaml_file = open('models/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")


# imgage = X_test[45].reshape((1, 1, 28, 28)).astype("float32") / 255
# # print(img)
# result = loaded_model.predict(np.copy(imgage));
# print(result)
# result = result[0]
#
# max = np.where(result == np.amax(result))
# print(chr(max[0] + 64));
# print(y_test[45])


hletter = cv2.imread('images/kletter.jpg', cv2.IMREAD_GRAYSCALE)

#make white black and black white

hletter = cv2.resize(hletter, (28, 28))
hletter = cv2.bitwise_not(hletter)


imgage = hletter.reshape((1, 1, 28, 28)).astype("float32") / 255
# print(img)
result = loaded_model.predict(np.copy(imgage));
print(result)
result = result[0]

max = np.where(result == np.amax(result))
print(chr(max[0] + 64));
# print("H")


# return max == label

# def predictImage(img, label):
#     global loaded_model
#     imgage = img.reshape((1, 1, 28, 28)).astype("float32") / 255
#     # print(img)
#     result = loaded_model.predict(np.copy(imgage));
#     print(result)
#     result = result[0]
#
#     max = np.where(result == np.amax(result))
#     # print(chr(max[0] + 64));
#     return max == label
#
#
# counter = 0
#
# for i in range(0, 6):
#     imagee = X_test[i]
#     label = np.where(y_test[i] == np.amax(y_test[i]))
#     if predictImage(np.copy(imagee), label):
#         counter += 1
#
# print("ya yeeeeeeeeeeeeeeeet")
# print(counter / 20800)

# # evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test, verbose=0)
# print("CNN Error: %.2f%%" % (100-score[1]*100))

#try on a letter
#
#
# img = cv2.imread('images/hletter.jpg', cv2.IMREAD_GRAYSCALE)
#
# #make white black and black white
#
# img = cv2.resize(img, (28, 28))
# img = cv2.bitwise_not(img)
#
# cv2.imshow("image", img)
# cv2.waitKey(0)
#
#
# img = np.array(img)
#
# #check h letter, then compare data values - might have to do with reshape issue
#
#
# # print(img)
#
# img = img.reshape((1, 1, 28, 28)).astype("float32") / 255
#
# print(img)
#
#
# result = loaded_model.predict(img);
# result = result[0]
#
# max = np.where(result == np.amax(result))
#
# print(chr(max[0] + 65));
#
# def formatImage(image): #formats the raw image (big dimensions) into a smaller size to match the dimensions of the training set
#     # first, format drawing since it is too big
#     features = []  # ending array
#
#     # add empty values to features
#     for i in range(28):
#         t = []
#         for j in range(28):
#             t.append(0)
#         features.append(t)
#
#     # now, scale down image
#     multiplier = int(WINDOW_SIZE / 28)
#     for i in range(0, len(image)):
#         for j in range(0, len(image[i])):
#             features[int(j / multiplier)][int(i / multiplier)] += image[i][j]
#
#     # print("picture")
#     # for k in features:
#     #     t = ""
#     #     for u in k:
#     #         if u > 0:
#     #             t += "$"
#     #         else:
#     #             t += "."
#     #     print(t)
#
#     features = np.array(features)  # convert the features into
#     features = features.flatten()  # make 1 dimension
#     print(features)
#     features = np.true_divide(features, multiplier**2) # average out
#     features = np.true_divide(features, 255) # normalize out
#     return features
#
#
# WINDOW_SIZE = 140;
#
# #vars
# mouse_pressed = False #is mouse pressed
# pixels = []
#
# def printPixels(): #prints the pixels
#     for i in range(0, WINDOW_SIZE):
#         text = ""
#         for j in range(0, WINDOW_SIZE):
#             if pixels[j][i] == 0:
#                 text += "."
#             elif pixels[j][i] == 255:
#                 text += "$"
#         print(text)
#
#
# def createPixels(): #create matrix representing screen
#     global pixels
#     pixels = []
#     for i in range(0, WINDOW_SIZE):
#         pixels.append([])
#         for j in range(0, WINDOW_SIZE):
#             pixels[i].append(0)
#
#
# def addPixels(x, y):
#     global pixels
#     for i in range(x-1, x+2):
#         if i < 0 or i >= WINDOW_SIZE:
#             continue
#         for j in range(y-1, y+2):
#             if j < 0 or j >= WINDOW_SIZE:
#                 continue
#             pixels[i][j] = 255
#
# createPixels()
#
# #event handlers
# def drawline(event):
#     global pixels
#     x, y = event.x, event.y
#     if canvas.old_coords and mouse_pressed:
#         x1, y1 = canvas.old_coords
#         canvas.create_line(x, y, x1, y1)
#         addPixels(x, y)
#         addPixels(x1, y1)
#         #pixels[x][y] = 255
#         #pixels[x1][y1] = 255
#         #print(str(x) + " " + str(y))
#     canvas.old_coords = x, y
#
# def keydown(e):
#     printPixels()
#     if e.char == "c":
#         canvas.delete("all")
#         createPixels()
#     elif e.char == "d":
#         # predictDrawing(pixels, all_theta)
#         formatted = formatImage(pixels)
#         img = formatted.reshape((1, 1, 28, 28)).astype("float32") / 255
#         result = loaded_model.predict(img);
#         result = result[0]
#
#         max = np.where(result == np.amax(result))
#
#         print(chr(max[0] + 65));
#
#
# def pressed(event):
#     global mouse_pressed
#     mouse_pressed = True
#
# def released(event):
#     global mouse_pressed
#     mouse_pressed = False
#
# #window
# root = tk.Tk()
#
# root.geometry("" + str(WINDOW_SIZE) + "x" + str(WINDOW_SIZE))
#
# #create canvas
# canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE)
# canvas.pack()
# canvas.old_coords = None
#
# #binds
# root.bind('<Motion>', drawline)
# root.bind("<KeyPress>", keydown)
# root.bind("<Button-1>", pressed)
# root.bind("<ButtonRelease-1>", released)
#
# root.mainloop() #loop, no code after gets run
