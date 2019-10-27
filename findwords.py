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

K.set_image_dim_ordering('th') #sets depth, input_depth, rows, columns for the convolutional neural network


#LOAD THE MODEL---------------------------------------

# load YAML and create model
yaml_file = open('models/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model1 = model_from_yaml(loaded_model_yaml)
# load weights into new model
model1.load_weights("models/model.h5")
print("Loaded model 1 from disk")

# load YAML and create model
yaml_file = open('models/model3.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model2 = model_from_yaml(loaded_model_yaml)
# load weights into new model
model2.load_weights("models/model3.h5")
print("Loaded model 2 from disk")


#LOAD THE WORDSEARCH
wordsearch = cv2.imread('public/resources/search.jpg', 0)

wordsearch = cv2.bitwise_not(wordsearch)

#check if png: if it is, convert to jpg

#make white black and black white

#display image
# cv2.imshow("image", wordsearch)
# cv2.waitKey(0)

#get the number of rows and columns, and image dimensions

#LOAD DATA FROM JSON

def getRowsColumns():
    with open('public/resources/info.json') as json_file:
        data = json.load(json_file)
        return int(data["board"]["rows"]), int(data["board"]["columns"])

def getWords():
    with open('public/resources/info.json') as json_file:
        data = json.load(json_file)
        return (data["words"]["list"]).lower()

rows, columns = getRowsColumns()

if rows < 0 or columns < 0:
    exit()


words = getWords().split(",")

print(rows)
print(columns)
print(words)

#--------------

height, width = wordsearch.shape #height and width of image (in pixels)

#gets the bounds

letter_width = int(width / columns)
letter_height = int(height / rows)

print(letter_width)
print(letter_height)

#method for cropping the image

def crop_image(img,tol=1):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_two(img):
    mask = img > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    return img[x0:x1, y0:y1]

#get array format using classifier

def getFormattedBoard(model, rows, columns, letter_width, letter_height, picture, crop):
    newsearch = []

    for r in range(rows):

        row_letters = []

        for c in range(columns):
            # print(str(r) + " " + str(c))

            #subtracting helps with overshoot
            #adding helps with undershoot
            #right now, we need to account for undershooting
            width_movement = (c * letter_width) + int((letter_width / 3) * (c / columns))
            height_movement = (r * letter_height) + int((letter_height / 3) * (r / rows))

            if r == 0:
                height_movement = 0

            if c == 0:
                width_movement = 0

            #get the letter
            letter = np.copy(picture[height_movement:height_movement + letter_height, width_movement:width_movement+letter_width])


            #make all pixels either white or black
            #deals with highlighter
            for pixel_row in range(len(letter)):
                for pixel_col in range(len(letter[pixel_row])):
                    if(letter[pixel_row][pixel_col] > 150):
                        letter[pixel_row][pixel_col] = 255
                    else:
                        letter[pixel_row][pixel_col] = 0


            # cv2.imshow("image", letter)
            # cv2.waitKey(0)
            if crop:
                letter = crop_image(letter) #crop image

            # letter = crop_image_two(letter)
            # cv2.imshow("image", letter)
            # cv2.waitKey(0)
            letter = cv2.resize(letter, (28, 28)) #resize the image

            # cv2.imshow("image", letter)
            # cv2.waitKey(0)

            #reshape and normalize
            image = letter.reshape((1, 1, 28, 28)).astype("float32") / 255

            result = model.predict(image);
            result = result[0]

            # print(result)

            #find whats the highest probability
            max = np.where(result == np.amax(result))


            predicted_letter = chr(max[0] + 64)
            row_letters.append(predicted_letter.lower())
            # print(predicted_letter.lower())

        newsearch.append(row_letters)

    return newsearch


newsearch = getFormattedBoard(model1, rows, columns, letter_width, letter_height, wordsearch, False)
newsearch2 = getFormattedBoard(model2, rows, columns, letter_width, letter_height, wordsearch, True)

#now we find the words

def inRange(x, y, max_x, max_y):
    return x >= 0 and x < max_x and y >= 0 and y < max_y

def search(board, actual_word):
    for r in range(rows): #for each row
        for c in range(columns): #for each column
            letter = board[r][c] #get the letter
            positions = [[r, c]]
            if letter == actual_word[0:1]: #make sure the letter is the first letter in the word
                # print(str(r) + " " + str(c))
                #now we look in all 8 directions
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if a == 0 and b == 0: #middle spot
                            continue
                        positions = [[r, c]] #reset the list of positions
                        counter = 1
                        word = letter
                        while(word == actual_word[0:counter] and inRange(r + a * counter, c + b * counter, rows, columns)): #keep adding onto the word until you reach the border,
                            #or the word doesn't match the actual word
                            word += board[r + a * counter][c + b * counter]
                            positions.append([r + a * counter, c + b * counter])
                            counter += 1
                        if word[0:len(word)-1] == actual_word: #word found?
                            # print("found")
                            return positions[0:len(positions)-1]
                        elif word == actual_word:
                            # print("found")
                            return positions
    return []

# words = ["diagnosis" , "autoimmune", "cns", "hope", "support", "treatment", "fatigue"]
# words = ["alliteration", "antagonist", "aside", "characterization", "irony", "conflit", "hyperbole", "oxymoron", "plot", "theme", "simile"]
# words = ["acoustics", "anatomy", "anthropology", "biology", "chemistry", "kinetics", "geology", "Ecology", "gemology", "virology", "physics", "metrology"]
#wordsearch4
# words = ["allele", "antigen", "bacteria", "biology", "cell", "chitin", "ecosystem", "genotype", "homeostasis", "mitosis", "virus", "vacuole", "zygote", "xylem", "photosynthesis"]
#crosswordb
highlighted = []


for word in words:
    highlighted = search(np.copy(newsearch), word) + highlighted

print("*")
for word in words:
    highlighted = search(np.copy(newsearch2), word) + highlighted


#now we gotta highlight

# print(highlighted)

wordsearch = cv2.bitwise_not(wordsearch) #turn back into original

def white_black(wordsearch): #turns every pixel in wordsearch either black or white
    height, width = wordsearch.shape

    for r in range(height):
        for c in range(width):
            if wordsearch[r][c] > 150:
                wordsearch[r][c] = 255
            else:
                wordsearch[r][c] = 0
    return wordsearch

wordsearch = white_black(wordsearch)
wordsearch = cv2.cvtColor(wordsearch, cv2.COLOR_GRAY2BGR)

def highlight(picture, r, c, letter_width, letter_height, rows, columns): #highlights the letter in the row and column; passed in the actual board, the r and c index, and data of the board
    width_movement = (c * letter_width) + int((letter_width / 3) * (c / columns))
    height_movement = (r * letter_height) + int((letter_height / 3) * (r / rows))

    if r == 0:
        height_movement = 0

    if c == 0:
        width_movement = 0

    #get the region
    region = np.copy(picture[height_movement:height_movement + letter_height, width_movement:width_movement+letter_width])

    #now we highlight

    region = (region - (255, 255, 0)) / 1


    picture[height_movement:height_movement + letter_height, width_movement:width_movement+letter_width] = region

    return picture



for h in highlighted:
    wordsearch = highlight(wordsearch, h[0], h[1], letter_width, letter_height, rows, columns)

# cv2.imshow("image", wordsearch)
# cv2.waitKey(0)

cv2.imwrite("public/resources/answer.jpg", wordsearch)

#wordsearch has to have letters in the four corners on the edge of the picture
