import math
import numpy as np
import re
import random
from sklearn.metrics import confusion_matrix
import cv2 as cv

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from os import listdir
from os.path import isfile, join

#training CNN
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend

import sys

import seaborn as sn
import pandas as pd

conv_model = models.load_model("NN_object")

PATH_testing = "./testing_data"
folder_testing = PATH_testing
files_testing = [f for f in listdir(PATH_testing) if isfile(join(PATH_testing, f))]

# Parse the Image Titles
image_names = []
for title in files_testing:
  title = title.replace("plate_", "").replace(".png", "")
  image_names.append(title)
print(image_names)

# Generate classes
A_asci = 65
Z_asci = 90
classes = np.array([])
for i in range(Z_asci-A_asci+1):
  character = chr(A_asci+i)
  classes = np.append(classes, character)
for i in range(10):
  classes = np.append(classes, i)

def convert_to_one_hot(license_plate, class_array):
    first_run = True
    for char in license_plate:
      index = np.where(class_array == char)[0]
      char_encoding = np.array([])
      for i in range(len(class_array)):
        if i == index:
          char_encoding = np.append(char_encoding, 1)
        else:
          char_encoding = np.append(char_encoding, 0)
      if first_run:
        one_hot_encoding = char_encoding
        first_run = False
      else:
        one_hot_encoding = np.vstack((one_hot_encoding, char_encoding))
    return one_hot_encoding

first_run = True
for plate in image_names:
  plate_encoding = convert_to_one_hot(plate, classes)
  if first_run:
    Y_dataset = plate_encoding
    first_run = False
  else:
    Y_dataset = np.vstack((Y_dataset, plate_encoding))

#OVERALL PLATE DIMENSIONS CONSTANTS
RESIZE_WIDTH = 320 #must be multiple of 4
RESIZE_HEIGHT = 120

resize_width = RESIZE_WIDTH
resize_height = RESIZE_HEIGHT
split = RESIZE_WIDTH/4

INITIAL_RESIZE_WIDTH = 75
INITIAL_RESIZE_HEIGHT = 25

def split_images(imgset0,training_flag):

  #final overall plate dimensions
  resize_width = RESIZE_WIDTH
  resize_height = RESIZE_HEIGHT

  split = resize_width / 4
  #plate = imgset0[0]

  #put all the letters in one big array
  #put that plate array into a bigger array
  first_plate = True
  for plate in imgset0:
    #Resize images
    #Found this function from https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    resized_plate = cv.resize(plate, (resize_width, resize_height))
    resized_plate = cv.cvtColor(resized_plate,cv.COLOR_BGR2RGB) #convert image colour back to what it usually is.
    LL = resized_plate[:, 0:int(split)]
    LC = resized_plate[:, int(split):int(split*2)]
    RC = resized_plate[:, int(split*2):int(split*3)]
    RR = resized_plate[:, int(split*3):int(split*4)]
    if first_plate:
      X_dataset = np.stack((LL,LC,RC,RR))
      first_plate = False
    else:
      X_dataset = np.vstack((X_dataset,LL.reshape(1,int(resize_height),int(split),3),
                                  LC.reshape(1,int(resize_height),int(split),3),
                                  RC.reshape(1,int(resize_height),int(split),3),
                                  RR.reshape(1,int(resize_height),int(split),3)))
  return X_dataset

'''img1 = np.array(Image.open(folder + "/" + files[2]))
img1 = img1.reshape(1,img1.shape[0],img1.shape[1],3)
testing_set = split_images(img1,training_flag=False)
img = testing_set[1]

cv.imshow("first",img)
cv.waitKey(2000)
cv.destroyAllWindows()
  
img_aug = np.expand_dims(img, axis=0)
y_predict = conv_model.predict(img_aug)[0]
print("Model Prediction:")
print(y_predict)'''

def mapPredictionToCharacter(y_predict):
    #maps NN predictions to the numbers based on the max probability.
    y_predicted_max = np.max(y_predict)
    index_predicted = np.where(y_predict == y_predicted_max)
    character = classes[index_predicted]
    return character[0]

def grey_images(dataset):
    dataset_grey = []
    first_run = True
    for i in range(len(dataset)):
        grey = cv.cvtColor(dataset[i], cv.COLOR_BGR2GRAY)
        #th, im_th = cv.threshold(grey, 128, 255, cv.THRESH_BINARY)
        if first_run:
            dataset_grey = grey.reshape(1,grey.shape[0],grey.shape[1])
            first_run = False
        else:
            dataset_grey = np.vstack((dataset_grey,grey.reshape(1,grey.shape[0],grey.shape[1])))
    return dataset_grey

def testNN(files):
    y_pred = np.array([])
    y_true = np.array([])
    corr_count = 0
    tot_count = 0
    for i in range(len(files)):
        img = np.array(Image.open(folder_testing + "/" + files[i]))
        img = img.reshape(1,img.shape[0],img.shape[1],3)
        testing_set = split_images(img,training_flag=False)
        for j in range(len(testing_set)):
            grey = cv.cvtColor(testing_set[j], cv.COLOR_BGR2GRAY).reshape(testing_set[j].shape[0],testing_set[j].shape[1],1)
            #cv.imshow("current_specimen",grey)
            #cv.waitKey(1500)
            #cv.destroyAllWindows()
            img_aug = np.expand_dims(grey, axis=0)
            #img_aug = np.expand_dims(testing_set[j], axis=0)
            y_predict = conv_model.predict(img_aug)[0]

            predicted_character = mapPredictionToCharacter(y_predict)
            y_pred = np.append(y_pred,predicted_character)

            true_character = image_names[i][j]
            y_true = np.append(y_true,true_character)

            print("predicted: ", predicted_character)
            print("actual: ", true_character)
            print("\n")

            if predicted_character == true_character:
                corr_count+=1
            tot_count+=1

    print("num correct:", corr_count)
    print("num total:", tot_count)

    return y_true, y_pred

y_true, y_pred = testNN(files_testing)
np.set_printoptions(threshold=sys.maxsize)
'''print(y_true)
print(y_pred)'''
#from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
confusion_matrix = confusion_matrix(y_true,y_pred,labels=classes)
#print(confusion_matrix)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in classes],
                  columns = [i for i in classes])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

def testAllImages(dataset):
    for image in dataset:
        img_aug = np.expand_dims(img, axis=0)
        y_predict = conv_model.predict(img_aug)[0]
        y_predicted_max = np.max(y_predict)
        index_predicted = np.where(y_predict == y_predicted_max)
        print(index_predicted)
    return None