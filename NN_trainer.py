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
#from keras import layers
#from keras import models
#from keras import optimizers

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend
#from keras.utils import plot_model
#from keras import backend

import sys

import seaborn as sn
import pandas as pd

PATH = "./training_data"
folder0 = PATH

files0 = [f for f in listdir(PATH) if isfile(join(PATH, f))]
random.shuffle(files0)

#GET IMAGES FROM FOLDER
imgset0 = np.array([np.array(Image.open(folder0 + "/" + file))
                    for file in files0[:]])
print("Loaded {:} images from folder:\n{}".format(imgset0.shape[0], folder0))

#OVERALL PLATE DIMENSIONS CONSTANTS
RESIZE_WIDTH = 200 #must be multiple of 4
RESIZE_HEIGHT = 70

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
    cutoff_margin = random.randint(10,30)
    if training_flag:
        plate = plate[cutoff_margin:(plate.shape[0]-cutoff_margin),:]
        blur = cv.GaussianBlur(plate,(31,31),0)
        plate = cv.resize(blur, (INITIAL_RESIZE_WIDTH, INITIAL_RESIZE_HEIGHT))
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

X_dataset = split_images(imgset0,training_flag=True)

# Parse the Image Titles
image_names = []
for title in files0:
  title = title.replace("plate_", "").replace(".png", "")
  image_names.append(title)
print(image_names)

'''cv.imshow(image_names[0],X_dataset[0])
cv.imshow(image_names[1],X_dataset[4])
cv.waitKey(3000)
cv.destroyAllWindows()'''

# Generate classes
A_asci = 65
Z_asci = 90
classes = np.array([])
for i in range(Z_asci-A_asci+1):
  character = chr(A_asci+i)
  classes = np.append(classes, character)
for i in range(10):
  classes = np.append(classes, i)

NUMBER_OF_LABELS = 36
NUMBERS_ON_PLATE = 4
CONFIDENCE_THRESHOLD = 0.01
#get the license plate one hot as array
#append each license plate array onto a Y dataset array
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
''' 
# Normalize X (images) dataset
X_dataset = X_dataset_orig/255.

# Convert Y dataset to one-hot encoding
Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T
'''

first_run = True
for plate in image_names:
  plate_encoding = convert_to_one_hot(plate, classes)
  if first_run:
    Y_dataset = plate_encoding
    first_run = False
  else:
    Y_dataset = np.vstack((Y_dataset, plate_encoding))

# Display images in the training data set. 
def displayImage(letter):
  plt.imshow(X_dataset[letter])
  caption = ("y = " + str(Y_dataset[letter]))#str(np.squeeze(Y_dataset_orig[:, index])))
  plt.text(0.5, 0.5, caption, 
           color='orange', fontsize = 20,
           horizontalalignment='left', verticalalignment='top')

#displayImage(4)


#BLUR THE DATA
#grabbed these functions from https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
#num_blur_imgs = random.randint(len(X_dataset)/2,len(X_dataset)-1)
#print(num_blur_imgs)

'''for i in range(len(X_dataset)):
  img = X_dataset[i]
  
  if i % 3 == 0:
    pass
  elif i % 3 - 1 == 0:
    blur = cv.GaussianBlur(img,(11,11),0)
    X_dataset[i] = blur
  else:
    blur = cv.GaussianBlur(img,(15,15),0)
    X_dataset[i] = blur
  blur = cv.GaussianBlur(img,(15,15),0)
  X_dataset[i] = blur
'''


#AUGMENT THE DATA
IDG_object = ImageDataGenerator(brightness_range=[0.35,1.0],rotation_range=1.5,
                                shear_range=2.0)
xy_iterator = IDG_object.flow(x=(X_dataset,Y_dataset),shuffle=False,batch_size=1)
#displayImage(4)

for i in range(len(X_dataset)):
  xy = next(xy_iterator)
  X_dataset[i] = xy[0][0]
  Y_dataset[i] = xy[1][0]


'''cv.imshow("first",X_dataset[0])
cv.imshow("second",X_dataset[1])
cv.waitKey(3000)
cv.destroyAllWindows()
print(Y_dataset[0])
print(Y_dataset[1])'''

VALIDATION_SPLIT = 0.2

print("Total examples: %d \nTraining examples: %d \nTest examples: %d" % (X_dataset.shape[0],
            math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)),
            math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
print("X shape: " + str(X_dataset.shape))
print("Y shape: " + str(Y_dataset.shape))


#RESET WEIGHTS
#function for reinitializing the model parameters
# Source: https://stackoverflow.com/questions/63435679
def reset_weights(model):
  for ix, layer in enumerate(model.layers):
      if (hasattr(model.layers[ix], 'kernel_initializer') and 
          hasattr(model.layers[ix], 'bias_initializer')):
          weight_initializer = model.layers[ix].kernel_initializer
          bias_initializer = model.layers[ix].bias_initializer

          old_weights, old_biases = model.layers[ix].get_weights()

          model.layers[ix].set_weights([
              weight_initializer(shape=old_weights.shape),
              bias_initializer(shape=len(old_biases))])

#Model Definition

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (2, 2), activation='relu',
                             input_shape=(resize_height, int(split), 3)))
conv_model.add(layers.MaxPooling2D((5, 5)))
conv_model.add(layers.Conv2D(12, (2, 2), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu')) #512
conv_model.add(layers.Dense(36, activation='softmax')) #36

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

np.set_printoptions(threshold=sys.maxsize)
'''cv.imshow("first",X_dataset[-1])
cv.imshow("second",X_dataset[-2])
cv.waitKey(5000)
cv.destroyAllWindows()
print("first", Y_dataset[-1])
print("second", Y_dataset[-2])'''

history_conv = conv_model.fit(X_dataset, Y_dataset, 
                              validation_split=VALIDATION_SPLIT, 
                              epochs=50, 
                              batch_size=16) #20 epochs

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

#Test Model
np.set_printoptions(threshold=sys.maxsize)

# Display images in the training data set. 
def checkImage(letter):
  img = X_dataset[letter]
  
  img_aug = np.expand_dims(img, axis=0)
  y_predict = conv_model.predict(img_aug)[0]
  print("Model Prediction:")
  print(y_predict)
  print("\n")
  print("Actual: ")
  print(Y_dataset[letter])
  index = np.where(Y_dataset[letter] == 1)
  print("Letter Confidence: ", y_predict[index])
  print("Confusion Matrix: ")
  print("\n")

def testAllImages():
  total_count = 0
  train_count = 0
  val_count = 0
  y_true = []
  y_pred = []
  for image in range(len(X_dataset)):
    img = X_dataset[image]
    img_aug = np.expand_dims(img, axis=0)
    y_predict = conv_model.predict(img_aug)[0]
    index_encoding = np.where(Y_dataset[image] == 1)
    if y_predict[index_encoding[0][0]] < 0.90:
      total_count+=1
      if image < (1-0.2)*len(X_dataset):
        train_count+=1
      else:
        val_count+=1

    y_predicted_max = np.max(y_predict)
    index_predicted = np.where(y_predict == y_predicted_max)
    y_pred.append(index_predicted[0][0])
    y_true.append(index_encoding[0][0])
  return total_count, train_count, val_count, y_pred, y_true

total_count, train_count, val_count, y_pred, y_true = testAllImages()
print(len(X_dataset))
#print("Total Accuracy out of 1: ", 1-total_count/float(len(X_dataset)))
#print("Train Accuracy out of 1: ", 1-train_count/((1-VALIDATION_SPLIT)*len(X_dataset)))
#print("Validation Accuracy out of 1 ", 1-val_count/float(VALIDATION_SPLIT*len(X_dataset)))
#print("length y_true", len(y_true))
#print("length y_pred", len(y_pred))
print(y_true)
print(y_pred)
print("Confusion Matrix:\n")
confusion_matrix = confusion_matrix(y_true,y_pred)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in classes],
                  columns = [i for i in classes])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()


'''
PATH_testing = "./testing_data"
folder1 = PATH_testing

files1 = [f for f in listdir(PATH_testing) if isfile(join(PATH_testing, f))]
#random.shuffle(files1)

img1 = np.array(Image.open(folder1 + "/" + files1[2]))

img1 = img1.reshape(1,img1.shape[0],img1.shape[1],3)
print(img1.shape)

testing_set = split_images(img1,training_flag=False)

img = testing_set[3]
  
img_aug = np.expand_dims(img, axis=0)
y_predict = conv_model.predict(img_aug)[0]
print("Model Prediction:")
print(y_predict)

cv.imshow("first",img)
cv.imshow("second",testing_set[0])
cv.imshow("third",testing_set[2])
cv.imshow("fourth",testing_set[3])
cv.imshow("training1",X_dataset[0])
cv.imshow("training2",X_dataset[1])
cv.imshow("training3",X_dataset[2])
cv.imshow("training4",X_dataset[3])
cv.waitKey(10000)
cv.destroyAllWindows()


print(X_dataset.shape)
print(img.shape)

models.save_model(conv_model,"NN_object")
'''