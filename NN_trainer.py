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
import tensorflow as tf

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

include_sim_plates = False

PATH = "./training_data"
folder0 = PATH
if include_sim_plates:
	PATH_sim = "./sim_training_data"
	folder_sim = PATH_sim

files0 = [f for f in listdir(PATH) if isfile(join(PATH, f))]
if include_sim_plates:
	files_sim = [f for f in listdir(PATH_sim) if isfile(join(PATH_sim, f))]
#print(files_sim)
random.shuffle(files0)

if include_sim_plates:
	random.shuffle(files_sim)

#GET IMAGES FROM FOLDER
imgset0 = np.array([np.array(Image.open(folder0 + "/" + file))
                    for file in files0[:]])
print("Loaded {:} images from folder:\n{}".format(imgset0.shape[0], folder0))

#OVERALL PLATE DIMENSIONS CONSTANTS
RESIZE_WIDTH = 320 #must be multiple of 4
RESIZE_HEIGHT = 120

resize_width = RESIZE_WIDTH
resize_height = RESIZE_HEIGHT
split = RESIZE_WIDTH/4

INITIAL_RESIZE_WIDTH = 50
INITIAL_RESIZE_HEIGHT = 70

def split_images(imgset0,training_flag):

  #final overall plate dimensions
  resize_width = RESIZE_WIDTH
  resize_height = RESIZE_HEIGHT

  split = resize_width / 4
  #plate = imgset0[0]

  #from https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
  
  kernel_size = 15
  kernel_v = np.zeros((kernel_size, kernel_size))
  kernel_h = np.copy(kernel_v)
  kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
  kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
  kernel_v /= kernel_size
  kernel_h /= kernel_size

  #put all the letters in one big array
  #put that plate array into a bigger array
  first_plate = True
  for plate in imgset0:
    img_width = plate.shape[1]
    img_height = plate.shape[0]
    #Resize images
    #Found this function from https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    cutoff_margin = random.randint(10,30)
    if training_flag:
        #plate = plate[cutoff_margin:(plate.shape[0]-cutoff_margin),:]
        plate = cv.GaussianBlur(plate,(31,31),0)
        # from https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
        #plate = cv.filter2D(plate, -1, kernel_v)
        #plate = cv.filter2D(plate,-1,kernel_h)
        plate = cv.resize(plate, (INITIAL_RESIZE_WIDTH, INITIAL_RESIZE_HEIGHT))
        plate = cv.resize(plate, (img_width, img_height))
    resized_plate = cv.resize(plate, (resize_width, resize_height))
    #resized_plate = cv.GaussianBlur(resized_plate,(11,11),0)
    resized_plate = cv.cvtColor(resized_plate,cv.COLOR_BGR2RGB) #convert image colour back to what it usually is.
    #resized_plate = plate
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

'''
def crop_images(data):
	original_size = [resize_height,int(split)]
	crop_size = [resize_height-20, int(split)-20, 3]
	first_crop = True
	for i in range(len(data)):
		seed = np.random.randint(20)
		cropped = tf.image.random_crop(data[i], size = crop_size)
		print(cropped)
		cropped = tf.image.resize(cropped, size = original_size)
		cropped = np.array(cropped)
		print(cropped)
		if first_crop:
			ret_data = cropped
		else:
			ret_data = np.vstack((ret_data,cropped))
	return ret_data

X_dataset = crop_images(X_dataset)
'''

def shift_images(data):
	for i in range(len(data)):
		shift = tf.keras.preprocessing.image.random_shift(data[i],0.05,0.20,row_axis=0,col_axis=1,channel_axis=2,fill_mode='nearest')
		data[i] = shift
	return data

X_dataset = shift_images(X_dataset)

if include_sim_plates:
	#GET SIM IMAGES
	first_sim_image = True
	for i in range(len(files_sim)):
		sim_img = np.array(Image.open(folder_sim + "/" + files_sim[i]))
		sim_img = sim_img.reshape(1,sim_img.shape[0],sim_img.shape[1],3)
		split_sim_img = split_images(sim_img,training_flag=False)
		if first_sim_image:
			X_dataset_sim = split_sim_img
			first_sim_image = False
		else:
			X_dataset_sim = np.vstack((X_dataset_sim,split_sim_img))
	print("Loaded {:} images from folder:\n{}".format(X_dataset_sim.shape[0], folder_sim))


# Parse the Image Titles
def parse_image_titles(files):
	image_names_set = []
	for title in files:
		title = title.replace("plate_", "").replace(".png", "")
		image_names_set.append(title)
	#print(image_names_set)
	return image_names_set

image_names = parse_image_titles(files0)
if include_sim_plates:
	image_names_sim = parse_image_titles(files_sim)

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

#print(convert_to_one_hot("M",classes))

def get_Y_dataset(image_names_set):
	global classes
	first_run = True
	for plate in image_names_set:
		plate_encoding = convert_to_one_hot(plate, classes)
		if first_run:
			Y_data = plate_encoding
			first_run = False
		else:
			Y_data = np.vstack((Y_data, plate_encoding))
	return Y_data

Y_dataset = get_Y_dataset(image_names)
if include_sim_plates:
	Y_dataset_sim = get_Y_dataset(image_names_sim)

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
IDG_object = ImageDataGenerator(brightness_range=[0.20,1.0],rotation_range=1.5,
                                shear_range=2.0,zoom_range=[0.80,1.1])
xy_iterator = IDG_object.flow(x=(X_dataset,Y_dataset),shuffle=False,batch_size=1)
#displayImage(4)

for i in range(len(X_dataset)):
  xy = next(xy_iterator)
  X_dataset[i] = xy[0][0]
  Y_dataset[i] = xy[1][0]

def grey_images(dataset):
	dataset_grey = []
	first_run = True
	for i in range(len(dataset)):
		grey = cv.cvtColor(dataset[i], cv.COLOR_BGR2GRAY)
		#th, im_th = cv.threshold(grey, 128, 255, cv.THRESH_BINARY)
		if first_run:
			dataset_grey = grey.reshape(1,grey.shape[0],grey.shape[1],1)
			first_run = False
		else:
			dataset_grey = np.vstack((dataset_grey,grey.reshape(1,grey.shape[0],grey.shape[1],1)))
	return dataset_grey

X_dataset = grey_images(X_dataset)


'''cv.imshow("first",X_dataset[0])
cv.imshow("second",X_dataset[1])
cv.waitKey(5000)
cv.destroyAllWindows()
print(Y_dataset[0])
print(Y_dataset[1])'''

'''for x in X_dataset:
	cv.imshow("pic",x)
	cv.waitKey(3000)
	cv.destroyAllWindows()'''

def combine_datasets(X_data0, X_data1, Y_data0, Y_data1):
	X_dataset_comb = np.vstack((X_data0,X_data1))
	Y_dataset_comb = np.vstack((Y_data0,Y_data1))

	return X_dataset_comb, Y_dataset_comb

if include_sim_plates:
	X_dataset, Y_dataset = combine_datasets(X_dataset_sim,X_dataset,Y_dataset_sim,Y_dataset)

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
#conv_model.add(layers.Conv2D(32, (2, 2), activation='relu',
#                             input_shape=(resize_height, int(split), 3)))
conv_model.add(layers.Conv2D(32, (2, 2), activation='relu',
                             input_shape=(resize_height, int(split),1)))
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

def mapPredictionToCharacter(y_predict):
    #maps NN predictions to the numbers based on the max probability.
    y_predicted_max = np.max(y_predict)
    index_predicted = np.where(y_predict == y_predicted_max)
    character = classes[index_predicted]
    return character[0]

def testAllImages():
  total_count = 0
  train_count = 0
  val_count = 0
  y_true = []
  y_pred = []
  for i in range(len(X_dataset)):
    img = X_dataset[i]
    img_aug = np.expand_dims(img, axis=0)
    y_predict = conv_model.predict(img_aug)[0]
    predicted_character = mapPredictionToCharacter(y_predict)
    y_pred = np.append(y_pred,predicted_character)

    true_character = classes[np.argmax(Y_dataset[i])]
    y_true = np.append(y_true,true_character)

  return total_count, train_count, val_count, y_pred, y_true

total_count, train_count, val_count, y_pred, y_true = testAllImages()
#print("Total Accuracy out of 1: ", 1-total_count/float(len(X_dataset)))
#print("Train Accuracy out of 1: ", 1-train_count/((1-VALIDATION_SPLIT)*len(X_dataset)))
#print("Validation Accuracy out of 1 ", 1-val_count/float(VALIDATION_SPLIT*len(X_dataset)))
#print("length y_true", len(y_true))
#print("length y_pred", len(y_pred))

print("Confusion Matrix:\n")
#from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
confusion_matrix = confusion_matrix(y_true,y_pred,labels=classes)
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
'''
models.save_model(conv_model,"NN_object")
