from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K
import sklearn
from keras.models import load_model
import pandas as pd  
from keras.preprocessing import image
from PIL import Image
import os
import keras
import keras.utils
from keras import utils as np_utils

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# dimensions of our images.
img_width, img_height = 400,400#400,400

test_data_dir = './testing_data'
nb_test_samples = 433#75
batch_size = 1

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')#should learn about sead


test_generator.reset()
model=load_model('speakermodel_1.h5')#changed
print("model loaded and result saved")


f1 = open("speakerresult_1.csv",'w')#changed
titlerow = "actual,predicted\n"
f1.write(titlerow)
for root, dirs, files in os.walk("./testing_data", topdown=False):
    if root == "./testing_data":
        for name in dirs:
            TEST_DIR="./testing_data/"+name+"/"  
            img_file=os.listdir(TEST_DIR)
            for f in (img_file):
                img = Image.open(TEST_DIR+f)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                preds = model.predict_classes(x)
                #label = { 0:"speaker0", 1:"speaker1", 2:"speaker2", 3:"speaker3", 4:"speaker4", 5:"speaker5", 6:"speaker6", 7:"speaker7", 8:"speaker8", 9:"speaker9", 10:"speaker10", 11:"speaker11", 12:"speaker12", 13:"speaker13", 14:"speaker14", 15:"speaker15", 16:"speaker16", 17:"speaker17", 18:"speaker18", 19:"speaker19"}
                label = { 0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11", 12:"12", 13:"13", 14:"14", 15:"15", 16:"16", 17:"17", 18:"18", 19:"19" }
                f1.write(name+","+ label[preds[0]] +"\n")#label[preds[0]]

f1.close()
                
