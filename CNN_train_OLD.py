from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# dimensions of our images.
img_width, img_height = 200, 200


#########################
### Tunables
#########################
train_data_dir = './training_data'
test_data_dir = './testing_data'
nb_train_samples = 1583
nb_test_samples = 538
epochs = 10
batch_size = 30

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)#channel_last

#########################
### Setup the model
#########################


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))#32 filters
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
    

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


#########################
### Setup the generators
###fits the model on batches with real-time data augmentation
#########################


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical')


#########################
### Build the model
#########################

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size)


#########################
### Performance evaluation
#########################
score = model.evaluate_generator(test_generator,nb_test_samples/batch_size)
print(" Total: ", len(test_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])
#print("Accuracy = ",score[1])

model.save('speakermodel.h5')

