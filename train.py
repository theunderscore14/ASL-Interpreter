import numpy as np
import PIL
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=36, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'ASL_Datasets/Train',
    target_size=(64, 64),
    batch_size=1,
    color_mode='grayscale',
    class_mode='categorical')

test_set = test_datagen.flow_from_directory('ASL_Datasets/Test',
    target_size=(64, 64),
    batch_size=1,
    color_mode='grayscale',
    class_mode='categorical')

model.fit(
    training_set,
    steps_per_epoch=720,
    epochs=10,
    validation_data=test_set,
    validation_steps=72)

test_loss, test_accuracy = model.evaluate(test_set)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model_bw.weights.h5')
