import tensorflow as tf
from tf import keras
from tf.keras.models import Sequential
from tf.keras.preprocessing.image import ImageDataGenerator

from tf.keras.callbacks import EarlyStopping
from tf.keras import layers

import matplotlib.pyplot as plt
import numpy as np


# Do Matplotlib extension below
# use this savefig call at the end of your graph instead of using plt.show()
# plt.savefig('static/images/my_plots.png')

#create data generator
print("\nLoading training data...")

training_data_generator = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05)

CLASS_MODE = "categorical"
COLOR_MODE= "grayscale"
TARGET_SIZE= (256,256)
BATCH_SIZE= 6

training_iterator= training_data_generator.flow_from_directory(
    "Covid19-dataset/train",
    class_mode = CLASS_MODE,
    color_mode= COLOR_MODE,
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE)

sample_batch_input, sample_batch_labels = training_iterator.next()
print(sample_batch_input.shape, sample_batch_labels.shape)


#test data iterator
print("\nLoading validation data...")

test_data_generator = ImageDataGenerator(
        rescale=1.0/255)

test_iterator = test_data_generator.flow_from_directory(
        'Covid19-dataset/test',
        class_mode=CLASS_MODE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE)
print("\nBuilding model...")