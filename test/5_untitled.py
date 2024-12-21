# -*- coding: utf-8 -*-
"""5-Untitled.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19_DRAaCiW2PcxnHEhjPPOtBrPz715EqQ
"""

# Importing libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
"""train_images = train_images[:10000]  # Take the first 10,000 samples from x_train
train_labels = train_labels[:10000]  # Take the first 10,000 samples from x_train"""

# Preprocessing the data
image_size = 28  # MNIST images are 28x28
batch_size = 32

# Reshape and normalize the images
train_images = train_images.reshape((-1, image_size, image_size, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, image_size, image_size, 1)).astype('float32') / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select a subset of the data (e.g., use only the first 10,000 samples)
x_train = x_train[:10000]  # Take the first 10,000 samples from x_train
y_train = y_train[:10000]  # Take the corresponding labels for those samples

# Reshape the data to be compatible with CNN input (height, width, channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalize the pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Resize the images to 64x64 as per your initial code
x_train_resized = tf.image.resize(x_train, (64, 64))
x_test_resized = tf.image.resize(x_test, (64, 64))

# Convert labels to categorical format (one-hot encoding)
y_train_categorical = tf.keras.utils.to_categorical(y_train, 10)
y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

# Building the model
input = Input(shape=(image_size, image_size, 1))
# 1st Conv Block
x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input)
x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

# 2nd Conv Block
x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

# 3rd Conv block
x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

# Fully connected layers
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=512, activation='relu')(x)
output = Dense(units=10, activation='softmax')(x)

# Creating the model
model = Model(inputs=input, outputs=output)
model.summary()

# Compiling the Model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# Training the Model
history = model.fit(
    train_images, train_labels,
    batch_size=batch_size,
    epochs=3,
    validation_data=(test_images, test_labels)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")