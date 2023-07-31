""" -*- coding: utf-8 -*-
AI_DLBDSEAIS02_task 3:Emotion_detection_in_images.ipynb
Original file is located at
https://colab.research.google.com/drive/1ojD8yCGLKULYMpYgD2PGbpYfYc7fT32L
"""

# Importing necessary libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint

print(tf.__version__)

# Mounting drive to access data
# from google.colab import drive

# Assigning Path for Dataset
# drive.mount('/content/drive')

# TRAIN_DIR and TEST_DIR are variables that point to train and test sets!
# uncomment this for 'AI half' set
#TRAIN_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI half/train/"
#TEST_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI half/test/"

# Define the input shape of the images
input_size = 48
BATCH_SIZE = 32
# 7 emotion categories for sets 'AI' and 'AI half'
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# uncomment this for 'AI' set
#TRAIN_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI/train/"
#TEST_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI/test/"

# uncomment this block for 'AI2' set
TRAIN_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI2/train/"
TEST_DIR = "C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI2/test/"
input_size = 208
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'] # 6 emotion categories for sets 'AI2'

nr_of_emotions = len(emotions)

# Plotting random 4 emotions from each set
plt.figure(figsize=(20, 15))

for label in range(nr_of_emotions):
  img_folder = TRAIN_DIR + emotions[label]

  for x in range(4):
      # 4 images per emotion
      plt.subplot(len(emotions), 4, label*4+x+1) 
      # random image within specific emotion
      random_image = random.choice(os.listdir(img_folder))
      # image drawing with matplotlib
      img = mpimg.imread(img_folder + '/' + random_image)
      plt.imshow(img)
      plt.title(emotions[label])
      plt.axis('off')
plt.show()

# Preparing learning set
# Data preprocessing / normalization
train_datagen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2)

# Define data generator for testing images
test_datagen = ImageDataGenerator(rescale = 1./255)

# Define data flow for training images
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (input_size, input_size),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',# Change color_mode to grayscale
    class_mode='categorical')

# Define data flow for testing images
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (input_size,input_size),
    batch_size = BATCH_SIZE,
    color_mode='grayscale',# Change color_mode to grayscale
    class_mode = 'categorical')


# Initialisation
# Define the model

model = Sequential()

# Add convolutional layers with increasing filter number
model.add(Conv2D(16, kernel_size = (3, 3), activation ='relu', input_shape = (input_size, input_size, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

# Flatten the output of the last convolutional layer
model.add(Flatten())

# Dense layers with dropout to prevent overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nr_of_emotions, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint filename
checkpoint_filename = 'model_checkpoint.h5'

# Define the checkpoint callback to save the model every epoch
checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=False, save_weights_only=False)

# Callback to reduce waiting when fitting a model save for eventual use
"""
class firstCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.80): # Experiment with changing this value
      print("\nReached 80% accuracy, cancel training!")
      self.model.stop_training = True
callbacks = firstCallback()      
"""

# Train the model

history = model.fit(
    train_generator,
    epochs = 50, 
    batch_size = 32,
    validation_data = test_generator)

# Save the final model
model.save(TRAIN_DIR + 'model.h5')

# Plot the loss and accuracy for both the training and validation sets

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


input_shape = (input_size, input_size, 1)
# Load the saved model
model = load_model(TRAIN_DIR + 'model.h5')

# Load the image
img_path = 'C:/Users/Szilvi/Meine Ablage/Colab Notebooks/AI/user_images/image1.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (input_size, input_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # make grayscale

# Reshape the image to match the input shape of the model
img = img.reshape((1,) + img.shape)

# Normalize the image
img = img / 255.0

# Make a prediction
prediction = model.predict(img)

# Get the predicted emotion

predicted_emotion = emotions[np.argmax(prediction)]

# Display the test image with predicted emotion label
# print (np.shape(img))
plt.imshow(img.reshape(input_size, input_size ,1))
plt.title(predicted_emotion)
plt.axis('off')
plt.show()




