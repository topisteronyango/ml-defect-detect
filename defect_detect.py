# -*- coding: utf-8 -*-
"""Copy of trial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gAvONbvwg7rSCv1oaDgZOslFDPGjKtFT
"""

from google.colab import drive
drive.mount('/content/drive')

# Directory paths
train_dir = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/train'
valid_dir = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/valid'
test_dir = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/test'

"""# Step 1: Preprocessing the Data"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# %matplotlib inline

CLASSES = 6       # Here, there are 6 classes of defects ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled','Scratches']
HEIGHT = 224
WIDTH = 224
CHANNELS = 3

baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(WIDTH, HEIGHT, CHANNELS)))

# Enable Transfer Learning by freezing weights of the base VGG16 Model
for layer in baseModel.layers:
	layer.trainable = False

model = Sequential()
model.add(baseModel)
model.add(Flatten(name="flatten"))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(CLASSES, activation='softmax'))

model.summary()

# Image preprocessing for robustness
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(WIDTH, HEIGHT),
                                                 batch_size=32,
                                                 class_mode='categorical')
validation_set = train_datagen.flow_from_directory(valid_dir,
                                                 target_size=(WIDTH, HEIGHT),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(WIDTH, HEIGHT),
                                            batch_size=232,
                                            class_mode='categorical',
                                            shuffle=False)

model.compile(
    loss="categorical_crossentropy",
    optimizer = Adam(lr=0.001),
    metrics=["accuracy"]
)

"""# New Section

# New Section
"""

Epochs = 10

history = model.fit(
    training_set,
    epochs=Epochs,
    steps_per_epoch=training_set.samples//32,
    validation_data=validation_set,
    validation_steps=validation_set.samples//32
)

score = model.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the trained model
model.save('defect_detection_model.h5')

from PIL import Image

def load_image(image_path):
    img = Image.open(image_path)
    return img

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(img):
    # Resize the image to the required input size (224x224 for VGG16)
    img = img.resize((224, 224))

    # Convert the image to RGB if it has a single channel (grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert PIL Image to numpy array
    img_array = img_to_array(img)

    # Normalize pixel values to be in the range [0, 1]
    img_array = img_array / 255.0

    return img_array

from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('defect_detection_model.h5')


# Use the loaded model to make predictions on new data
def detect_defect(image_path):
    img = load_image(image_path)  # Load the image (implement this function)
    img = preprocess_image(img)   # Preprocess the image (implement this function)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(np.expand_dims(img, axis=0))

    # Check the prediction result
    if prediction[0][0] >= 0.5:  # Assuming binary classification (defect or non-defect)
        return "Defect Detected"
    else:
        return "No Defect Detected"

# Replace 'test_image.jpg' with the path to your test image
test_image_path = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/test/Crazing/Cr_109.bmp'

result = detect_defect(test_image_path)
print(result)

# Replace 'test_image.jpg' with the path to your test image
test_image_path = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/test/Crazing/Cr_108.bmp'

result = detect_defect(test_image_path)
print(result)

# Replace 'test_image.jpg' with the path to your test image
test_image_path = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/test/Crazing/Cr_104.bmp'

result = detect_defect(test_image_path)
print(result)

# Replace 'test_image.jpg' with the path to your test image
test_image_path = '/content/drive/My Drive/ML/Transfer Learning/NEU Metal Surface Defects Data/train/Rolled/RS_183.bmp'

result = detect_defect(test_image_path)
print(result)

# Replace 'test_image.jpg' with the path to your test image
test_image_path = '/content/drive/My Drive/ML/Transfer Learning/img064.jpg'

result = detect_defect(test_image_path)
print(result)

