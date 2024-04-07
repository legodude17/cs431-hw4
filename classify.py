# Imports
import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

# Constants
FILE = sys.argv[1]
IMGS = sys.argv[2:]

model = keras.models.load_model(FILE) # Load the model

# Loading the data

dataset = tf.data.Dataset.from_tensor_slices(IMGS).map( # Load the given images
  lambda x: tf.io.decode_image(tf.io.read_file(x), channels=3, expand_animations=False), num_parallel_calls=tf.data.AUTOTUNE # Load the images
).map(
  lambda x: keras.layers.Rescaling(1./255)(x) # Rescale the data, like the trainer does
).batch(16).prefetch(tf.data.AUTOTUNE) # Batch and prefetch the data

predictions = model.predict(dataset) # Make the predictions

# Go through each prediction and print it
for img, pred in zip(IMGS, predictions):
  pred = float(keras.ops.sigmoid(pred[0])) # Result of the sigmoid is a probability
  # Closer to 1 is a dog, closer to 0 is a cat, since that's the labels I chose
  if pred > 0.5:
    predict = "Dog"
  elif pred < 0.5:
    predict = "Cat"

  print(img + ": " + predict) # Print the prediction

