import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import keras

FILE = sys.argv[1]
IMGS = sys.argv[2:]

model = keras.models.load_model(FILE)

dataset = tf.data.Dataset.from_tensor_slices(IMGS).map(
  lambda x: tf.io.decode_image(tf.io.read_file(x), channels=3, expand_animations=False)
).batch(32)

predictions = model.predict(dataset)

for img, pred in zip(IMGS, predictions):
  predict = "Unknown"
  if pred > 0.5:
    predict = "Dog"
  elif pred < 0.5:
    predict = "Cat"

  print(img + ": " + predict)

