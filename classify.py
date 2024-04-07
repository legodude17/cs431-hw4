import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

FILE = sys.argv[1]
IMGS = sys.argv[2:]

model = keras.models.load_model(FILE)

dataset = tf.data.Dataset.from_tensor_slices(IMGS).map(
  lambda x: tf.io.decode_image(tf.io.read_file(x), channels=3, expand_animations=False)
).map(
  lambda x: keras.layers.Rescaling(1./255)(x)
).batch(16)

predictions = model.predict(dataset)

for img, pred in zip(IMGS, predictions):
  pred = float(keras.ops.sigmoid(pred[0]))
  predict = "Unknown"
  if pred > 0.5:
    predict = "Dog"
  elif pred < 0.5:
    predict = "Cat"

  print(img + ": " + predict)

