import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers

EPOCHS = 25
DIR = sys.argv[1]
FILE = sys.argv[2]

augment_layers = [
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.Rescaling(1./255)
]

@tf.function
def process_path(x):
  file = tf.io.read_file(x)
  label = tf.strings.split(x, os.sep)[-1]
  label = tf.strings.substr(label, 0, 1)
  if tf.math.equal(label, "c"):
    label = 0.
  elif tf.math.equal(label, "d"):
    label = 1.
  else:
    label = -1.
  image = tf.io.decode_image(file, channels=3, expand_animations=False)
  return image, label

def augment_data(images):
  for layer in augment_layers:
    images = layer(images, training=True)
  return images

dataset = tf.data.Dataset.list_files(DIR + "/*.jpg")
dataset = dataset.shuffle(dataset.cardinality()).map(
  process_path, num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.map(
  lambda x, y: (augment_data(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.batch(128).prefetch(tf.data.AUTOTUNE)

inputs = keras.Input((100, 100, 3))

x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

prev_act = x

for size in [256, 512, 728]:
  x = layers.Activation("relu")(x)
  x = layers.SeparableConv2D(size, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)

  x = layers.Activation("relu")(x)
  x = layers.SeparableConv2D(size, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

  residual = layers.Conv2D(size, 1, strides=2, padding="same")(prev_act)
  x = layers.add([x, residual])
  prev_act = x

x = layers.SeparableConv2D(1024, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(1, activation=None)(x)

model = keras.Model(inputs, outputs)

model.compile(
  optimizer=keras.optimizers.Adam(3e-4),
  loss=keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

model.summary()

model.fit(dataset, epochs=EPOCHS)

model.save(FILE, include_optimizer = False)
