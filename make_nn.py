# Imports
import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers

# Constants
EPOCHS = 25 # 25 epochs takes about 17 hours on my laptop, but gives around 95% accuracy
DIR = sys.argv[1]
FILE = sys.argv[2]

# Data Loading

# Data augmentation layers
augment_layers = [
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.Rescaling(1./255)
]

# Process a path, by loading the image and getting the label
@tf.function
def process_path(x):
  label = tf.strings.split(x, os.sep)[-1]
  label = tf.strings.substr(label, 0, 1)
  if tf.math.equal(label, "c"):
    label = 0.
  elif tf.math.equal(label, "d"):
    label = 1.
  else:
    label = -1.
  file = tf.io.read_file(x)
  image = tf.io.decode_image(file, channels=3, expand_animations=False)
  return image, label

# Augment the data using the augmentation layers
def augment_data(images):
  for layer in augment_layers:
    images = layer(images, training=True)
  return images

dataset = tf.data.Dataset.list_files(DIR + "/*.jpg") # Get all jpgs in the trainging folder
dataset = dataset.shuffle(dataset.cardinality()).map( # Shuffle them fully
  process_path, num_parallel_calls=tf.data.AUTOTUNE # Process all the paths
)
dataset = dataset.map(
  lambda x, y: (augment_data(x), y), num_parallel_calls=tf.data.AUTOTUNE # Augment all the data
)
dataset = dataset.batch(128).prefetch(tf.data.AUTOTUNE) # Batch and prefetch the data, to increase training performance

# Building the Model

inputs = keras.Input((100, 100, 3)) # Inputs

# First convolution
x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)


# Create the residual layers
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

# Final convolution
x = layers.SeparableConv2D(1024, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# Output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation=None)(x) # Outputs

# Create the model
model = keras.Model(inputs, outputs)

# Compile the model, we're using binary mode, since we have two classes
model.compile(
  optimizer=keras.optimizers.Adam(3e-4),
  loss=keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# Print a summary of the model
#model.summary()

# Fit the model to the dataset
model.fit(dataset, epochs=EPOCHS)

# Save the model
model.save(FILE, include_optimizer = False)
