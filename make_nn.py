import os
import sys
from functools import partial
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

EPOCHS = 1
DIR = sys.argv[1]
FILE = sys.argv[2]

labels = []

for root, dirs, files in os.walk(DIR):
    for file in files:
        if file[0] == "c":
          labels.append(0)
        if file[0] == "d":
           labels.append(1)

dataset = keras.utils.image_dataset_from_directory(
    DIR,
    labels=labels,
    label_mode="binary",
    color_mode="rgb",
    batch_size= 32,
    image_size=(100, 100),
    shuffle=True,
    verbose=True
)

DefaultConv2D = partial(layers.Conv2D, padding = "same", activation = "relu")

model = keras.Sequential([
   layers.Input((100, 100, 3)),
   DefaultConv2D(16, (7, 7)),
   layers.MaxPooling2D(),
   DefaultConv2D(32, (3, 3)),
   DefaultConv2D(32, (3, 3)),
   layers.MaxPooling2D(),
   DefaultConv2D(64, (3, 3)),
   DefaultConv2D(64, (3, 3)),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(64, activation="relu"),
   layers.Dropout(0.5),
   layers.Dense(16, activation="relu"),
   layers.Dropout(0.5),
   layers.Dense(1, activation="relu")
])

model.compile(
  optimizer="adam",
  loss=keras.losses.BinaryCrossentropy(),
  metrics=["accuracy"],
)

model.summary()

model.fit(dataset, epochs=EPOCHS)

model.save(FILE, include_optimizer = False)
