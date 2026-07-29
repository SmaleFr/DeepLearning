import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers

DATASET_FILE = os.path.join(os.path.dirname(__file__), "..", "dataset.npz")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model_v1.h5")

data = np.load(DATASET_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
class_names = data["class_names"]
num_classes = len(class_names)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=15,
    restore_best_weights=True,
    verbose=1,
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.path.dirname(__file__), "logs")
)

model = tf.keras.Sequential(
    [
        layers.Input(shape=(64, 64, 3)),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), 
        layers.RandomZoom(0.1), 

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),

        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ]
)

model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    verbose=2,
    callbacks=[early_stopping, tensorboard],
)
model.evaluate(X_test, y_test, verbose=2)

model.save(MODEL_FILE)
print(f"\nModèle sauvegardé dans {MODEL_FILE}")