import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.applications import resnet50

DATASET_FILE = os.path.join(os.path.dirname(__file__), "..", "dataset_256.npz")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model_v2.h5")
IMG_SIZE = 256
FINE_TUNE_LAYERS = 10

data = np.load(DATASET_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
class_names = data["class_names"]
num_classes = len(class_names)

X_train = resnet50.preprocess_input(X_train.astype("float32"))
X_test = resnet50.preprocess_input(X_test.astype("float32"))

def make_early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.path.dirname(__file__), "logs")
)

Model_ResNet50 = resnet50.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'
)
Model_ResNet50.trainable = False

model = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        Model_ResNet50,

        layers.Dropout(0.2),
        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
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
    epochs=20,
    verbose=2,
    callbacks=[make_early_stopping(), tensorboard],
)

print(f"=== Fine-tuning ===")
Model_ResNet50.trainable = True
for layer in Model_ResNet50.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in Model_ResNet50.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.00001),
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    verbose=2,
    callbacks=[make_early_stopping(), tensorboard],
)
model.evaluate(X_test, y_test, verbose=2)

model.save(MODEL_FILE)
print(f"\nModèle sauvegardé dans {MODEL_FILE}")
