"""Démo webcam gamifiée : charge un modèle entraîné et demande à l'utilisateur
de montrer chaque objet devant la caméra, en le maintenant 3 secondes.

Usage:
    python demo.py vX/model_vX.h5
"""

import random
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50

CLASSES = ["objet1", "objet2", "objet3", "autre", "background"]
CAMERA_INDEX = 0
CROP_WIDTH_RATIO = 0.6
RESNET_INPUT_SIZE = 256
HOLD_DURATION = 3.0



def preprocess(frame, img_size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype("float32")
    if img_size == RESNET_INPUT_SIZE:
        img = resnet50.preprocess_input(img)
    else:
        img = img / 255.0
    return np.expand_dims(img, axis=0)


def draw_text(frame, lines, color=(0, 255, 0)):
    y = 30
    for line in lines:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )
        y += 40
    return frame


def predict(model, frame, img_size):
    predictions = model.predict(preprocess(frame, img_size), verbose=0)[0]
    label_idx = int(np.argmax(predictions))
    return CLASSES[label_idx], float(predictions[label_idx])


def play_round(cap, model, img_size, target):
    hold_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        label, confidence = predict(model, frame, img_size)

        if label == target:
            if hold_start is None:
                hold_start = time.time()
            held_for = time.time() - hold_start
            if held_for >= HOLD_DURATION:
                draw_text(frame, ["Bravo !"], color=(0, 255, 0))
                cv2.imshow("Demo", frame)
                cv2.waitKey(800)
                return True
            draw_text(
                frame,
                [
                    f"Montre-moi : {target}",
                    f"Maintiens... {held_for:.1f}s / {HOLD_DURATION:.0f}s",
                ],
                color=(0, 255, 255),
            )
        else:
            hold_start = None
            draw_text(
                frame,
                [f"Montre-moi : {target}", f"Vu : {label} ({confidence:.0%})"],
                color=(0, 0, 255),
            )

        cv2.imshow("Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python demo.py <chemin_vers_model.h5>")
        sys.exit(1)

    model = tf.keras.models.load_model(sys.argv[1])
    img_size = model.input_shape[1]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    try:
        targets = [c for c in CLASSES if c != "background"]
        random.shuffle(targets)

        for target in targets:
            if not play_round(cap, model, img_size, target):
                return

        ret, frame = cap.read()
        if ret:
            draw_text(frame, ["Partie terminee !"], color=(0, 255, 0))
            cv2.imshow("Demo", frame)
            cv2.waitKey(2000)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
