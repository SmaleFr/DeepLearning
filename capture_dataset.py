"""Capture webcam pour constituer un dataset d'images multi-classes.

Classes : 4 objets cibles + "autre" (objet inconnu) + "background" (rien devant la caméra).
Rafale automatique minutée par classe, déclenchée au clavier.
"""

import os

import cv2

CLASSES = ["objet1", "objet2", "objet3", "autre", "background"]
KEY_TO_CLASS = {
    ord("a"): "objet1",
    ord("z"): "objet2",
    ord("e"): "objet3",
    ord("o"): "autre",
    ord("b"): "background",
}
OUTPUT_DIR = "projet/dataset"
CAMERA_INDEX = 0
BURST_COUNT = 30
BURST_INTERVAL = 0.4
IMG_EXTENSION = "jpg"
CLASS_LIMITS = {}


def ensure_class_dirs():
    for class_name in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)


def next_index(class_dir, class_name):
    existing = [
        f
        for f in os.listdir(class_dir)
        if f.startswith(f"{class_name}_") and f.endswith(f".{IMG_EXTENSION}")
    ]
    if not existing:
        return 0
    indices = []
    for f in existing:
        stem = f[len(class_name) + 1 : -(len(IMG_EXTENSION) + 1)]
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices, default=-1) + 1


def count_photos(class_name):
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    return len(os.listdir(class_dir)) if os.path.isdir(class_dir) else 0


def draw_overlay(frame, status_lines):
    y = 20
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return frame


def idle_status_lines():
    def fmt(c):
        limit = CLASS_LIMITS.get(c)
        return f"{c} : {count_photos(c)}/{limit}" if limit else f"{c} : {count_photos(c)}"

    counts = "\n".join(fmt(c) for c in CLASSES)
    return [
        "a-z-e: objet1-3 | o: autre | b: background | q: quitter",
        counts
    ]


def run_burst(cap, class_name, count=BURST_COUNT, interval=BURST_INTERVAL):
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    limit = CLASS_LIMITS.get(class_name)
    if limit is not None:
        remaining = limit - count_photos(class_name)
        if remaining <= 0:
            print(f"Limite de {limit} photos atteinte pour '{class_name}', rafale ignorée.")
            return
        count = min(count, remaining)

    index = next_index(class_dir, class_name)
    taken = 0
    while taken < count:
        ret, frame = cap.read()
        if not ret:
            continue

        filename = os.path.join(class_dir, f"{class_name}_{index}.{IMG_EXTENSION}")
        cv2.imwrite(filename, frame)
        index += 1
        taken += 1

        preview = frame.copy()
        draw_overlay(
            preview,
            [f"Rafale [{class_name}]: {taken}/{count} (ESC pour annuler)"],
        )
        cv2.imshow("Capture dataset", preview)

        key = cv2.waitKey(int(interval * 1000)) & 0xFF
        if key == 27:  # ESC
            break


def main():
    ensure_class_dirs()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            draw_overlay(frame, idle_status_lines())
            cv2.imshow("Capture dataset", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in KEY_TO_CLASS:
                run_burst(cap, KEY_TO_CLASS[key])
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
