import os
import sys

import cv2
import numpy as np

CLASSES = ["objet1", "objet2", "objet3", "autre", "background"]
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
DEFAULT_IMG_SIZE = 64
CROP_WIDTH_RATIO = 0.6
TRAIN_RATIO = 0.8
SEED = 42


def output_file(img_size):
    suffix = "" if img_size == DEFAULT_IMG_SIZE else f"_{img_size}"
    return os.path.join(os.path.dirname(__file__), f"dataset{suffix}.npz")


def capture_order(filename):
    stem = os.path.splitext(filename)[0]
    index = stem.rsplit("_", 1)[-1]
    return int(index) if index.isdigit() else stem


def center_crop_width(img, ratio=CROP_WIDTH_RATIO):
    w = img.shape[1]
    crop_w = int(w * ratio)
    x0 = (w - crop_w) // 2
    return img[:, x0 : x0 + crop_w]


def load_class_images(class_name, img_size):
    class_dir = os.path.join(DATASET_DIR, class_name)
    images = []
    for filename in sorted(os.listdir(class_dir), key=capture_order):
        path = os.path.join(class_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = center_crop_width(img)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
    return np.array(images, dtype="uint8")


def split_train_test(images, label):
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    labels = np.full(n, label, dtype="int64")
    return images[:n_train], labels[:n_train], images[n_train:], labels[n_train:]


def build_dataset(img_size=DEFAULT_IMG_SIZE):
    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test = [], [], [], []

    for label, class_name in enumerate(CLASSES):
        images = load_class_images(class_name, img_size)
        print(f"{class_name}: {len(images)} images")
        if len(images) == 0:
            continue
        train_x, train_y, test_x, test_y = split_train_test(images, label)
        X_train.append(train_x)
        y_train.append(train_y)
        X_test.append(test_x)
        y_test.append(test_y)

    if not X_train:
        raise RuntimeError("Aucune image trouvée dans projet/dataset/*, rien à agréger.")

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    train_shuffle = rng.permutation(len(X_train))
    test_shuffle = rng.permutation(len(X_test))
    X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
    X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]

    out_file = output_file(img_size)
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(CLASSES),
    )

    print(f"\nDataset sauvegardé dans {out_file}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG_SIZE
    build_dataset(size)
