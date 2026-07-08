import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from cascade_utils import get_face_cascade

DATA_DIR = Path("dataset")
MODEL_PATH = Path("face_model.yml")
FACE_CASCADE = get_face_cascade()
RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()


def train():
    faces = []
    labels = []
    label_names = {}

    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        label = len(label_names)
        label_names[label] = person_dir.name
        for image_path in person_dir.glob("*.jpg"):
            image = Image.open(image_path).convert("L")
            face = np.array(image, dtype=np.uint8)
            faces.append(face)
            labels.append(label)

    if len(faces) == 0:
        raise RuntimeError("No training data found in dataset/")

    RECOGNIZER.train(faces, np.array(labels))
    RECOGNIZER.save(str(MODEL_PATH))
    print("Training complete. Model saved to", MODEL_PATH)
    print("Known labels:", label_names)


if __name__ == "__main__":
    train()
