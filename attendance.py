import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from cascade_utils import get_face_cascade

MODEL_PATH = Path("face_model.yml")
FACE_CASCADE = get_face_cascade()
RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
RECOGNIZER.read(str(MODEL_PATH))

LABELS = {
    0: "Unknown"
}


def load_labels():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train the model first.")

    # Fallback names from dataset folder
    dataset_dir = Path("dataset")
    label_names = {}
    for person_dir in dataset_dir.iterdir():
        if person_dir.is_dir():
            label_names[len(label_names)] = person_dir.name
    return label_names


LABELS = load_labels()


def mark_attendance(name: str):
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = Path(f"attendance_{today}.csv")
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not any(row.get("name") == name for row in rows):
        rows.append({"name": name, "time": datetime.now().strftime("%H:%M:%S")})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "time"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Attendance marked for {name}")


def run_attendance():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            label_id, confidence = RECOGNIZER.predict(face)
            name = LABELS.get(label_id, "Unknown")
            if confidence < 70:
                mark_attendance(name)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()
