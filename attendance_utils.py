import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from cascade_utils import get_face_cascade

DATA_DIR = Path("dataset")
STUDENTS_FILE = Path("students.json")
MODEL_PATH = Path("face_model.yml")
ATTENDANCE_DIR = Path("attendance_logs")
FACE_CASCADE = get_face_cascade()
LABELS = {}


def load_students() -> List[str]:
    if not STUDENTS_FILE.exists():
        return []
    try:
        data = json.loads(STUDENTS_FILE.read_text(encoding="utf-8"))
        return [str(item).strip().title() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        return []


def save_students(students: List[str]) -> None:
    clean_students = [str(name).strip().title() for name in students if str(name).strip()]
    STUDENTS_FILE.write_text(json.dumps(clean_students, indent=2), encoding="utf-8")


def add_student(name: str) -> str:
    normalized_name = str(name).strip().title()
    if not normalized_name:
        raise ValueError("Student name cannot be empty")

    students = load_students()
    if normalized_name not in students:
        students.append(normalized_name)
        save_students(students)
    (DATA_DIR / normalized_name).mkdir(parents=True, exist_ok=True)
    return normalized_name


def collect_faces(student_name: str, sample_count: int = 25) -> int:
    student_dir = DATA_DIR / student_name
    student_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    count = 0
    print(f"Collecting samples for {student_name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(str(student_dir / f"{count}.jpg"), face_img)
            count += 1
            if count >= sample_count:
                break

        cv2.putText(frame, f"Samples: {count}/{sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect Samples", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or count >= sample_count:
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def load_label_names() -> dict:
    label_names = {}
    if not DATA_DIR.exists():
        return label_names
    for index, person_dir in enumerate(sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])):
        label_names[index] = person_dir.name
    return label_names


def train_model() -> dict:
    faces = []
    labels = []
    label_names = {}

    dataset_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    if not dataset_dirs:
        raise RuntimeError("No student folders found in dataset/")

    for person_dir in dataset_dirs:
        label = len(label_names)
        label_names[label] = person_dir.name
        for image_path in person_dir.glob("*.jpg"):
            image = Image.open(image_path).convert("L")
            face = np.array(image, dtype=np.uint8)
            faces.append(face)
            labels.append(label)

    if len(faces) == 0:
        raise RuntimeError("No training images found. Collect samples first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(str(MODEL_PATH))
    global LABELS
    LABELS = label_names
    return label_names


def mark_attendance(name: str) -> bool:
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = ATTENDANCE_DIR / f"attendance_{today}.csv"

    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    if any(row.get("name") == name for row in rows):
        return False

    rows.append({"name": name, "time": datetime.now().strftime("%H:%M:%S")})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "time"])
        writer.writeheader()
        writer.writerows(rows)
    return True


def export_attendance_csv(output_path: str | None = None) -> Path:
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(ATTENDANCE_DIR / f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    export_path = Path(output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_file in sorted(ATTENDANCE_DIR.glob("attendance_*.csv")):
        with csv_file.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            all_rows.append({"date": csv_file.stem.replace("attendance_", ""), **row})

    with export_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "name", "time"])
        writer.writeheader()
        writer.writerows(all_rows)

    return export_path
