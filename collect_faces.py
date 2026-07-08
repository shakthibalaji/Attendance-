import cv2
import os
from pathlib import Path

from cascade_utils import get_face_cascade

DATA_DIR = Path("dataset")
FACE_CASCADE = get_face_cascade()


def collect_faces(student_name: str, sample_count: int = 30):
    student_dir = DATA_DIR / student_name
    student_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    count = 0
    print(f"Collecting face samples for {student_name}. Look at the camera...")

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
        cv2.imshow("Collect Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= sample_count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples for {student_name}")


if __name__ == "__main__":
    name = input("Enter student name: ").strip()
    collect_faces(name)
