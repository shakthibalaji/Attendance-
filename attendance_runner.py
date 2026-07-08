import cv2
from pathlib import Path

from attendance_utils import LABELS, MODEL_PATH, load_label_names, mark_attendance
from cascade_utils import get_face_cascade


def run_attendance_loop():
    if not Path(MODEL_PATH).exists():
        raise RuntimeError("Model not found. Train the model first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    labels = load_label_names()

    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            label_id, confidence = recognizer.predict(face)
            name = labels.get(label_id, LABELS.get(label_id, "Unknown"))
            color = (0, 255, 0)
            if confidence > 70:
                name = "Unknown"
                color = (0, 0, 255)
            else:
                mark_attendance(name)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
