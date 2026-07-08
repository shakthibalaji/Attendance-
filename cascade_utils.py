from pathlib import Path
import urllib.request
import cv2

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_DIR = Path(__file__).resolve().parent / "cascades"
CASCADE_PATH = CASCADE_DIR / "haarcascade_frontalface_default.xml"


def get_face_cascade():
    candidate_paths = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml",
        CASCADE_PATH,
        Path("haarcascade_frontalface_default.xml"),
    ]

    for path in candidate_paths:
        if path.exists():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                return cascade

    CASCADE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(CASCADE_URL, str(CASCADE_PATH))
    except Exception:
        pass

    if CASCADE_PATH.exists():
        cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if not cascade.empty():
            return cascade

    raise FileNotFoundError("Unable to find haarcascade_frontalface_default.xml")
