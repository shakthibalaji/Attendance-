# Smart Attendance System with Face Recognition

This project is a simple student attendance system that uses a webcam and face recognition to identify students and mark attendance automatically.

## Features
- Add students and store their names
- Capture face samples from the camera
- Train a face recognition model
- Run live attendance detection from the webcam
- Export attendance records to CSV
- Basic desktop interface for easier use

## Project Files
- `collect_faces.py` - Capture face images for a student
- `train_recognizer.py` - Train the face recognition model
- `attendance.py` - Run attendance marking from the webcam
- `app.py` - GUI-based desktop application
- `attendance_utils.py` - Core student, training, and attendance logic
- `attendance_runner.py` - Webcam recognition loop
- `test_attendance_utils.py` - Basic tests for the attendance utilities

## Requirements
Python 3.8+

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Collect student face samples
Run:

```bash
python collect_faces.py
```

Enter the student name when prompted and keep facing the camera.

### 2. Train the model
Run:

```bash
python train_recognizer.py
```

### 3. Start attendance marking
Run:

```bash
python attendance.py
```

### 4. Use the desktop app
Run:

```bash
python app.py
```

This opens a simple interface to add students, collect samples, train the model, and start attendance.

## Notes
- This is a beginner-friendly prototype.
- Accuracy depends on lighting, camera quality, and the number of face samples collected.
- For better performance in real classrooms, you can later upgrade to YOLO or deep learning-based face recognition.

## Testing
Run tests with:

```bash
python -m pytest -q
```
