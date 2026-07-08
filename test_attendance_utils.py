import os
import shutil
import tempfile
import unittest
from pathlib import Path

from attendance_utils import (
    ATTENDANCE_DIR,
    DATA_DIR,
    STUDENTS_FILE,
    add_student,
    export_attendance_csv,
    load_students,
    mark_attendance,
)


class AttendanceUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="attendance_test_", dir=".")
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(Path(__file__).resolve().parent)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_student_and_load_students(self):
        student = add_student("Alice")
        self.assertEqual(student, "Alice")
        self.assertEqual(load_students(), ["Alice"])
        self.assertTrue((DATA_DIR / "Alice").exists())

    def test_mark_attendance_and_export(self):
        add_student("Bob")
        self.assertTrue(mark_attendance("Bob"))
        self.assertFalse(mark_attendance("Bob"))

        attendance_files = list(ATTENDANCE_DIR.glob("attendance_*.csv"))
        self.assertEqual(len(attendance_files), 1)

        export_path = export_attendance_csv(output_path="exports/attendance.csv")
        self.assertTrue(export_path.exists())
        self.assertTrue(Path("exports/attendance.csv").exists())


if __name__ == "__main__":
    unittest.main()
