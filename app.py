import tkinter as tk
from tkinter import messagebox, simpledialog
from pathlib import Path

from attendance_utils import add_student, collect_faces, export_attendance_csv, load_students, mark_attendance, train_model


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("700x450")

        tk.Label(root, text="Student Attendance System", font=("Arial", 16, "bold")).pack(pady=10)

        frame = tk.Frame(root)
        frame.pack(pady=10)

        tk.Label(frame, text="Student name:").grid(row=0, column=0, padx=5, pady=5)
        self.name_var = tk.StringVar()
        tk.Entry(frame, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(frame, text="Add Student", command=self.add_student).grid(row=0, column=2, padx=5, pady=5)

        tk.Button(root, text="Collect Face Samples", command=self.collect_samples, width=25).pack(pady=5)
        tk.Button(root, text="Train Model", command=self.train, width=25).pack(pady=5)
        tk.Button(root, text="Start Attendance", command=self.start_attendance, width=25).pack(pady=5)
        tk.Button(root, text="Export Attendance", command=self.export_attendance, width=25).pack(pady=5)

        # Student list display removed per UI update

    def add_student(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input", "Please enter a student name")
            return
        try:
            student = add_student(name)
            messagebox.showinfo("Success", f"Student added: {student}")
            self.name_var.set("")
            self.refresh_students()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def collect_samples(self):
        name = self.name_var.get().strip() or self._choose_student()
        if not name:
            return
        try:
            count = collect_faces(name)
            messagebox.showinfo("Done", f"Collected {count} samples for {name}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def train(self):
        try:
            labels = train_model()
            messagebox.showinfo("Training Complete", f"Trained with: {labels}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def start_attendance(self):
        try:
            from attendance_runner import run_attendance_loop
            run_attendance_loop()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def export_attendance(self):
        try:
            path = export_attendance_csv()
            messagebox.showinfo("Exported", f"Attendance exported to {path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _choose_student(self):
        students = load_students()
        if not students:
            return ""
        return simpledialog.askstring("Student", "Select a student", initialvalue=students[0])


if __name__ == "__main__":
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()
