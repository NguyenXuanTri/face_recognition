import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from face_detector import FaceDetector


class FaceDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Detector")
        self.root.resizable(False, False)

        self.face_detector = FaceDetector()
        self.image_preview = tk.Label(root, text="Chọn ảnh để bắt đầu")
        self.image_preview.pack(padx=20, pady=20)

        select_button = tk.Button(
            root, text="Chọn ảnh", command=self.select_image
        )
        select_button.pack(pady=(0, 20))

    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Tệp ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not image_path:
            return

        try:
            face = self.face_detector.extract_face(image_path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Không xử lý được ảnh", str(error))
            return

        preview = ImageTk.PhotoImage(Image.fromarray(face))
        self.image_preview.configure(image=preview, text="")
        self.image_preview.image = preview


def main():
    root = tk.Tk()
    FaceDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
