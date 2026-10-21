import argparse
import logging
import os
from os.path import join
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Union[str, Path]: cho phép sử dụng một trong hai kiểu
# str: dạng chuỗi, vd: "D:/face_test/dataset/ronaldo/1.jpg"
# path: đối tượng đường dẫn của thư viện pathlib, vd: Path("D:/face_test/dataset/ronaldo/1.jpg")
ImagePath = Union[str, Path]

# Phát hiện khuôn mặt, cắt ra và resize về 160x160 pixel, trả về ảnh RGB (numpy array)
class FaceDetector:
    # Khởi tạo bộ phát hiện khuôn mặt MTCNN
    def __init__(self):
        self.detector = MTCNN()

    # Phát hiện khuôn mặt trong ảnh, cắt ra và resize về 160x160 pixel
    # Optional[ImagePath] nghĩa là output_path có thể nhận một đường dẫn hoặc None
    def extract_face(
        self, 
        filename: ImagePath, 
        output_path: Optional[ImagePath] = None
    ) -> np.ndarray:
        image = cv2.imread(str(filename))   # Đọc ảnh từ file, 
                                            # str(filename) để đảm bảo filename là chuỗi
                                            # Nếu ảnh không đọc được, cv2.imread() trả về None.
                                            # raise ValueError chủ động báo lỗi và dừng xử lý ảnh
        if image is None:
            raise ValueError(f"Could not read image: {filename}") 

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB (MTCNN)
        faces = self.detector.detect_faces(image_rgb)       # Phát hiện khuôn mặt trong ảnh RGB
                                                            # trả về danh sách các khuôn mặt được phát hiện
        if not faces:
            raise ValueError(f"No face detected in: {filename}")

        # Lấy tọa độ, kích thước khuôn mặt đầu tiên được phát hiện
        # x, y: tọa độ góc trên bên trái của bounding box
        # width, height: chiều rộng và chiều cao của bounding box
        x, y, width, height = faces[0]["box"]           

        # lấy kích thước ảnh gốc để đảm bảo bounding box không vượt ra ngoài ảnh
        # image_rgb.shape[:2] trả về (chiều cao, chiều rộng, số kênh màu)    
        image_height, image_width = image_rgb.shape[:2] 

        # Giới hạn bounding box trong ảnh gốc
        # x1, y1: tọa độ góc trên bên trái của bounding box, không âm
        # x2, y2: tọa độ góc dưới bên phải của bounding box, không vượt quá kích thước ảnh
        x1, y1 = abs(x), abs(y)  
        x2, y2 = min(image_width, x + width), min(image_height, y + height)

        # Nếu bounding box không hợp lệ (x1 >= x2 hoặc y1 >= y2), báo lỗi
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Detected face has an invalid bounding box: {filename}")

        # Cắt khuôn mặt từ ảnh gốc và resize về 160x160 pixel
        face = cv2.resize(image_rgb[y1:y2, x1:x2], (160, 160))

        return face

'''
class FACELOADING(FaceDetector):
    # Khởi tạo đối tượng FACELOADING với đường dẫn thư mục chứa dataset
    def __init__(self, directory: ImagePath):
        super().__init__()
        self.directory = directory
        self.X = []
        self.Y = []

    # Tải tất cả các khuôn mặt từ thư mục dataset, trả về mảng X (ảnh) và Y (nhãn)
    def load_faces(self, directory: ImagePath):
        faces = []
        for image_name in os.listdir(directory):
            image_path = join(directory, image_name)
            try:
                faces.append(self.extract_face(image_path))
            except (OSError, ValueError) as error:
                logging.warning("Skipping image %s: %s", image_path, error)
        return faces

    # Tải tất cả các lớp (subdirectory) từ thư mục dataset, trả về mảng X (ảnh) và Y (nhãn)
    def load_classes(self):
        for subdirectory in os.listdir(self.directory):
            directory = join(self.directory, subdirectory)
            faces = self.load_faces(directory)
            labels = [subdirectory for _ in range(len(faces))]

            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(faces)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)
'''


