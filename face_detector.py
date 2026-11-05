"""Phát hiện và cắt mặt (MTCNN) cho pipeline chống giả mạo + nhận diện.

Khác bản cũ:
- Nhận khung hình RGB dạng mảng (camera) hoặc đường dẫn file (huấn luyện).
- Chọn mặt lớn nhất trong số mặt vượt ngưỡng confidence.
- Kẹp toạ độ bbox đúng (max/min), không dùng abs().
- Crop vuông, không méo; có crop rộng hơn (context_crop) cho IDA.
- Giữ kích thước mặt gốc + 5 landmark cho cổng chất lượng và motion (Tầng 3, 4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

ImagePath = Union[str, Path]
Box = Tuple[int, int, int, int]  # x, y, w, h trong toạ độ khung hình
Point = Tuple[float, float]

FACENET_SIZE = (160, 160)
LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


@dataclass(frozen=True)
class DetectedFace:
    crop: np.ndarray          # RGB uint8, vuông, sát bbox -> đưa vào Color LBP
    context_crop: np.ndarray  # RGB uint8, vuông, rộng hơn -> đưa vào IDA
    bbox: Box                 # bbox gốc đã kẹp trong khung hình
    confidence: float
    landmarks: Dict[str, Point]  # toạ độ trong khung hình gốc

    @property
    def face_size(self) -> int:
        """Cạnh nhỏ nhất của bbox gốc (pixel), dùng cho cổng chất lượng."""
        return min(self.bbox[2], self.bbox[3])

    def facenet_crop(self) -> np.ndarray:
        """Crop 160x160 RGB cho FaceNet."""
        return cv2.resize(self.crop, FACENET_SIZE, interpolation=cv2.INTER_AREA)


def read_image_rgb(path: ImagePath) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _square_box(box: Box, scale: float, img_w: int, img_h: int) -> Tuple[int, int, int]:
    """Hình vuông cùng tâm bbox, cạnh = max(w, h) * scale, dịch vào trong khung hình."""
    x, y, w, h = box
    side = min(max(w, h) * scale, img_w, img_h)
    x1 = min(max(x + w / 2 - side / 2, 0), img_w - side)
    y1 = min(max(y + h / 2 - side / 2, 0), img_h - side)
    s = max(1, int(round(side)))
    x1 = int(min(max(round(x1), 0), img_w - s))
    y1 = int(min(max(round(y1), 0), img_h - s))
    return x1, y1, s


def _crop(frame: np.ndarray, box: Box, scale: float) -> np.ndarray:
    h_img, w_img = frame.shape[:2]
    x, y, s = _square_box(box, scale, w_img, h_img)
    return frame[y:y + s, x:x + s].copy()


def is_frontal(
    landmarks: Dict[str, Point], max_yaw: float = 0.35, max_roll_deg: float = 25.0
) -> bool:
    """Mặt nhìn tương đối thẳng, ước lượng từ 5 landmark MTCNN.

    yaw  ~ độ lệch ngang của mũi so với trung điểm hai mắt, chia cho khoảng cách mắt.
    roll ~ góc nghiêng của đường nối hai mắt.
    Đây chỉ là ước lượng thô, không phải góc đầu chính xác.
    """
    try:
        (lx, ly), (rx, ry) = landmarks["left_eye"], landmarks["right_eye"]
        nx, _ = landmarks["nose"]
    except KeyError:
        return False
    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < 1e-6:
        return False
    yaw = (nx - (lx + rx) / 2) / eye_dist
    roll = math.degrees(math.atan2(ry - ly, rx - lx))
    return abs(yaw) <= max_yaw and abs(roll) <= max_roll_deg


class FaceDetector:
    def __init__(
        self,
        detector=None,
        min_confidence: float = 0.90,
        context_scale: float = 1.5,
    ) -> None:
        """detector: facenet-pytorch MTCNN hoặc đối tượng có detect_faces(rgb)."""
        if detector is None:
            from facenet_pytorch import MTCNN  # import trễ để module nhẹ khi không cần

            detector = MTCNN(keep_all=True)
        self._detector = detector
        self.min_confidence = min_confidence
        self.context_scale = context_scale

    def detect(self, frame_rgb: np.ndarray) -> List[DetectedFace]:
        """Trả về các mặt đủ tin cậy, sắp xếp theo diện tích giảm dần."""
        if (
            frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3
            or frame_rgb.size == 0 or frame_rgb.dtype != np.uint8
        ):
            raise ValueError("Frame must be a non-empty uint8 RGB image")
        h_img, w_img = frame_rgb.shape[:2]

        faces: List[DetectedFace] = []
        if hasattr(self._detector, "detect_faces"):
            detections = self._detector.detect_faces(frame_rgb)
        else:
            boxes, probabilities, points = self._detector.detect(
                frame_rgb, landmarks=True
            )
            detections = []
            if boxes is not None and probabilities is not None:
                for index, (coords, probability) in enumerate(
                    zip(boxes, probabilities)
                ):
                    if coords is None or probability is None:
                        continue
                    keypoints = {}
                    if points is not None and points[index] is not None:
                        keypoints = dict(zip(LANDMARK_NAMES, points[index]))
                    x1, y1, x2, y2 = coords
                    detections.append({
                        "box": (x1, y1, x2 - x1, y2 - y1),
                        "confidence": probability,
                        "keypoints": keypoints,
                    })

        for item in detections:
            confidence = float(item.get("confidence") or 0.0)
            if confidence < self.min_confidence:
                continue
            x, y, w, h = item["box"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            box: Box = (x1, y1, x2 - x1, y2 - y1)
            landmarks = {
                name: (float(px), float(py))
                for name, (px, py) in item.get("keypoints", {}).items()
            }
            faces.append(
                DetectedFace(
                    crop=_crop(frame_rgb, box, 1.0),
                    context_crop=_crop(frame_rgb, box, self.context_scale),
                    bbox=box,
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces

    def detect_largest(self, frame_rgb: np.ndarray) -> Optional[DetectedFace]:
        faces = self.detect(frame_rgb)
        return faces[0] if faces else None

    def detect_file(self, path: ImagePath) -> Optional[DetectedFace]:
        return self.detect_largest(read_image_rgb(path))

    def extract_face(self, filename: ImagePath) -> np.ndarray:
        """Tương thích bản cũ: crop 160x160 RGB của mặt lớn nhất, lỗi nếu không có mặt."""
        face = self.detect_file(filename)
        if face is None:
            raise ValueError(f"No face detected: {filename}")
        return face.facenet_crop()
