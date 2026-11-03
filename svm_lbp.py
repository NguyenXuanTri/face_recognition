"""Tầng 1 chống giả mạo: Color LBP (YCbCr + HSV) -> SVM nhị phân real/spoof.

Cấu trúc dataset (cùng tên subject ở real/ và spoof/ sẽ được gộp vào cùng một nhóm
khi chia train/val/test, tránh rò rỉ):

    dataset/
      real/<subject_or_video>/*.jpg
      spoof/<subject_or_video>/*.jpg

Nhãn: real = 1, spoof = 0. Điểm (score) càng cao càng giống mặt thật.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

from face_detector import is_frontal

if TYPE_CHECKING:
    from face_detector import DetectedFace, FaceDetector

IMAGE_SIZE = (64, 64)
LBP_BINS = 59
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_MODES = ("ycbcr", "hsv", "both")

# Ngưỡng quality gate: giá trị khởi đầu, cần chỉnh theo camera thực tế.
GATE_DEFAULTS = {
    "min_face_px": 80,
    "min_blur_var": 40.0,
    "brightness_range": (40.0, 220.0),
    "max_yaw": 0.35,
    "max_roll_deg": 25.0,
}


# ----------------------------------------------------------------------------
# LBP / đặc trưng (giữ nguyên logic cũ)
# ----------------------------------------------------------------------------
def _uniform_lbp_lookup() -> np.ndarray:
    lookup = np.full(256, LBP_BINS - 1, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        transitions = sum(
            ((code >> b) & 1) != ((code >> ((b + 1) % 8)) & 1) for b in range(8)
        )
        if transitions <= 2:
            lookup[code] = next_bin
            next_bin += 1
    if next_bin != LBP_BINS - 1:
        raise RuntimeError(f"Expected 58 uniform patterns, found {next_bin}")
    return lookup


UNIFORM_LBP_LOOKUP = _uniform_lbp_lookup()


def lbp_histogram(channel: np.ndarray) -> np.ndarray:
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError("LBP input must be a non-empty single-channel image")
    padded = cv2.copyMakeBorder(channel, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    center = padded[1:-1, 1:-1]
    codes = np.zeros(channel.shape, dtype=np.uint8)
    neighbors = (
        padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
        padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2],
    )
    for bit, nb in enumerate(neighbors):
        codes |= (nb >= center).astype(np.uint8) << bit
    hist = np.bincount(
        UNIFORM_LBP_LOOKUP[codes].ravel(), minlength=LBP_BINS
    ).astype(np.float32)
    hist /= hist.sum()
    return hist


def extract_lbp_features(face_rgb: np.ndarray, feature_mode: str = "both") -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if (
        face_rgb.ndim != 3 or face_rgb.shape[2] != 3
        or face_rgb.shape[0] == 0 or face_rgb.shape[1] == 0
        or face_rgb.dtype != np.uint8
    ):
        raise ValueError("Face must be a non-empty uint8 RGB image with 3 channels")
    face = cv2.resize(face_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    parts: List[np.ndarray] = []
    if feature_mode in ("ycbcr", "both"):
        ycc = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)[:, :, (0, 2, 1)]  # Y, Cb, Cr
        parts.extend(lbp_histogram(ycc[:, :, i]) for i in range(3))
    if feature_mode in ("hsv", "both"):
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        parts.extend(lbp_histogram(hsv[:, :, i]) for i in range(3))
    return np.concatenate(parts)


# ----------------------------------------------------------------------------
# Quality gate (Tầng 4): chạy TRƯỚC khi chấm điểm PAD
# ----------------------------------------------------------------------------
def quality_ok(
    face_rgb: np.ndarray,
    gate: Dict | None = None,
    face_size: int | None = None,
    landmarks: Dict | None = None,
) -> bool:
    """face_size: cạnh bbox GỐC (từ DetectedFace.face_size); nếu None thì dùng kích thước crop.
    landmarks: nếu có thì kiểm tra thêm mặt nhìn thẳng."""
    g = {**GATE_DEFAULTS, **(gate or {})}
    size = face_size if face_size is not None else min(face_rgb.shape[:2])
    if size < g["min_face_px"]:
        return False
    if landmarks is not None and not is_frontal(
        landmarks, g["max_yaw"], g["max_roll_deg"]
    ):
        return False
    # Chuẩn hoá về cùng độ phân giải để điểm nét so sánh được giữa các khung.
    gray = cv2.cvtColor(
        cv2.resize(face_rgb, (128, 128), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2GRAY,
    )
    if cv2.Laplacian(gray, cv2.CV_64F).var() < g["min_blur_var"]:
        return False
    lo, hi = g["brightness_range"]
    return lo <= float(gray.mean()) <= hi


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def load_dataset(
    dataset_dir: Path, feature_mode: str, face_detector: "FaceDetector",
    use_gate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_dir)
    features, labels, groups = [], [], []
    for class_name, label in (("real", 1), ("spoof", 0)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise NotADirectoryError(f"Missing class folder: {class_dir}")
        count = 0
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Nhóm = tên thư mục subject/video; ảnh nằm trực tiếp thì mỗi ảnh một nhóm.
            group = path.parent.name if path.parent != class_dir else path.stem
            try:
                det = face_detector.detect_file(path)
                if det is None:
                    logging.warning("Skipping %s: no face detected", path)
                    continue
                if use_gate and not quality_ok(
                    det.crop, face_size=det.face_size, landmarks=det.landmarks
                ):
                    continue
                features.append(extract_lbp_features(det.crop, feature_mode))
            except Exception as error:  # đọc ảnh/detector có thể ném nhiều loại lỗi
                logging.warning("Skipping %s: %s", path, error)
                continue
            labels.append(label)
            groups.append(group)
            count += 1
        print(f"{class_name}: loaded {count} face(s)")
    if not features:
        raise ValueError("No faces could be extracted")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def split_by_group(
    y: np.ndarray, groups: np.ndarray, test_size: float, val_size: float, seed: int = 17
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(idx, y, groups))
    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed
    )
    tr, va = next(inner.split(trainval, y[trainval], groups[trainval]))
    train, val = trainval[tr], trainval[va]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if len(np.unique(y[part])) < 2:
            raise ValueError(
                f"Split '{name}' thiếu một lớp; cần thêm subject hoặc đổi tỉ lệ chia"
            )
    return train, val, test


# ----------------------------------------------------------------------------
# Chỉ số PAD (ISO/IEC 30107-3) và chọn ngưỡng
# ----------------------------------------------------------------------------
def pad_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> Dict[str, float]:
    pred_real = score >= thr
    apcer = float(np.mean(pred_real[y == 0]))        # spoof bị nhận là thật
    bpcer = float(np.mean(~pred_real[y == 1]))       # thật bị nhận là giả
    return {"APCER": apcer, "BPCER": bpcer, "HTER": (apcer + bpcer) / 2}


def eer_threshold(y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, score, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(thr[i]), float((fpr[i] + fnr[i]) / 2)


# ----------------------------------------------------------------------------
# Train / lưu
# ----------------------------------------------------------------------------
def train_pad(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    test_size: float = 0.25, val_size: float = 0.15, c: float = 1.0,
) -> Tuple[SVC, float, Dict[str, Dict[str, float]]]:
    train, val, test = split_by_group(y, groups, test_size, val_size)
    model = SVC(kernel="linear", C=c, class_weight="balanced")
    model.fit(x[train], y[train])

    thr, val_eer = eer_threshold(y[val], model.decision_function(x[val]))
    report = {
        "val": {**pad_metrics(y[val], model.decision_function(x[val]), thr), "EER": val_eer},
        "test": pad_metrics(y[test], model.decision_function(x[test]), thr),
    }
    for name, m in report.items():
        print(name, {k: round(v, 4) for k, v in m.items()})
    return model, thr, report


def save_pad(model: SVC, thr: float, feature_mode: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model, "threshold": thr, "feature_mode": feature_mode,
            "image_size": IMAGE_SIZE, "gate": GATE_DEFAULTS,
        },
        path,
    )


def load_pad(path: Path) -> dict:
    """Chỉ load file do chính bạn tạo (joblib/pickle không an toàn với file lạ)."""
    return joblib.load(path)


# ----------------------------------------------------------------------------
# Suy luận: gate -> điểm từng khung -> trung vị -> quyết định
# ----------------------------------------------------------------------------
def score_frame(face_rgb: np.ndarray, artifact: dict) -> float:
    """Tầng 1 thuần: một crop mặt RGB -> một điểm (cao = giống mặt thật)."""
    feats = extract_lbp_features(face_rgb, artifact["feature_mode"]).reshape(1, -1)
    return float(artifact["model"].decision_function(feats)[0])


def pad_decision(
    detections: Sequence["DetectedFace"], artifact: dict, min_frames: int = 3
) -> Tuple[str, float, int]:
    """Trả về (label, score, số_khung_dùng). label in {'real','spoof','retry'}.

    detections: các DetectedFace của cùng một người qua nhiều khung.
    'retry' = không đủ khung đạt chất lượng -> yêu cầu camera thu lại,
    KHÔNG gán nhãn giả mạo.
    """
    good = [
        d for d in detections
        if quality_ok(d.crop, artifact.get("gate"), d.face_size, d.landmarks)
    ]
    if len(good) < min_frames:
        return "retry", float("nan"), len(good)
    score = float(np.median([score_frame(d.crop, artifact) for d in good]))
    return ("real" if score >= artifact["threshold"] else "spoof"), score, len(good)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Color-LBP + SVM presentation attack detector.")
    p.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset")
    p.add_argument("--feature-mode", choices=FEATURE_MODES, default="both")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--quality-gate", action="store_true", help="Lọc khung kém chất lượng khi train")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "pad_lbp_model.joblib")
    args = p.parse_args()

    from face_detector import FaceDetector

    x, y, groups = load_dataset(args.dataset, args.feature_mode, FaceDetector(), args.quality_gate)
    model, thr, _ = train_pad(x, y, groups, args.test_size, args.val_size, args.c)
    save_pad(model, thr, args.feature_mode, args.output)
    print(f"Saved: {args.output} (threshold={thr:.4f})")


if __name__ == "__main__":
    main()