from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, TypedDict, Union

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

if TYPE_CHECKING:
    from face_detector import FaceDetector

ImagePath = Union[str, Path]

IMAGE_SIZE = (64, 64)
LBP_BINS = 59
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_MODES = ("ycbcr", "hsv", "both")


class ModelArtifact(TypedDict):
    model: SVC
    label_encoder: LabelEncoder
    feature_mode: str
    image_size: Tuple[int, int]


def _uniform_lbp_lookup() -> np.ndarray:
    """Map uniform 8-bit LBP codes to 58 bins and all others to bin 58."""
    lookup = np.full(256, LBP_BINS - 1, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        transitions = sum(
            ((code >> bit) & 1) != ((code >> ((bit + 1) % 8)) & 1)
            for bit in range(8)
        )
        if transitions <= 2:
            lookup[code] = next_bin
            next_bin += 1
    if next_bin != LBP_BINS - 1:
        raise RuntimeError(f"Expected 58 uniform LBP patterns, found {next_bin}")
    return lookup


UNIFORM_LBP_LOOKUP = _uniform_lbp_lookup()


def lbp_histogram(channel: np.ndarray) -> np.ndarray:
    """Return an L1-normalized 59-bin uniform LBP histogram for one channel."""
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError("LBP input must be a non-empty single-channel image")

    padded = cv2.copyMakeBorder(
        channel, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT_101
    )
    center = padded[1:-1, 1:-1]
    codes = np.zeros(channel.shape, dtype=np.uint8)

    # Clockwise neighbors, starting at the top-left; bit is set when neighbor >= center.
    neighbors = (
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)

    uniform_codes = UNIFORM_LBP_LOOKUP[codes]
    histogram = np.bincount(uniform_codes.ravel(), minlength=LBP_BINS).astype(
        np.float32
    )
    histogram /= histogram.sum()
    return histogram


def extract_lbp_features(
    face_rgb: np.ndarray, feature_mode: str = "both"
) -> np.ndarray:
    """Extract 177 YCbCr, 177 HSV, or 354 combined LBP features from an RGB face."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"feature_mode must be one of {', '.join(FEATURE_MODES)}"
        )
    if (
        face_rgb.ndim != 3
        or face_rgb.shape[2] != 3
        or face_rgb.shape[0] == 0
        or face_rgb.shape[1] == 0
        or face_rgb.dtype != np.uint8
    ):
        raise ValueError("Face must be a non-empty uint8 RGB image with 3 channels")

    face = cv2.resize(face_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    feature_parts: List[np.ndarray] = []

    if feature_mode in ("ycbcr", "both"):
        y_cr_cb = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)
        y_cb_cr = y_cr_cb[:, :, (0, 2, 1)]
        feature_parts.extend(lbp_histogram(y_cb_cr[:, :, index]) for index in range(3))

    if feature_mode in ("hsv", "both"):
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        feature_parts.extend(lbp_histogram(hsv[:, :, index]) for index in range(3))

    return np.concatenate(feature_parts)


def load_dataset(
    dataset_dir: ImagePath, feature_mode: str, face_detector: FaceDetector
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect faces in class subfolders and build feature and label arrays."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    features: List[np.ndarray] = []
    labels: List[str] = []
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in dataset: {root}")

    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        loaded_count = 0
        for image_path in image_paths:
            try:
                face_rgb = face_detector.extract_face(image_path)
                features.append(extract_lbp_features(face_rgb, feature_mode))
                labels.append(class_dir.name)
                loaded_count += 1
            except (OSError, ValueError) as error:
                logging.warning("Skipping image %s: %s", image_path, error)
        print(f"{class_dir.name}: loaded {loaded_count} face(s)")

    if not features:
        raise ValueError(f"No faces could be extracted from dataset: {root}")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train_svm(
    features: np.ndarray, labels: np.ndarray, test_size: float = 0.25
) -> Tuple[SVC, LabelEncoder]:
    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match")
    if len(np.unique(labels)) < 2:
        raise ValueError("SVM training requires at least two different classes")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    class_counts = np.bincount(encoded_labels)
    test_count = int(np.ceil(len(labels) * test_size))
    train_count = len(labels) - test_count
    if np.min(class_counts) < 2 or test_count < len(encoder.classes_):
        raise ValueError(
            "A stratified train/test split requires at least two images per class "
            "and enough test samples to include every class"
        )
    if train_count < len(encoder.classes_):
        raise ValueError("The training split must contain at least one sample per class")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=test_size,
        random_state=17,
        stratify=encoded_labels,
    )
    model = SVC(kernel="linear", probability=True, C=1.0, random_state=17)
    model.fit(x_train, y_train)

    print(f"Train accuracy: {accuracy_score(y_train, model.predict(x_train)):.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, model.predict(x_test)):.4f}")
    return model, encoder


def save_model(
    model: SVC, encoder: LabelEncoder, feature_mode: str, output_path: ImagePath
) -> None:
    artifact: ModelArtifact = {
        "model": model,
        "label_encoder": encoder,
        "feature_mode": feature_mode,
        "image_size": IMAGE_SIZE,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as model_file:
        pickle.dump(artifact, model_file, protocol=pickle.HIGHEST_PROTOCOL)


def predict_face(face_rgb: np.ndarray, model_path: ImagePath) -> Tuple[str, float]:
    """Predict a detected RGB face using a model artifact saved by this module."""
    with Path(model_path).open("rb") as model_file:
        artifact = pickle.load(model_file)
    feature_mode = artifact["feature_mode"]
    features = extract_lbp_features(face_rgb, feature_mode).reshape(1, -1)
    model = artifact["model"]
    encoder = artifact["label_encoder"]
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[predicted_class])
    label = str(encoder.inverse_transform([predicted_class])[0])
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a face-recognition SVM using uniform LBP color histograms."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Dataset with one class subdirectory per person (default: ./dataset)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="both",
        help="Color features to use; both produces 354 dimensions (default: both)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data reserved for testing (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "svm_lbp_model.pkl",
        help="Path to save the trained model (default: ./svm_lbp_model.pkl)",
    )
    args = parser.parse_args()

    from face_detector import FaceDetector

    face_detector = FaceDetector()
    features, labels = load_dataset(args.dataset, args.feature_mode, face_detector)
    model, encoder = train_svm(features, labels, args.test_size)
    save_model(model, encoder, args.feature_mode, args.output)
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
