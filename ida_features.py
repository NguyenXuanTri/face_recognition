"""Tầng 2: Image Distortion Analysis (IDA) + moiré nhẹ.

IDA theo hướng của Wen et al. (2015), 121 chiều, gồm 4 nhóm:
    specular reflection   3   (tỉ lệ pixel phản xạ, trung bình, độ lệch chuẩn của chúng)
    blurriness            2   (Crete blur, độ giảm gradient khi làm nhòe lại)
    chromatic moment     15   (mean/std/skew của H,S,V + tỉ lệ bin nhỏ nhất/lớn nhất)
    color diversity     101   (số màu khác nhau + phân bố 100 màu phổ biến nhất)

Đây là bản cài đặt lại theo mô tả, không phải mã gốc của tác giả: phát hiện specular
dùng ngưỡng V cao / S thấp (xấp xỉ), và số bin, số mức lượng tử hoá có thể khác bài báo.
Mọi ngưỡng nằm ở hằng số đầu file để chỉnh theo camera thật.

Moiré (4 chiều) là tín hiệu BỔ SUNG, chạy FFT trên crop nhỏ và tìm các đỉnh tần số cao
nổi bật so với nền phổ. Không dùng riêng lẻ để kết luận giả mạo.
"""
from __future__ import annotations

import cv2
import numpy as np

IDA_SIZE = (128, 128)  # IDA tính trên context_crop, lớn hơn 64x64 của LBP
IDA_DIM = 121
MOIRE_DIM = 4

# --- specular ---
SPEC_V_MIN = 210   # pixel đủ sáng (0..255)
SPEC_S_MAX = 50    # và ít bão hoà (0..255)

# --- chromatic moment ---
CHROMA_BINS = 32

# --- color diversity ---
COLOR_BITS = 4                      # 16 mức mỗi kênh -> 4096 màu
COLOR_BINS = 1 << (3 * COLOR_BITS)
TOP_COLORS = 100

# --- moiré ---
MOIRE_SIZE = 128
MOIRE_BAND = (0.12, 1.0)   # bán kính chuẩn hoá theo Nyquist; bỏ vùng tần số rất thấp
MOIRE_HF_START = 0.5       # từ đây trở lên coi là tần số cao
MOIRE_PEAK_THR = 1.0       # ngưỡng phần dư log-phổ để tính là "đỉnh"


def _check_rgb(face_rgb: np.ndarray) -> None:
    if (
        face_rgb.ndim != 3 or face_rgb.shape[2] != 3
        or face_rgb.shape[0] == 0 or face_rgb.shape[1] == 0
        or face_rgb.dtype != np.uint8
    ):
        raise ValueError("Face must be a non-empty uint8 RGB image with 3 channels")


# ----------------------------------------------------------------------------
# Specular reflection (3)
# ----------------------------------------------------------------------------
def specular_features(hsv: np.ndarray) -> np.ndarray:
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = (v >= SPEC_V_MIN) & (s <= SPEC_S_MAX)
    ratio = float(mask.mean())
    if not mask.any():
        return np.array([ratio, 0.0, 0.0], dtype=np.float32)
    vals = v[mask].astype(np.float32) / 255.0
    return np.array([ratio, vals.mean(), vals.std()], dtype=np.float32)


# ----------------------------------------------------------------------------
# Blurriness (2)
# ----------------------------------------------------------------------------
def _crete_direction(gray: np.ndarray, axis: int) -> float:
    kernel = np.ones(9, dtype=np.float32) / 9.0
    kernel = kernel.reshape(9, 1) if axis == 0 else kernel.reshape(1, 9)
    blurred = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    d_f = np.abs(np.diff(gray, axis=axis))
    d_b = np.abs(np.diff(blurred, axis=axis))
    variation = np.maximum(0.0, d_f - d_b)
    total = float(d_f.sum())
    if total < 1e-8:
        return 1.0  # ảnh phẳng: coi là rất nhòe
    return float((total - variation.sum()) / total)


def blur_features(gray01: np.ndarray) -> np.ndarray:
    """[Crete blur (cao = nhòe), độ giảm gradient sau khi nhòe lại (cao = sắc)]."""
    crete = max(_crete_direction(gray01, 0), _crete_direction(gray01, 1))

    def mean_grad(img: np.ndarray) -> float:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.hypot(gx, gy)))

    g0 = mean_grad(gray01)
    g1 = mean_grad(cv2.GaussianBlur(gray01, (0, 0), sigmaX=1.5))
    drop = 0.0 if g0 < 1e-8 else float(np.clip(1.0 - g1 / g0, 0.0, 1.0))
    return np.array([crete, drop], dtype=np.float32)


# ----------------------------------------------------------------------------
# Chromatic moment (15)
# ----------------------------------------------------------------------------
def chromatic_moment_features(hsv: np.ndarray) -> np.ndarray:
    scale = np.array([180.0, 255.0, 255.0], dtype=np.float32)  # thang OpenCV 8-bit
    chans = hsv.astype(np.float32) / scale
    moments, extremes = [], []
    for i in range(3):
        x = chans[:, :, i].ravel()
        mu = float(x.mean())
        centered = x - mu
        std = float(np.sqrt(np.mean(centered ** 2)))
        skew = float(np.cbrt(np.mean(centered ** 3)))
        moments.extend([mu, std, skew])
        hist, _ = np.histogram(x, bins=CHROMA_BINS, range=(0.0, 1.0))
        hist = hist / hist.sum()
        extremes.extend([hist[0], hist[-1]])
    return np.array(moments + extremes, dtype=np.float32)


# ----------------------------------------------------------------------------
# Color diversity (101)
# ----------------------------------------------------------------------------
def color_diversity_features(face_rgb: np.ndarray) -> np.ndarray:
    q = (face_rgb >> (8 - COLOR_BITS)).astype(np.int32)
    idx = (q[:, :, 0] << (2 * COLOR_BITS)) | (q[:, :, 1] << COLOR_BITS) | q[:, :, 2]
    counts = np.bincount(idx.ravel(), minlength=COLOR_BINS).astype(np.float32)
    total = counts.sum()
    distinct = float(np.count_nonzero(counts)) / COLOR_BINS
    top = np.sort(counts)[::-1][:TOP_COLORS] / total
    return np.concatenate([[distinct], top]).astype(np.float32)


# ----------------------------------------------------------------------------
# IDA đầy đủ
# ----------------------------------------------------------------------------
def ida_features(face_rgb: np.ndarray) -> np.ndarray:
    """121 chiều IDA từ một crop RGB (nên là context_crop). Resize về IDA_SIZE."""
    _check_rgb(face_rgb)
    face = cv2.resize(face_rgb, IDA_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    gray01 = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = np.concatenate(
        [
            specular_features(hsv),
            blur_features(gray01),
            chromatic_moment_features(hsv),
            color_diversity_features(face),
        ]
    ).astype(np.float32)
    if out.shape[0] != IDA_DIM:
        raise RuntimeError(f"Expected {IDA_DIM} IDA features, got {out.shape[0]}")
    return out


# ----------------------------------------------------------------------------
# Moiré nhẹ (4)
# ----------------------------------------------------------------------------
def moire_features(face_rgb: np.ndarray) -> np.ndarray:
    """[đỉnh phổ cao nhất, TB 5 đỉnh cao nhất, tỉ lệ pixel đỉnh, tỉ lệ năng lượng tần số cao].

    Các đỉnh được đo là phần dư của log-phổ so với nền phổ đã làm mượt, trong dải tần
    từ MOIRE_BAND[0] đến Nyquist. Một cửa sổ Hann giảm rò rỉ phổ do biên crop.
    """
    _check_rgb(face_rgb)
    n = MOIRE_SIZE
    gray = cv2.cvtColor(
        cv2.resize(face_rgb, (n, n), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    gray -= gray.mean()
    window = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(spectrum)
    log_mag = np.log1p(magnitude).astype(np.float32)

    center = n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.hypot(yy - center, xx - center) / (n / 2.0)
    band = (radius >= MOIRE_BAND[0]) & (radius <= MOIRE_BAND[1])

    residual = log_mag - cv2.GaussianBlur(log_mag, (0, 0), sigmaX=4)
    vals = residual[band]
    top5 = np.sort(vals)[-5:]
    peak_frac = float(np.mean(vals > MOIRE_PEAK_THR))

    energy = magnitude.astype(np.float64) ** 2
    total_energy = energy[radius >= 0.05].sum()
    hf_ratio = (
        float(energy[radius >= MOIRE_HF_START].sum() / total_energy)
        if total_energy > 0 else 0.0
    )
    return np.array(
        [vals.max(), top5.mean(), peak_frac, hf_ratio], dtype=np.float32
    )