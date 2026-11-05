"""Benchmark FaceDetector trên nhiều ảnh.

Ground-truth CSV (nếu có) cần các cột:
    image,x,y,w,h

Mỗi dòng mô tả một khuôn mặt. Một ảnh không có mặt được ghi một dòng
với image được điền và x,y,w,h để trống. Ảnh không xuất hiện trong CSV
sẽ không được tính vào các chỉ số chất lượng.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from face_detector import FaceDetector, read_image_rgb

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FaceDetector trên một tập nhiều ảnh."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Một hoặc nhiều file ảnh, hoặc thư mục ảnh (quét đệ quy).",
    )
    parser.add_argument(
        "--gt-csv",
        help="CSV ground truth gồm image,x,y,w,h để đánh giá bbox.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Ngưỡng IoU để ghép dự đoán với ground truth (mặc định: 0.5).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="File CSV lưu thời gian benchmark theo từng ảnh.",
    )
    return parser.parse_args()


def collect_images(inputs: Sequence[str]) -> List[Path]:
    paths = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.update(
                p.resolve()
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.add(path.resolve())
        else:
            print(f"[BỎ QUA] Không tìm thấy file ảnh hợp lệ: {item}")
    return sorted(paths)


def load_ground_truth(csv_path: Optional[str]) -> Dict[str, List[Box]]:
    """Đọc bbox chuẩn; dòng có x/y/w/h trống biểu thị ảnh không có mặt."""
    if not csv_path:
        return {}

    ground_truth: Dict[str, List[Box]] = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image", "x", "y", "w", "h"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Ground-truth CSV phải có các cột: image,x,y,w,h"
            )

        for row in reader:
            image_key = Path(row["image"]).name
            values = [row.get(key, "").strip() for key in ("x", "y", "w", "h")]
            if all(value == "" for value in values):
                ground_truth.setdefault(image_key, [])
            elif any(value == "" for value in values):
                raise ValueError(
                    f"BBox không đầy đủ ở dòng {reader.line_num}: {row}"
                )
            else:
                x, y, w, h = (int(round(float(value))) for value in values)
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"BBox phải có w,h > 0 ở dòng {reader.line_num}: {row}"
                    )
                ground_truth[image_key].append((x, y, w, h))
    return dict(ground_truth)


def summarize(times_ms: Sequence[float]) -> Tuple[float, float, float, float]:
    values = np.asarray(times_ms, dtype=np.float64)
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 95)),
        float(1000.0 / np.mean(values)) if np.mean(values) > 0 else 0.0,
    )


def time_function(function, warmup: int, runs: int):
    """Chạy warmup, sau đó đo thời gian; trả về kết quả cuối và danh sách ms."""
    result = None
    for _ in range(warmup):
        result = function()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = function()
        times.append((time.perf_counter() - start) * 1000.0)
    return result, times


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    predictions: Sequence[Box], targets: Sequence[Box], threshold: float
) -> Tuple[int, int, int, List[float]]:
    """Ghép greedy theo IoU cao nhất; mỗi prediction/target chỉ ghép một lần."""
    candidates = []
    for pred_index, pred in enumerate(predictions):
        for target_index, target in enumerate(targets):
            overlap = box_iou(pred, target)
            if overlap >= threshold:
                candidates.append((overlap, pred_index, target_index))

    candidates.sort(reverse=True)
    used_predictions, used_targets = set(), set()
    matched_ious = []
    for overlap, pred_index, target_index in candidates:
        if pred_index not in used_predictions and target_index not in used_targets:
            used_predictions.add(pred_index)
            used_targets.add(target_index)
            matched_ious.append(overlap)

    tp = len(matched_ious)
    fp = len(predictions) - tp
    fn = len(targets) - tp
    return tp, fp, fn, matched_ious


def print_timing(label: str, times: Sequence[float]) -> None:
    mean, median, p95, fps = summarize(times)
    print(
        f"{label:<30} mean={mean:8.2f} ms | median={median:8.2f} ms "
        f"| p95={p95:8.2f} ms | FPS~{fps:7.2f}"
    )


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("--runs phải >= 1 và --warmup phải >= 0.")
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold phải nằm trong khoảng (0, 1].")

    image_paths = collect_images(args.images)
    if not image_paths:
        raise SystemExit("Không tìm thấy ảnh hợp lệ để benchmark.")

    ground_truth = load_ground_truth(args.gt_csv)

    init_start = time.perf_counter()
    detector = FaceDetector()
    init_ms = (time.perf_counter() - init_start) * 1000.0

    print(f"Số ảnh: {len(image_paths)}")
    print(f"Khởi tạo FaceDetector: {init_ms:.2f} ms")
    print(f"Warmup: {args.warmup} | Số lượt đo/ảnh: {args.runs}")
    print(f"Ngưỡng confidence: {detector.min_confidence}")
    print(f"Ngưỡng IoU: {args.iou_threshold}")
    print("\n--- KẾT QUẢ TỐC ĐỘ ---")

    all_read_times, all_color_times = [], []
    all_preprocess_times, all_detect_times = [], []
    all_pipeline_times, all_facenet_crop_times = [], []
    result_rows = []

    total_tp = total_fp = total_fn = 0
    matched_ious = []
    evaluated_images = 0
    images_with_detection = 0
    total_detections = 0

    for index, image_path in enumerate(image_paths, start=1):
        # Đo riêng thao tác đọc file và chuyển BGR -> RGB.
        _, read_times = time_function(
            lambda: cv2.imread(str(image_path)), args.warmup, args.runs
        )
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            print(f"[LỖI] Không đọc được ảnh: {image_path}")
            continue

        _, color_times = time_function(
            lambda: cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
            args.warmup,
            args.runs,
        )

        # Đo tổng preprocessing bằng đúng hàm mà FaceDetector sử dụng.
        frame_rgb, preprocess_times = time_function(
            lambda: read_image_rgb(image_path), args.warmup, args.runs
        )

        # detect() trả về tất cả khuôn mặt đạt min_confidence.
        faces, detect_times = time_function(
            lambda: detector.detect(frame_rgb), args.warmup, args.runs
        )

        # Đo thêm chi phí resize crop sang kích thước đầu vào FaceNet 160x160.
        _, crop_times = time_function(
            lambda: [face.facenet_crop() for face in faces],
            args.warmup,
            args.runs,
        )

        # Pipeline: đọc ảnh + BGR->RGB + detect (không gồm khởi tạo model).
        pipeline_result, pipeline_times = time_function(
            lambda: detector.detect(read_image_rgb(image_path)),
            args.warmup,
            args.runs,
        )

        all_read_times.extend(read_times)
        all_color_times.extend(color_times)
        all_preprocess_times.extend(preprocess_times)
        all_detect_times.extend(detect_times)
        all_pipeline_times.extend(pipeline_times)
        all_facenet_crop_times.extend(crop_times)

        predicted_boxes = [face.bbox for face in faces]
        total_detections += len(predicted_boxes)
        if predicted_boxes:
            images_with_detection += 1

        # Chất lượng chỉ được tính cho ảnh có khai báo trong ground-truth CSV.
        image_key = image_path.name
        if image_key in ground_truth:
            evaluated_images += 1
            tp, fp, fn, ious = match_boxes(
                predicted_boxes, ground_truth[image_key], args.iou_threshold
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            matched_ious.extend(ious)

        read_mean = summarize(read_times)[0]
        color_mean = summarize(color_times)[0]
        preprocess_mean = summarize(preprocess_times)[0]
        detect_mean = summarize(detect_times)[0]
        pipeline_mean = summarize(pipeline_times)[0]
        crop_mean = summarize(crop_times)[0]

        result_rows.append(
            {
                "image": str(image_path),
                "width": frame_rgb.shape[1],
                "height": frame_rgb.shape[0],
                "detected_faces": len(predicted_boxes),
                "read_mean_ms": round(read_mean, 3),
                "bgr_to_rgb_mean_ms": round(color_mean, 3),
                "preprocess_mean_ms": round(preprocess_mean, 3),
                "detect_mean_ms": round(detect_mean, 3),
                "facenet_crop_mean_ms": round(crop_mean, 3),
                "pipeline_mean_ms": round(pipeline_mean, 3),
                "ground_truth_evaluated": image_key in ground_truth,
            }
        )

        print(
            f"[{index:>3}/{len(image_paths)}] {image_path.name}: "
            f"{frame_rgb.shape[1]}x{frame_rgb.shape[0]}, "
            f"phát hiện {len(predicted_boxes)} mặt"
        )

    if not result_rows:
        raise SystemExit("Không có ảnh nào đọc được.")

    print_timing("Đọc file", all_read_times)
    print_timing("BGR -> RGB", all_color_times)
    print_timing("Preprocessing tổng", all_preprocess_times)
    print_timing("FaceDetector.detect()", all_detect_times)
    print_timing("Crop FaceNet 160x160", all_facenet_crop_times)
    print_timing("Pipeline preprocess + detect", all_pipeline_times)

    print("\n--- TỔNG HỢP ---")
    print(f"Ảnh xử lý thành công: {len(result_rows)}/{len(image_paths)}")
    print(f"Ảnh có phát hiện >= 1 mặt: {images_with_detection}")
    print(f"Tổng số khuôn mặt phát hiện: {total_detections}")

    if args.gt_csv:
        if evaluated_images == 0:
            print(
                "Không có ảnh nào khớp với cột image trong ground-truth CSV. "
                "Hãy dùng tên file ảnh trong CSV."
            )
        else:
            precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
            recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
            print("\n--- ĐÁNH GIÁ CHẤT LƯỢNG BBOX ---")
            print(f"Ảnh có ground truth được đánh giá: {evaluated_images}")
            print(f"TP={total_tp} | FP={total_fp} | FN={total_fn}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"Mean IoU (các cặp ghép đúng): {mean_iou:.4f}")
            print(
                "Lưu ý: các chỉ số này đánh giá phát hiện bbox, không đánh giá "
                "nhận diện danh tính hay chống giả mạo (PAD)."
            )
    else:
        print(
            "\nChưa có --gt-csv: chỉ có kết quả tốc độ/số mặt phát hiện, "
            "chưa thể kết luận độ chính xác."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nĐã lưu kết quả từng ảnh: {output_path.resolve()}")


if __name__ == "__main__":
    main()
