"""Đo thời gian từng khối của pipeline PAD + nhận diện (chạy trên Raspberry Pi 4).

Đặt file này cạnh face_detector.py, svm_lbp.py, ida_features.py rồi chạy trong thư mục repo.

Ví dụ:
    # Chụp 20 khung bằng Picamera2 (ngồi trước camera, nhìn thẳng) rồi đo
    python benchmark_blocks.py --capture 20 --pad-model pad_casia_lbp.joblib
    # Đo lại trên đúng các khung đã chụp
    python benchmark_blocks.py --frames bench_frames --pad-model pad_casia_lbp.joblib
    # Không có khung đầy đủ: chỉ đo các khối sau MTCNN trên crop mặt sẵn có
    python benchmark_blocks.py --crops dataset/CASIA-FASD/Test/real/s1

Kết quả: bảng in ra màn hình, <out>.csv và <out>_env.json.
Thời gian tính bằng ms cho MỘT lần gọi (một khung / một mặt), sau khi đã warm-up.
Không tính thời gian đọc file và chụp ảnh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import ida_features as ida
from face_detector import FaceDetector, read_image_rgb
from svm_lbp import extract_lbp_features, load_pad, quality_ok

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Nới hết ngưỡng để quality_ok luôn chạy đủ mọi phép kiểm tra (không thoát sớm).
GATE_OPEN = {
    "min_face_px": 0, "min_blur_var": -1.0, "brightness_range": (-1.0, 999.0),
    "max_yaw": 99.0, "max_roll_deg": 999.0,
}


def read_first_line(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "?"


def pi_state() -> dict:
    temp = read_first_line("/sys/class/thermal/thermal_zone0/temp")
    freq = read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    try:
        throttled = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        throttled = "?"
    return {
        "temp_c": int(temp) / 1000 if temp.isdigit() else None,
        "freq_mhz": int(freq) / 1000 if freq.isdigit() else None,
        "governor": read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "throttled": throttled,
    }


def list_images(folder: Path) -> list:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"Không có ảnh trong {folder}")
    return paths


def capture_frames(n: int, folder: Path, size=(640, 480)) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError as error:
        raise SystemExit(
            f"Không import được picamera2 ({error}). Dùng venv có --system-site-packages, "
            "hoặc chụp bằng rpicam-still rồi chạy với --frames."
        )
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise SystemExit(f"{folder} không rỗng; đổi --frames-out hoặc dùng --frames {folder}")
    cam = Picamera2()
    # Picamera2: định dạng "BGR888" trả về mảng theo thứ tự R, G, B.
    cam.configure(cam.create_video_configuration(main={"size": size, "format": "BGR888"}))
    cam.start()
    print("Nhìn thẳng vào camera, bắt đầu chụp sau 3 giây...")
    time.sleep(3)
    for i in range(n):
        rgb = cam.capture_array("main")
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        time.sleep(0.2)
    cam.stop()
    print(f"Đã lưu {n} khung vào {folder}")


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:  # không phóng to
        return img
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


class Bench:
    def __init__(self, runs: int, warmup: int) -> None:
        self.runs, self.warmup, self.rows = runs, warmup, []

    def run(self, name: str, fn, inputs: list, note: str = "") -> float:
        """Gọi fn lần lượt trên các input (xoay vòng). Trả về trung vị (ms)."""
        for i in range(self.warmup):
            fn(inputs[i % len(inputs)])
        times = []
        for i in range(self.runs):
            x = inputs[i % len(inputs)]
            t0 = time.perf_counter_ns()
            fn(x)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        t = np.array(times)
        row = {
            "block": name, "median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
            "p95_ms": float(np.percentile(t, 95)), "min_ms": float(t.min()),
            "max_ms": float(t.max()), "runs": len(t), "note": note,
        }
        self.rows.append(row)
        print(
            f"{name:<28} median={row['median_ms']:9.2f}  p95={row['p95_ms']:9.2f}  "
            f"max={row['max_ms']:9.2f}  {note}", flush=True
        )
        return row["median_ms"]


def main() -> None:
    p = argparse.ArgumentParser(description="Đo thời gian từng khối trên Raspberry Pi.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", type=int, metavar="N", help="chụp N khung bằng Picamera2")
    src.add_argument("--frames", type=Path, help="thư mục khung hình ĐẦY ĐỦ (đo cả MTCNN)")
    src.add_argument("--crops", type=Path, help="thư mục crop mặt (bỏ qua MTCNN)")
    p.add_argument("--frames-out", type=Path, default=Path("bench_frames"))
    p.add_argument("--widths", type=int, nargs="+", default=[640, 320])
    p.add_argument("--min-face", type=int, nargs="+", default=[20, 80], help="min_face_size của MTCNN")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", type=int, default=None, help="số luồng torch + OpenCV")
    p.add_argument("--pad-model", type=Path, help="file .joblib do chính bạn tạo")
    p.add_argument("--no-facenet", action="store_true")
    p.add_argument("--out", default="pi_timing")
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0:
        p.error("--runs >= 1 và --warmup >= 0")

    try:
        import torch
    except ImportError:
        torch = None
        if not (args.crops and args.no_facenet):
            raise SystemExit("Thiếu torch (chỉ chạy được với --crops ... --no-facenet)")
    if args.threads:
        cv2.setNumThreads(args.threads)
        if torch:
            torch.set_num_threads(args.threads)

    env = {
        "machine": platform.machine(), "python": sys.version.split()[0],
        "numpy": np.__version__, "opencv": cv2.__version__,
        "torch": torch.__version__ if torch else None,
        "torch_threads": torch.get_num_threads() if torch else None,
        "opencv_threads": cv2.getNumThreads(), "cpu_count": os.cpu_count(),
        "model": read_first_line("/proc/device-tree/model").rstrip("\x00"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "before": pi_state(),
    }
    print(json.dumps(env, ensure_ascii=False, indent=2))
    bench = Bench(args.runs, args.warmup)
    detect_ms = {}

    # ---- Khối 1: phát hiện + cắt mặt ------------------------------------
    if args.crops:
        faces = [(read_image_rgb(f), None, None) for f in list_images(args.crops)[:50]]
    else:
        if args.capture:
            if args.capture < 1:
                p.error("--capture phải >= 1")
            capture_frames(args.capture, args.frames_out)
            args.frames = args.frames_out
        frames = [read_image_rgb(f) for f in list_images(args.frames)[:50]]

        from facenet_pytorch import MTCNN

        faces = []
        for width in args.widths:
            resized = [resize_to_width(f, width) for f in frames]
            size = f"{resized[0].shape[1]}x{resized[0].shape[0]}"
            for min_face in args.min_face:
                detector = FaceDetector(MTCNN(keep_all=True, min_face_size=min_face))
                found = [detector.detect_largest(f) for f in resized]
                hits = [d for d in found if d is not None]
                label = f"mtcnn_{size}_min{min_face}"
                detect_ms[label] = bench.run(
                    label, detector.detect_largest, resized,
                    note=f"phát hiện {len(hits)}/{len(found)} khung",
                )
                # Crop cho các khối sau lấy từ cấu hình đầu tiên có phát hiện được mặt.
                if not faces and hits:
                    faces = [(d.crop, d.face_size, d.landmarks) for d in hits]
                    sides = [d.face_size for d in hits]
                    env["face_px_median"] = float(np.median(sides))
                    env["crop_source"] = label
        if not faces:
            raise SystemExit("MTCNN không phát hiện được mặt nào; chụp lại khung.")

    crops = [f[0] for f in faces]

    # ---- Khối 2: quality gate -------------------------------------------
    gate_ms = bench.run(
        "quality_gate",
        lambda f: quality_ok(f[0], GATE_OPEN, f[1], f[2]),
        faces,
    )

    # ---- Khối 3: Color LBP ----------------------------------------------
    bench.run("lbp_ycbcr", lambda c: extract_lbp_features(c, "ycbcr"), crops)
    bench.run("lbp_hsv", lambda c: extract_lbp_features(c, "hsv"), crops)
    lbp_ms = bench.run("lbp_both", lambda c: extract_lbp_features(c, "both"), crops)

    # ---- Khối 4: IDA (tổng và từng nhóm) --------------------------------
    ida_ms = bench.run("ida_total", ida.ida_features, crops)
    small = [cv2.resize(c, ida.IDA_SIZE, interpolation=cv2.INTER_AREA) for c in crops]
    hsvs = [cv2.cvtColor(s, cv2.COLOR_RGB2HSV) for s in small]
    grays = [cv2.cvtColor(s, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for s in small]
    bench.run("  ida_specular", ida.specular_features, hsvs)
    bench.run("  ida_blur", ida.blur_features, grays)
    bench.run("  ida_chromatic", ida.chromatic_moment_features, hsvs)
    bench.run("  ida_color_diversity", ida.color_diversity_features, small)

    # ---- Khối 5: moiré ---------------------------------------------------
    moire_ms = bench.run("moire", ida.moire_features, crops)

    # ---- Khối 6: SVM PAD --------------------------------------------------
    svm_ms = None
    if args.pad_model:
        art = load_pad(args.pad_model)
        model = art["model"]
        svm = model.steps[-1][1] if hasattr(model, "steps") else model
        feats = [extract_lbp_features(c, art["feature_mode"]).reshape(1, -1) for c in crops]
        svm_ms = bench.run(
            "pad_svm_predict", model.decision_function, feats,
            note=f"{art['feature_mode']}, {len(svm.support_)} support vector",
        )

    # ---- Khối 7: FaceNet embedding ----------------------------------------
    facenet_ms = None
    if not args.no_facenet:
        from facenet_pytorch import InceptionResnetV1

        net = InceptionResnetV1(pretrained="vggface2").eval()  # lần đầu cần mạng để tải trọng số

        def preprocess(crop):
            face = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0)
            return (t - 127.5) / 128.0

        def forward(t):
            with torch.inference_mode():
                return net(t)

        pre_ms = bench.run("facenet_preprocess", preprocess, crops)
        fwd_ms = bench.run("facenet_forward", forward, [preprocess(c) for c in crops])
        facenet_ms = pre_ms + fwd_ms

    env["after"] = pi_state()

    # ---- Ước lượng độ trễ một lần quyết định -------------------------------
    if detect_ms and facenet_ms is not None:
        missing = "" if svm_ms is not None else " (CHƯA gồm SVM PAD)"
        print(f"\nƯớc lượng một lần quyết định = N khung PAD + 1 lần FaceNet{missing}")
        print("Cộng các trung vị; chưa gồm chụp ảnh, SVM nhận diện và điều khiển khoá.")
        env["estimate_ms"] = {}
        for label, d in detect_ms.items():
            base = d + gate_ms + lbp_ms + (svm_ms or 0.0)
            for n in (3, 5):
                lbp_total = n * base + facenet_ms
                full = lbp_total + n * (ida_ms + moire_ms)
                env["estimate_ms"][f"{label}_N{n}"] = {"lbp": lbp_total, "lbp+ida+moire": full}
                print(f"  {label:<24} N={n}: LBP {lbp_total:8.0f} ms | +IDA+moiré {full:8.0f} ms")
        print("  (cột +IDA+moiré chỉ cộng thời gian trích đặc trưng, SVM vẫn tính như bản LBP)")

    out_csv = Path(f"{args.out}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bench.rows[0].keys()))
        w.writeheader()
        w.writerows(bench.rows)
    Path(f"{args.out}_env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTrước đo: {env['before']}\nSau đo : {env['after']}")
    print(f"Đã ghi {out_csv} và {args.out}_env.json")


if __name__ == "__main__":
    main()