"""Thu dữ liệu PAD bằng Picamera2 trên Raspberry Pi (chạy trong venv env2).

Mỗi lần chạy thu MỘT video (chuỗi khung PNG) của một subject với một loại mẫu:
    dataset/picam/<real|spoof>/<subject>/<video_id>/frame_0001.png, ...
    dataset/picam/<real|spoof>/<subject>/<video_id>/meta.json
    dataset/picam/index.csv   (mỗi video một dòng)

video_id = <subject>_<kind>_<light>_<dist>_t<lần thu>, ví dụ p01_print_indoor_50cm_t01.
Với mẫu giả, <subject> là người có mặt trên ảnh in / màn hình (để chia tập theo subject).

Camera:
- Cố định chế độ sensor (mặc định 1640x1232 = đủ góc nhìn của IMX219), ảnh ra 640x480.
- Để AE/AWB tự chỉnh vài giây rồi KHOÁ phơi sáng, gain và cân bằng trắng trong suốt video.
  Dùng --lock-from <meta.json cũ> để dùng lại đúng thông số của một video trước
  (ví dụ: mẫu thật và mẫu giả cùng điều kiện ánh sáng).
- Ghi metadata từng khung (ExposureTime, AnalogueGain, ColourGains, ScalerCrop...) vào meta.json.

Ví dụ:
    python capture_picam.py --subject p01 --kind real --light indoor --dist 50cm
    python capture_picam.py --subject p01 --kind print --light indoor --dist 50cm \
        --lock-from dataset/picam/real/p01/p01_real_indoor_50cm_t01/meta.json
    python capture_picam.py --list-modes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

KINDS = {  # kind -> lớp
    "real": "real",
    "print": "spoof",        # ảnh in giấy
    "phone_photo": "spoof",  # ảnh tĩnh trên màn hình điện thoại
    "phone_video": "spoof",  # phát lại video trên điện thoại
    "laptop": "spoof",       # ảnh/video trên màn hình laptop
}
TAG_RE = re.compile(r"^[A-Za-z0-9]+$")
LOCK_KEYS = ("ExposureTime", "AnalogueGain", "ColourGains")
FRAME_MD_KEYS = (
    "SensorTimestamp", "ExposureTime", "AnalogueGain", "DigitalGain",
    "ColourGains", "ColourTemperature", "Lux", "FocusFoM", "ScalerCrop",
)
INDEX_FIELDS = (
    "video_id", "label", "subject", "kind", "light", "dist", "take", "n_frames",
    "fps_measured", "dropped", "sensor_mode", "exposure_us", "analogue_gain",
    "colour_gains", "face_rate", "gate_rate", "median_face_px", "created", "note",
)


def tag(value: str, name: str) -> str:
    if not TAG_RE.match(value):
        raise SystemExit(f"--{name} chỉ được chứa chữ và số, không dấu cách/gạch: {value!r}")
    return value


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def pick_md(md: dict, keys=FRAME_MD_KEYS) -> dict:
    return {k: to_jsonable(md[k]) for k in keys if k in md}


def open_camera():
    try:
        from picamera2 import Picamera2
    except ImportError as error:
        raise SystemExit(
            f"Không import được picamera2 ({error}). Chạy trong venv env2 "
            "(tạo bằng --system-site-packages)."
        )
    return Picamera2()


def list_modes() -> None:
    cam = open_camera()
    for mode in cam.sensor_modes:
        print({k: to_jsonable(v) for k, v in mode.items()})
    cam.close()


def next_take(base: Path, prefix: str) -> int:
    takes = [
        int(m.group(1)) for p in base.glob(f"{prefix}_t*")
        if (m := re.fullmatch(re.escape(prefix) + r"_t(\d+)", p.name))
    ]
    return max(takes, default=0) + 1


def capture(args) -> tuple[list, list, dict, dict]:
    """Trả về (khung RGB, metadata từng khung, cấu hình camera, thông số đã khoá)."""
    cam = open_camera()
    frame_us = int(round(1e6 / args.fps))
    config = cam.create_video_configuration(
        main={"size": tuple(args.size), "format": "BGR888"},
        sensor={"output_size": tuple(args.sensor_size), "bit_depth": 10},
        controls={"FrameDurationLimits": (frame_us, frame_us)},
        buffer_count=6,
    )
    cam.configure(config)
    camera_config = {
        "sensor": to_jsonable(cam.camera_configuration().get("sensor")),
        "main": to_jsonable({k: v for k, v in cam.camera_configuration()["main"].items()
                             if k in ("size", "format", "stride")}),
        "frame_duration_us": frame_us,
        "camera_model": to_jsonable(cam.camera_properties.get("Model")),
    }
    cam.start()
    try:
        if args.lock_from:
            locked = json.loads(Path(args.lock_from).read_text(encoding="utf-8"))["locked_controls"]
            print(f"Dùng thông số khoá từ {args.lock_from}: {locked}")
        else:
            print(f"Để AE/AWB tự chỉnh {args.settle:.0f} giây...")
            time.sleep(args.settle)
            md = cam.capture_metadata()
            locked = {k: to_jsonable(md[k]) for k in LOCK_KEYS}
        if args.no_lock:
            locked = {}
            print("KHÔNG khoá phơi sáng/cân bằng trắng (--no-lock).")
        else:
            controls = {"AeEnable": False, "AwbEnable": False, **locked}
            controls["ColourGains"] = tuple(controls["ColourGains"])
            if controls["ExposureTime"] > frame_us:
                print(f"Cảnh báo: ExposureTime {controls['ExposureTime']} us > thời gian khung "
                      f"{frame_us} us, fps thực sẽ thấp hơn {args.fps}.")
            cam.set_controls(controls)
            time.sleep(0.5)  # chờ vài khung để thông số mới có hiệu lực
            print(f"Đã khoá: {locked}")

        for s in range(args.countdown, 0, -1):
            print(f"Bắt đầu thu sau {s}...", flush=True)
            time.sleep(1)
        n = int(round(args.seconds * args.fps))
        print(f"ĐANG THU {n} khung ({args.seconds} giây)...", flush=True)
        frames, frame_md = [], []
        for _ in range(n):
            request = cam.capture_request()
            try:
                frames.append(request.make_array("main").copy())  # thứ tự kênh: R, G, B
                frame_md.append(pick_md(request.get_metadata()))
            finally:
                request.release()
        print("Xong phần thu.")
    finally:
        cam.stop()
        cam.close()
    return frames, frame_md, camera_config, locked


def timing_stats(frame_md: list, fps: float) -> dict:
    ts = np.array([m["SensorTimestamp"] for m in frame_md if "SensorTimestamp" in m], dtype=np.int64)
    if len(ts) < 2:
        return {"fps_measured": None, "dropped": None}
    dt = np.diff(ts) / 1e9
    expected = 1.0 / fps
    dropped = int(np.clip(np.round(dt / expected) - 1, 0, None).sum())
    return {"fps_measured": round(float(1.0 / np.median(dt)), 2), "dropped": dropped}


def check_faces(frames: list, step: int, gate: dict | None) -> dict:
    """Chạy MTCNN trên mỗi `step` khung để phát hiện sớm video hỏng (mất mặt, quá nhỏ, tối...)."""
    from face_detector import FaceDetector
    from svm_lbp import quality_ok
    from facenet_pytorch import MTCNN

    detector = FaceDetector(MTCNN(keep_all=True, min_face_size=40))
    picked = frames[::step]
    sizes, gate_ok = [], 0
    for rgb in picked:
        face = detector.detect_largest(rgb)
        if face is None:
            continue
        sizes.append(face.face_size)
        gate_ok += bool(quality_ok(face.crop, gate, face.face_size, face.landmarks))
    n = len(picked)
    return {
        "checked_frames": n,
        "face_rate": round(len(sizes) / n, 3) if n else None,
        "gate_rate": round(gate_ok / n, 3) if n else None,
        "median_face_px": int(np.median(sizes)) if sizes else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-modes", action="store_true", help="in các chế độ sensor rồi thoát")
    p.add_argument("--subject", help="mã người, ví dụ p01")
    p.add_argument("--kind", choices=sorted(KINDS), help="real hoặc loại tấn công")
    p.add_argument("--light", default="indoor", help="nhãn ánh sáng, ví dụ indoor, window, dim")
    p.add_argument("--dist", default="50cm", help="nhãn khoảng cách, ví dụ 30cm, 50cm, 80cm")
    p.add_argument("--note", default="", help="ghi chú tự do (thiết bị tấn công, giấy in...)")
    p.add_argument("--root", type=Path, default=Path("dataset/picam"))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--sensor-size", type=int, nargs=2, default=[1640, 1232], metavar=("W", "H"),
                   help="chế độ sensor; 1640x1232 = đủ góc nhìn, 640x480 = góc hẹp")
    p.add_argument("--settle", type=float, default=3.0, help="giây để AE/AWB tự chỉnh trước khi khoá")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--lock-from", type=Path, help="meta.json của video trước để dùng lại thông số khoá")
    p.add_argument("--no-lock", action="store_true", help="không khoá AE/AWB (không khuyến nghị)")
    p.add_argument("--pad-model", type=Path, help="lấy ngưỡng quality gate từ model .joblib (tự tạo)")
    p.add_argument("--check-step", type=int, default=5, help="kiểm tra MTCNN mỗi N khung; 0 = bỏ qua")
    args = p.parse_args()

    if args.list_modes:
        list_modes()
        return
    if not args.subject or not args.kind:
        p.error("cần --subject và --kind (hoặc --list-modes)")
    subject, light, dist = tag(args.subject, "subject"), tag(args.light, "light"), tag(args.dist, "dist")
    label = KINDS[args.kind]

    base = args.root / label / subject
    prefix = f"{subject}_{args.kind}_{light}_{dist}"
    take = next_take(base, prefix)
    video_id = f"{prefix}_t{take:02d}"
    out = base / video_id
    if out.exists():
        raise SystemExit(f"{out} đã tồn tại")
    print(f"Video: {out}")

    frames, frame_md, camera_config, locked = capture(args)
    out.mkdir(parents=True, exist_ok=False)  # chỉ tạo sau khi thu xong, tránh thư mục rỗng

    print(f"Ghi {len(frames)} khung PNG...", flush=True)
    for i, rgb in enumerate(frames, 1):
        cv2.imwrite(str(out / f"frame_{i:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

    stats = timing_stats(frame_md, args.fps)
    check = {}
    if args.check_step > 0:
        gate = None
        if args.pad_model:
            from svm_lbp import load_pad
            gate = load_pad(args.pad_model).get("gate")
        print("Kiểm tra mặt bằng MTCNN...", flush=True)
        check = check_faces(frames, args.check_step, gate)

    created = datetime.now().isoformat(timespec="seconds")
    try:
        import picamera2
        picam_version = getattr(picamera2, "__version__", None)
    except Exception:
        picam_version = None
    meta = {
        "video_id": video_id, "label": label, "label_int": int(label == "real"),
        "subject": subject, "kind": args.kind, "light": light, "dist": dist, "take": take,
        "note": args.note, "created": created, "host": socket.gethostname(),
        "python": sys.version.split()[0], "picamera2": picam_version,
        "args": {k: to_jsonable(str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "camera_config": camera_config, "locked_controls": locked,
        "n_frames": len(frames), **stats, "check": check, "frames": frame_md,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    index = args.root / "index.csv"
    new_file = not index.exists()
    with index.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new_file:
            writer.writeheader()
        sensor = camera_config.get("sensor") or {}
        writer.writerow({
            "video_id": video_id, "label": label, "subject": subject, "kind": args.kind,
            "light": light, "dist": dist, "take": take, "n_frames": len(frames),
            "fps_measured": stats["fps_measured"], "dropped": stats["dropped"],
            "sensor_mode": "x".join(map(str, sensor.get("output_size", []))),
            "exposure_us": locked.get("ExposureTime"), "analogue_gain": locked.get("AnalogueGain"),
            "colour_gains": locked.get("ColourGains"), "face_rate": check.get("face_rate"),
            "gate_rate": check.get("gate_rate"), "median_face_px": check.get("median_face_px"),
            "created": created, "note": args.note,
        })

    print(f"\nĐã lưu {len(frames)} khung vào {out}")
    print(f"fps đo được: {stats['fps_measured']}, khung bị rơi: {stats['dropped']}")
    if check:
        print(f"Có mặt: {check['face_rate']:.0%}, qua quality gate: {check['gate_rate']:.0%}, "
              f"cạnh mặt trung vị: {check['median_face_px']} px (trên {check['checked_frames']} khung)")
        if (check["face_rate"] or 0) < 0.9:
            print("CẢNH BÁO: dưới 90% khung có mặt, nên xoá video này và thu lại.")
        elif (check["gate_rate"] or 0) < 0.5:
            print("Lưu ý: dưới 50% khung qua quality gate (có thể do khoảng cách/ánh sáng). "
                  "Với mẫu giả đây có thể là bình thường; với mẫu thật nên xem lại.")


if __name__ == "__main__":
    main()
