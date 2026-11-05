import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(
    description="Benchmark face detection using an image file."
)
parser.add_argument("image", help="Path to the image used for benchmarking")
image_path = parser.parse_args().image

from face_detector import FaceDetector, read_image_rgb

frame = read_image_rgb(image_path)

start = time.perf_counter()
detector = FaceDetector()
print(f"Khởi tạo model: {(time.perf_counter() - start) * 1000:.1f} ms")


def benchmark(name, run, count_faces, warmup=5, runs=50):
    for _ in range(warmup):  # Bỏ qua các lượt đầu để model ổn định
        result = run()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - start) * 1000)

    print(
        f"{name}: mặt={count_faces(result)}, "
        f"trung bình={np.mean(times):.1f} ms, "
        f"median={np.median(times):.1f} ms, "
        f"p95={np.percentile(times, 95):.1f} ms, "
        f"FPS~{1000 / np.mean(times):.2f}"
    )


benchmark(
    "Phát hiện (ảnh đã đọc)",
    lambda: detector.detect(frame),
    lambda faces: len(faces),
)

benchmark(
    "Đọc file + phát hiện",
    lambda: detector.detect_file(image_path),
    lambda face: int(face is not None),
)