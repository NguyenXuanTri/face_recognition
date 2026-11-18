"""Đọc và cache đặc trưng PAD cho MSU-MFSD và CASIA-FASD (ảnh crop mặt, đã chia Train/Val/Test).

Cách dùng:
    from pad_dataset import load_features
    d = load_features("Train", "lbp+ida", "casia", frame_step=3)
    d.X          # (N, D) float32
    d.y          # real = 1, spoof/attack = 0 (cùng quy ước với svm_lbp.py)
    d.groups     # subject, dùng cho GroupKFold
    d.video      # khoá nhóm video, dùng cho pad_decision / đánh giá theo video

Với mỗi (dataset, split, frame_step), đặc trưng của cả bốn khối được trích MỘT lần rồi lưu .npz:
    lbp_ycbcr 177 | lbp_hsv 177 | ida 121 | moire 4
`mode` chỉ ghép lại các khối này nên đổi cấu hình ablation không phải trích lại.
Mode là các token nối bằng "+": ycbcr, hsv, lbp (= ycbcr + hsv), ida, moire.
Ví dụ: "lbp", "ida", "lbp+ida", "lbp+ida+moire".

Lưu ý:
- Dataset chỉ có crop mặt từ MTCNN, không có context_crop, nên IDA và moiré chạy trên chính crop này.
- Cache tự bị bỏ nếu danh sách ảnh đổi hoặc mã nguồn đặc trưng (ida_features.py, hàm LBP) đổi.
- Thêm thư mục feature_cache/ vào .gitignore.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import ida_features as _ida_module
from ida_features import IDA_DIM, MOIRE_DIM, moire_features
from ida_features import ida_features as compute_ida
from svm_lbp import (
    IMAGE_SIZE, LBP_BINS, _uniform_lbp_lookup, extract_lbp_features, lbp_histogram,
)

SPLITS = ("Train", "Val", "Test")
DATASETS = ("msu", "casia")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

_HERE = Path(__file__).resolve().parent
DEFAULT_ROOTS = {
    "msu": Path(os.environ.get("PAD_MSU_ROOT", _HERE / "dataset" / "MSU-MFSD")),
    "casia": Path(os.environ.get("PAD_CASIA_ROOT", _HERE / "dataset" / "CASIA-FASD")),
}
DEFAULT_CACHE_DIR = _HERE / "feature_cache"

LBP_HALF = LBP_BINS * 3  # 3 kênh x 59 bin
BLOCK_ORDER = ("lbp_ycbcr", "lbp_hsv", "ida", "moire")
BLOCK_DIMS = {"lbp_ycbcr": LBP_HALF, "lbp_hsv": LBP_HALF, "ida": IDA_DIM, "moire": MOIRE_DIM}
TOKEN_BLOCKS = {
    "ycbcr": ("lbp_ycbcr",),
    "hsv": ("lbp_hsv",),
    "lbp": ("lbp_ycbcr", "lbp_hsv"),
    "ida": ("ida",),
    "moire": ("moire",),
}


# ----------------------------------------------------------------------------
# Parse tên thư mục / tên file
# ----------------------------------------------------------------------------
_MSU_RE = re.compile(
    r"^(attack|real)_(client\d+)_(android|laptop)_SD_"
    r"(?:(ipad_video|iphone_video|printed_photo)_)?scene01$"
)
_CASIA_RE = re.compile(r"^s(\d+)v(HR_\d|\d)f(\d+)")
_CASIA_REAL = {"1", "2", "HR_1"}
# Theo trí nhớ, cần đối chiếu README gốc của CASIA-FASD trước khi ghi vào báo cáo.
_CASIA_ATTACK = {
    "3": "warped", "4": "warped", "HR_2": "warped",
    "5": "cut", "6": "cut", "HR_3": "cut",
    "7": "replay", "8": "replay", "HR_4": "replay",
}


def parse_msu(video_dir: Path) -> Dict:
    """Một video MSU = một thư mục. Trả về label (real=1), subject, video_id, camera, attack_type."""
    name = Path(video_dir).name
    m = _MSU_RE.match(name)
    if not m:
        raise ValueError(f"MSU: tên thư mục không khớp quy ước: {name}")
    kind, subject, camera, attack = m.groups()
    if (kind == "attack") != (attack is not None):
        raise ValueError(f"MSU: loại tấn công không khớp nhãn lớp: {name}")
    return {
        "label": 1 if kind == "real" else 0,
        "subject": subject,
        "video_id": name,
        "camera": camera,
        "attack_type": attack or "real",
        "frame": 0,
    }


def parse_casia(path: Path) -> Dict:
    """Mã video nằm trong tên file. 'camera' ở đây là chất lượng: low (video lẻ), normal (chẵn), high (HR)."""
    name = Path(path).name
    m = _CASIA_RE.match(name)
    if not m:
        raise ValueError(f"CASIA: tên file không khớp quy ước: {name}")
    subj, code, frame = m.groups()
    if code not in _CASIA_REAL and code not in _CASIA_ATTACK:
        raise ValueError(f"CASIA: mã video lạ '{code}' trong {name}")
    subject = f"s{subj}"
    label = 1 if code in _CASIA_REAL else 0
    if code.startswith("HR"):
        quality = "high"
    else:
        quality = "low" if int(code) % 2 else "normal"
    return {
        "label": label,
        "subject": subject,
        "video_id": f"{subject}_v{code}",
        "camera": quality,
        "attack_type": "real" if label else _CASIA_ATTACK[code],
        "frame": int(frame),
    }


# ----------------------------------------------------------------------------
# Liệt kê ảnh (đã lấy mẫu theo frame_step trong từng video)
# ----------------------------------------------------------------------------
def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _list_msu(root: Path, split: str, step: int) -> List[Dict]:
    items: List[Dict] = []
    for cls, want in (("attack", 0), ("real", 1)):
        cdir = root / split / cls
        if not cdir.is_dir():
            raise NotADirectoryError(f"Thiếu thư mục: {cdir}")
        for vdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            meta = parse_msu(vdir)
            if meta["label"] != want:
                raise ValueError(f"{vdir}: nhãn theo tên không khớp thư mục lớp '{cls}'")
            frames = sorted(f for f in vdir.iterdir() if _is_image(f))
            for f in frames[::step]:
                items.append({**meta, "path": f})
    return items


def _list_casia(root: Path, split: str, step: int) -> List[Dict]:
    items: List[Dict] = []
    for cls, want in (("spoof", 0), ("real", 1)):
        cdir = root / split / cls
        if not cdir.is_dir():
            raise NotADirectoryError(f"Thiếu thư mục: {cdir}")
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            by_video: Dict[str, List[Tuple[int, Path, Dict]]] = {}
            for f in sdir.iterdir():
                if not _is_image(f):
                    continue
                meta = parse_casia(f)
                if meta["subject"] != sdir.name:
                    raise ValueError(f"{f}: subject trong tên file khác thư mục '{sdir.name}'")
                if meta["label"] != want:
                    raise ValueError(f"{f}: nhãn theo mã video không khớp thư mục lớp '{cls}'")
                by_video.setdefault(meta["video_id"], []).append((meta["frame"], f, meta))
            for vid in sorted(by_video):
                for _, f, meta in sorted(by_video[vid], key=lambda t: t[0])[::step]:
                    items.append({**meta, "path": f})
    return items


_LISTERS = {"msu": _list_msu, "casia": _list_casia}


def _check_structure(dataset: str, items: List[Dict]) -> None:
    """Cảnh báo (không dừng) nếu subject thiếu video: CASIA 3 real + 9 spoof, MSU 2 real + 6 attack."""
    want_real, want_spoof = {"casia": (3, 9), "msu": (2, 6)}[dataset]
    by_subject: Dict[str, Dict[int, set]] = {}
    for it in items:
        by_subject.setdefault(it["subject"], {}).setdefault(it["label"], set()).add(it["video_id"])
    problems = []
    for subject, v in sorted(by_subject.items()):
        n_real, n_spoof = len(v.get(1, ())), len(v.get(0, ()))
        if (n_real, n_spoof) != (want_real, want_spoof):
            problems.append(f"{subject}: real={n_real}/{want_real}, spoof={n_spoof}/{want_spoof}")
    if problems:
        logging.warning("Cấu trúc %s bất thường ở %d subject: %s", dataset, len(problems), "; ".join(problems[:8]))


# ----------------------------------------------------------------------------
# Trích đặc trưng
# ----------------------------------------------------------------------------
def _extract_one(path: str):
    """Trả về ((lbp_ycbcr, lbp_hsv, ida, moire), None) hoặc (None, thông_báo_lỗi)."""
    try:
        cv2.setNumThreads(1)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("không đọc được ảnh")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lbp = extract_lbp_features(rgb, "both")  # [Y, Cb, Cr, H, S, V] -> 177 + 177
        return (lbp[:LBP_HALF], lbp[LBP_HALF:], compute_ida(rgb), moire_features(rgb)), None
    except Exception as error:  # ảnh hỏng không được làm dừng cả lượt trích
        return None, f"{path}: {error}"


def _extract_all(paths: List[str], n_jobs: int) -> List:
    total, results = len(paths), []
    pool = None if n_jobs <= 1 else ProcessPoolExecutor(max_workers=n_jobs)
    try:
        it = map(_extract_one, paths) if pool is None else pool.map(_extract_one, paths, chunksize=64)
        for i, r in enumerate(it, 1):
            results.append(r)
            if i % 5000 == 0 or i == total:
                print(f"  {i}/{total}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return results


def _signature() -> str:
    """Đổi khi mã nguồn đặc trưng đổi (kể cả sửa chú thích trong ida_features.py: chấp nhận được)."""
    h = hashlib.sha1()
    h.update(Path(_ida_module.__file__).read_bytes())
    for fn in (extract_lbp_features, lbp_histogram, _uniform_lbp_lookup):
        h.update(inspect.getsource(fn).encode())
    h.update(repr((IMAGE_SIZE, LBP_BINS)).encode())
    return h.hexdigest()[:16]


def _cache_file(cache_dir: Path, dataset: str, split: str, step: int) -> Path:
    return cache_dir / f"{dataset}_{split.lower()}_step{step}.npz"


def _read_cache(file: Path, sig: str, phash: str) -> Optional[Dict[str, np.ndarray]]:
    if not file.is_file():
        return None
    try:
        with np.load(file, allow_pickle=False) as z:
            if str(z["sig"]) != sig or str(z["phash"]) != phash:
                print(f"Cache cũ ({file.name}), trích lại.")
                return None
            return {k: z[k] for k in z.files}
    except Exception as error:
        print(f"Không đọc được cache {file.name} ({error}), trích lại.")
        return None


def _build(
    dataset: str, split: str, root: Path, step: int, n_jobs: int, sig: str, phash: str,
    items: List[Dict],
) -> Dict[str, np.ndarray]:
    print(f"Trích đặc trưng {dataset}/{split}: {len(items)} ảnh, {n_jobs} tiến trình")
    results = _extract_all([str(it["path"]) for it in items], n_jobs)
    kept = [(it, r[0]) for it, r in zip(items, results) if r[0] is not None]
    failed = [r[1] for r in results if r[0] is None]
    if failed:
        print(f"Bỏ {len(failed)} ảnh lỗi, ví dụ: {failed[:3]}")
    if not kept:
        raise ValueError(f"Không trích được ảnh nào trong {dataset}/{split}")
    data: Dict[str, np.ndarray] = {
        f"b_{name}": np.stack([feats[i] for _, feats in kept]).astype(np.float32)
        for i, name in enumerate(BLOCK_ORDER)
    }
    data["y"] = np.array([it["label"] for it, _ in kept], dtype=np.int64)
    for key in ("subject", "video_id", "camera", "attack_type"):
        data[key] = np.array([it[key] for it, _ in kept])
    data["path"] = np.array([it["path"].relative_to(root).as_posix() for it, _ in kept])
    data["sig"], data["phash"] = np.array(sig), np.array(phash)
    return data


# ----------------------------------------------------------------------------
# API chính
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PadData:
    X: np.ndarray
    y: np.ndarray        # real = 1, spoof = 0
    subject: np.ndarray
    video: np.ndarray    # MSU: tên thư mục; CASIA: "<subject>_v<mã>"
    camera: np.ndarray   # MSU: android/laptop; CASIA: low/normal/high
    attack: np.ndarray   # "real" hoặc loại tấn công
    path: np.ndarray
    mode: str

    @property
    def groups(self) -> np.ndarray:
        return self.subject


def resolve_mode(mode: str) -> Tuple[str, ...]:
    tokens = mode.lower().replace("é", "e").replace(" ", "").split("+")
    blocks = set()
    for tok in tokens:
        if tok not in TOKEN_BLOCKS:
            raise ValueError(f"Token '{tok}' không hợp lệ; chọn trong {sorted(TOKEN_BLOCKS)}")
        blocks.update(TOKEN_BLOCKS[tok])
    return tuple(b for b in BLOCK_ORDER if b in blocks)


def load_features(
    split: str,
    mode: str,
    dataset: str,
    *,
    root: Optional[Path] = None,
    frame_step: int = 1,
    cache_dir: Optional[Path] = None,
    n_jobs: Optional[int] = None,
    rebuild: bool = False,
) -> PadData:
    """Đặc trưng của một tập (Train/Val/Test) của 'msu' hoặc 'casia', theo `mode`.

    frame_step=k: lấy 1 khung trong mỗi k khung của từng video (k=1: lấy hết).
    Dùng cùng frame_step cho mọi cấu hình khi so sánh ablation.
    """
    split = split.capitalize()
    dataset = dataset.lower()
    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {SPLITS}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset phải thuộc {DATASETS}")
    if frame_step < 1:
        raise ValueError("frame_step phải >= 1")
    blocks = resolve_mode(mode)

    root = Path(root) if root is not None else DEFAULT_ROOTS[dataset]
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)

    items = _LISTERS[dataset](root, split, frame_step)
    if not items:
        raise ValueError(f"Không có ảnh nào trong {root / split}")
    sig = _signature()
    phash = hashlib.sha1(
        "\n".join(it["path"].relative_to(root).as_posix() for it in items).encode()
    ).hexdigest()

    file = _cache_file(cache_dir, dataset, split, frame_step)
    data = None if rebuild else _read_cache(file, sig, phash)
    if data is None:
        _check_structure(dataset, items)
        data = _build(dataset, split, root, frame_step, n_jobs, sig, phash, items)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(file.stem + ".tmp.npz")
        np.savez(tmp, **data)
        os.replace(tmp, file)

    return PadData(
        X=np.concatenate([data[f"b_{b}"] for b in blocks], axis=1),
        y=data["y"],
        subject=data["subject"],
        video=data["video_id"],
        camera=data["camera"],
        attack=data["attack_type"],
        path=np.array([str(root / p) for p in data["path"]]),
        mode=mode,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Trích và cache đặc trưng PAD (Color LBP, IDA, moiré).")
    p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    p.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    p.add_argument("--mode", default="lbp+ida+moire", help="chỉ để in kích thước; cache luôn chứa đủ khối")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--root", type=Path, default=None, help="chỉ dùng khi chọn một dataset")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    if args.root is not None and args.dataset == "all":
        p.error("--root cần đi kèm --dataset msu hoặc casia")

    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        for split in SPLITS if args.split == "all" else (args.split,):
            d = load_features(
                split, args.mode, dataset, root=args.root, frame_step=args.frame_step,
                n_jobs=args.jobs, rebuild=args.rebuild,
            )
            print(
                f"{dataset}/{split} [{args.mode}] X={d.X.shape} real={int((d.y == 1).sum())} "
                f"spoof={int((d.y == 0).sum())} subjects={len(set(d.subject))} videos={len(set(d.video))}"
            )


if __name__ == "__main__":
    main()
