"""Baseline Color LBP + SVM tuyến tính cho PAD: so sánh 3 chế độ ycbcr, hsv, lbp (= ycbcr + hsv).

Chạy trên tập đã chia sẵn Train / Val / Test của MSU-MFSD và CASIA-FASD (xem pad_dataset.py).
Đặt file này cạnh svm_lbp.py và pad_dataset.py.

Quy trình cho mỗi (dataset, mode):
  1. Chọn C bằng GroupKFold theo subject trên Train (tiêu chí: EER của điểm out-of-fold).
  2. Huấn luyện SVM trên Train với C đã chọn (class_weight='balanced').
  3. Chốt ngưỡng tại EER trên Val.
  4. Đo APCER / BPCER / HTER trên Test MỘT lần với ngưỡng cố định đó, ở mức khung và mức video
     (mức video = trung vị điểm các khung của cùng một video).

Ví dụ:
    python pad_dataset.py --frame-step 3                 # trích và cache đặc trưng (làm một lần)
    python baseline_pad.py --frame-step 3                # cả hai dataset, cả ba chế độ
    python baseline_pad.py --datasets casia --modes lbp --frame-step 5   # chạy thử nhanh
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pad_dataset import DATASETS, PadData, load_features  # noqa: E402
from svm_lbp import eer_threshold, pad_metrics, save_pad  # noqa: E402

MODES = ("ycbcr", "hsv", "lbp")
# Tên chế độ trong pad_dataset -> tên chế độ mà svm_lbp.extract_lbp_features hiểu (dùng khi lưu model).
SVM_LBP_MODE = {"ycbcr": "ycbcr", "hsv": "hsv", "lbp": "both"}


def make_svm(c: float) -> SVC:
    return SVC(kernel="linear", C=c, class_weight="balanced")


def choose_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, c_grid: Sequence[float], n_splits: int
) -> Tuple[float, Dict[float, float]]:
    """Chọn C theo EER của điểm out-of-fold (GroupKFold theo subject, không rò subject giữa các fold)."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Cần ít nhất 2 subject trong Train để chạy GroupKFold")
    cv_eer: Dict[float, float] = {}
    for c in c_grid:
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
            oof[te] = make_svm(c).fit(x[tr], y[tr]).decision_function(x[te])
        cv_eer[c] = eer_threshold(y, oof)[1]
        print(f"    C={c:g}: CV EER = {cv_eer[c]:.4f}", flush=True)
    best = min(cv_eer, key=cv_eer.get)  # hoà thì lấy C nhỏ nhất (mô hình đơn giản hơn)
    return best, cv_eer


def video_level(d: PadData, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trung vị điểm theo video. Trả về (nhãn video, điểm video)."""
    vids, inverse = np.unique(d.video, return_inverse=True)
    v_score = np.array([np.median(score[inverse == i]) for i in range(len(vids))])
    v_label = np.array([d.y[inverse == i][0] for i in range(len(vids))])
    return v_label, v_score


def evaluate(dataset: str, mode: str, args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    print(f"\n=== {dataset.upper()} | mode={mode} ===", flush=True)
    kw = dict(frame_step=args.frame_step, n_jobs=args.jobs)
    tr = load_features("Train", mode, dataset, **kw)
    va = load_features("Val", mode, dataset, **kw)
    te = load_features("Test", mode, dataset, **kw)
    print(
        f"  Train {tr.X.shape} | Val {va.X.shape} | Test {te.X.shape} | "
        f"subject: {len(set(tr.subject))}/{len(set(va.subject))}/{len(set(te.subject))}"
    )
    for name, d in (("Train", tr), ("Val", va), ("Test", te)):
        if len(np.unique(d.y)) < 2:
            raise ValueError(f"{dataset}/{name} thiếu một lớp")

    c, cv_eer = choose_c(tr.X, tr.y, tr.groups, args.c_grid, args.cv_splits)
    model = make_svm(c).fit(tr.X, tr.y)

    thr, val_eer = eer_threshold(va.y, model.decision_function(va.X))
    s_te = model.decision_function(te.X)
    frame = pad_metrics(te.y, s_te, thr)
    v_label, v_score = video_level(te, s_te)
    video = pad_metrics(v_label, v_score, thr)
    test_eer = eer_threshold(te.y, s_te)[1]  # chỉ để tham khảo/so với bài báo; không dùng để chọn gì

    row = {
        "dataset": dataset, "mode": mode, "frame_step": args.frame_step, "C": c,
        "dim": tr.X.shape[1], "cv_eer": cv_eer[c], "val_eer": val_eer, "threshold": thr,
        "frame_APCER": frame["APCER"], "frame_BPCER": frame["BPCER"], "frame_HTER": frame["HTER"],
        "video_APCER": video["APCER"], "video_BPCER": video["BPCER"], "video_HTER": video["HTER"],
        "test_eer_reference": test_eer,
        "n_test_frames": len(te.y), "n_test_videos": len(v_label),
    }
    print(
        f"  C={c:g} | thr={thr:.4f} | Val EER={val_eer:.4f}\n"
        f"  Test mức khung : APCER={frame['APCER']:.4f} BPCER={frame['BPCER']:.4f} HTER={frame['HTER']:.4f}\n"
        f"  Test mức video : APCER={video['APCER']:.4f} BPCER={video['BPCER']:.4f} HTER={video['HTER']:.4f}"
    )

    # APCER theo từng loại tấn công (mức khung) để phân tích phụ.
    by_attack = []
    for atk in sorted(set(te.attack) - {"real"}):
        m = (te.attack == atk)
        by_attack.append({
            "dataset": dataset, "mode": mode, "attack_type": atk, "n_frames": int(m.sum()),
            "APCER": float(np.mean(s_te[m] >= thr)),
        })
        print(f"    {atk:<14} APCER={by_attack[-1]['APCER']:.4f} (n={by_attack[-1]['n_frames']})")

    if args.save_models:
        out = Path(args.out) / "models" / f"pad_{dataset}_{mode}.joblib"
        save_pad(model, thr, SVM_LBP_MODE[mode], out)
        print(f"  Đã lưu model: {out}")
    return row, by_attack


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline Color LBP + SVM: ycbcr / hsv / lbp.")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--frame-step", type=int, default=3, help="lấy 1 khung mỗi k khung; dùng chung cho mọi cấu hình")
    p.add_argument("--c-grid", type=lambda s: [float(v) for v in s.split(",")], default=[0.1, 1.0, 10.0])
    p.add_argument("--cv-splits", type=int, default=4)
    p.add_argument("--jobs", type=int, default=None, help="số tiến trình trích đặc trưng lần đầu")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    all_rows: List[Dict] = []
    all_attack: List[Dict] = []
    for dataset in args.datasets:
        rows = []
        for mode in args.modes:
            row, by_attack = evaluate(dataset, mode, args)
            rows.append(row)
            all_attack.extend(by_attack)
        write_csv(Path(args.out) / f"baseline_{dataset}.csv", rows)
        all_rows.extend(rows)

    print("\n=== TỔNG HỢP (Test, ngưỡng cố định từ Val; HTER mức khung / mức video) ===")
    print(f"{'dataset':<7} {'mode':<6} {'C':>5} {'ValEER':>8} {'APCER':>7} {'BPCER':>7} {'HTER':>7} {'vHTER':>7}")
    for r in all_rows:
        print(
            f"{r['dataset']:<7} {r['mode']:<6} {r['C']:>5g} {r['val_eer']:>8.4f} {r['frame_APCER']:>7.4f} "
            f"{r['frame_BPCER']:>7.4f} {r['frame_HTER']:>7.4f} {r['video_HTER']:>7.4f}"
        )
    if all_attack:
        write_csv(Path(args.out) / "baseline_by_attack.csv", all_attack)
    print(f"\nĐã ghi CSV vào: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()