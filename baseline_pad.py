"""Baseline Color LBP + SVM tuyến tính cho PAD: so sánh 3 chế độ ycbcr, hsv, lbp (= ycbcr + hsv).

Chạy trên tập đã chia sẵn Train / Val / Test của MSU-MFSD và CASIA-FASD (xem pad_dataset.py).
Đặt file này cạnh svm_lbp.py và pad_dataset.py.

Quy trình cho mỗi (dataset, mode):
  1. Chọn C bằng GroupKFold theo subject trên Train (tiêu chí: EER của điểm out-of-fold).
     Mặc định có StandardScaler trước SVM (tắt bằng --no-scale).
  2. Huấn luyện SVM trên Train với C đã chọn (class_weight='balanced').
  3. Chốt ngưỡng tại EER trên Val (mặc định) hoặc trên điểm out-of-fold của Train (--thr-source oof).
  4. Đo APCER / BPCER / HTER trên Test MỘT lần với ngưỡng cố định đó, ở mức khung và mức video
     (mức video = trung vị điểm các khung của cùng một video).

Ví dụ:
    python pad_dataset.py --frame-step 3                 # trích và cache đặc trưng (làm một lần)
    python baseline_pad.py --frame-step 3                # cả hai dataset, cả ba chế độ
    python baseline_pad.py --datasets casia --modes lbp --frame-step 5   # chạy thử nhanh
    # Ablation (cần đã chạy pad_dataset.py --frame-step 3), kèm đánh giá chéo dataset:
    python baseline_pad.py --modes lbp ida lbp+ida lbp+ida+moire --frame-step 3 --cross --tag ablation \
        --c-grid 0.00001,0.0001,0.001,0.01,0.1,1 --out results_ablation
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pad_dataset import DATASETS, PadData, load_features, resolve_mode  # noqa: E402
from svm_lbp import eer_threshold, pad_metrics, save_pad  # noqa: E402

MODES = ("ycbcr", "hsv", "lbp")  # baseline mặc định; ablation: --modes lbp ida lbp+ida lbp+ida+moire
# Tên chế độ trong pad_dataset -> tên chế độ mà svm_lbp.extract_lbp_features hiểu (dùng khi lưu model).
# Chế độ có IDA/moiré chưa có hàm suy luận trong svm_lbp.py (sẽ làm ở pipeline.py) nên không lưu model.
SVM_LBP_MODE = {"ycbcr": "ycbcr", "hsv": "hsv", "lbp": "both"}


def _mode_arg(text: str) -> str:
    """Chấp nhận mọi chế độ hợp lệ của pad_dataset: ycbcr, hsv, lbp, ida, moire và tổ hợp nối bằng '+'."""
    try:
        resolve_mode(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    return text


def make_svm(c: float, scale: bool = True):
    """SVM tuyến tính. scale=True: chuẩn hoá từng đặc trưng (trung bình 0, độ lệch 1) trước khi vào SVM.

    Scaler được fit lại trên dữ liệu huấn luyện của từng vòng CV nên không rò dữ liệu. Đặc trưng LBP là các
    tỉ lệ rất nhỏ (~0,02), không chuẩn hoá thì C bị đẩy lên rất lớn mới học nổi. Pipeline vẫn có
    decision_function(x_thô), nên file .joblib vẫn dùng được với svm_lbp.score_frame.
    """
    svm = SVC(kernel="linear", C=c, class_weight="balanced")
    return make_pipeline(StandardScaler(), svm) if scale else svm


def choose_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, c_grid: Sequence[float], n_splits: int,
    scale: bool = True,
) -> Tuple[float, Dict[float, float], np.ndarray]:
    """Chọn C theo EER của điểm out-of-fold (GroupKFold theo subject, không rò subject giữa các fold).

    Trả về (C tốt nhất, CV EER của từng C, mảng điểm out-of-fold của C tốt nhất).
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Cần ít nhất 2 subject trong Train để chạy GroupKFold")
    cv_eer: Dict[float, float] = {}
    oofs: Dict[float, np.ndarray] = {}
    for c in c_grid:
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
            oof[te] = make_svm(c, scale).fit(x[tr], y[tr]).decision_function(x[te])
        cv_eer[c] = eer_threshold(y, oof)[1]
        oofs[c] = oof
        print(f"    C={c:g}: CV EER = {cv_eer[c]:.4f}", flush=True)
    best = min(cv_eer, key=cv_eer.get)  # hoà thì lấy C đứng trước trong lưới (lưới tăng dần: C nhỏ hơn)
    return best, cv_eer, oofs[best]


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

    scale = not args.no_scale
    c, cv_eer, oof_best = choose_c(tr.X, tr.y, tr.groups, args.c_grid, args.cv_splits, scale)
    if len(args.c_grid) > 1 and c in (min(args.c_grid), max(args.c_grid)):
        print(f"  CẢNH BÁO: C={c:g} nằm ở đầu mút của lưới, nên thêm giá trị ngoài đó vào --c-grid")
    model = make_svm(c, scale).fit(tr.X, tr.y)

    thr_val, val_eer = eer_threshold(va.y, model.decision_function(va.X))
    # val: ngưỡng tại EER trên Val (ít subject nên dao động); oof: tại EER trên điểm out-of-fold của Train (nhiều subject hơn).
    thr = thr_val if args.thr_source == "val" else eer_threshold(tr.y, oof_best)[0]
    s_te = model.decision_function(te.X)
    frame = pad_metrics(te.y, s_te, thr)
    v_label, v_score = video_level(te, s_te)
    video = pad_metrics(v_label, v_score, thr)
    test_eer = eer_threshold(te.y, s_te)[1]  # chỉ để tham khảo/so với bài báo; không dùng để chọn gì

    # Cross-dataset: mô hình và ngưỡng của dataset này (ngưỡng từ Val của chính nó) chấm Test của dataset kia.
    cross: Dict = {}
    if args.cross:
        other = next(d for d in DATASETS if d != dataset)
        xt = load_features("Test", mode, other, **kw)
        xs = model.decision_function(xt.X)
        xf = pad_metrics(xt.y, xs, thr)
        xl, xv = video_level(xt, xs)
        xvid = pad_metrics(xl, xv, thr)
        # EER trên dataset đích KHÔNG phụ thuộc ngưỡng: cho biết đặc trưng còn phân biệt được thật/giả hay không,
        # tách khỏi việc ngưỡng của dataset nguồn lệch khi sang dataset đích (HTER chéo có thể gần 50% chỉ vì lệch ngưỡng).
        x_eer = eer_threshold(xt.y, xs)[1]
        xv_eer = eer_threshold(xl, xv)[1]
        cross = {
            "cross_test_on": other, "cross_frame_APCER": xf["APCER"], "cross_frame_BPCER": xf["BPCER"],
            "cross_frame_HTER": xf["HTER"], "cross_video_HTER": xvid["HTER"],
            "cross_frame_EER": x_eer, "cross_video_EER": xv_eer,
        }
        print(
            f"  Cross-dataset ({dataset} -> {other}) : khung HTER={xf['HTER']:.4f} "
            f"(APCER={xf['APCER']:.4f} BPCER={xf['BPCER']:.4f}) | video HTER={xvid['HTER']:.4f} | "
            f"EER (không phụ thuộc ngưỡng): khung={x_eer:.4f} video={xv_eer:.4f}"
        )

    row = {
        "dataset": dataset, "mode": mode, "frame_step": args.frame_step, "scaled": scale, "C": c,
        "dim": tr.X.shape[1], "cv_eer": cv_eer[c], "val_eer": val_eer, "threshold": thr,
        "thr_source": args.thr_source, "thr_val": thr_val,
        "frame_APCER": frame["APCER"], "frame_BPCER": frame["BPCER"], "frame_HTER": frame["HTER"],
        "video_APCER": video["APCER"], "video_BPCER": video["BPCER"], "video_HTER": video["HTER"],
        "test_eer_reference": test_eer,
        "n_test_frames": len(te.y), "n_test_videos": len(v_label),
        **cross,
    }
    print(
        f"  C={c:g} | thr={thr:.4f} (nguồn: {args.thr_source}; thr_val={thr_val:.4f}) | Val EER={val_eer:.4f}\n"
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
        if mode in SVM_LBP_MODE:
            out = Path(args.out) / "models" / f"pad_{dataset}_{mode}.joblib"
            save_pad(model, thr, SVM_LBP_MODE[mode], out)
            print(f"  Đã lưu model: {out}")
        else:
            print(f"  Không lưu model cho '{mode}' (svm_lbp.py chưa trích được đặc trưng này khi suy luận)")
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
    p.add_argument(
        "--modes", nargs="+", type=_mode_arg, default=list(MODES),
        help="ycbcr, hsv, lbp, ida, moire hoặc tổ hợp nối bằng '+', ví dụ lbp+ida+moire",
    )
    p.add_argument("--cross", action="store_true", help="thêm đánh giá chéo dataset (huấn luyện ở A, thử Test của B)")
    p.add_argument("--tag", default="baseline", help="tiền tố tên file CSV (tránh ghi đè giữa các lần chạy)")
    p.add_argument("--frame-step", type=int, default=3, help="lấy 1 khung mỗi k khung; dùng chung cho mọi cấu hình")
    p.add_argument(
        "--c-grid", type=lambda s: [float(v) for v in s.split(",")],
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], help="liệt kê TĂNG DẦN (hoà thì lấy C đứng trước)",
    )
    p.add_argument("--no-scale", action="store_true", help="không chuẩn hoá đặc trưng (hành vi của bản cũ)")
    p.add_argument(
        "--thr-source", choices=("val", "oof"), default="val",
        help="val: ngưỡng EER trên Val (mặc định); oof: ngưỡng EER trên điểm out-of-fold của Train",
    )
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
        write_csv(Path(args.out) / f"{args.tag}_{dataset}.csv", rows)
        all_rows.extend(rows)

    print(f"\n=== TỔNG HỢP (Test, ngưỡng cố định từ {args.thr_source}; HTER mức khung / mức video) ===")
    has_x = bool(all_rows) and "cross_frame_HTER" in all_rows[0]
    print(
        f"{'dataset':<7} {'mode':<14} {'C':>7} {'ValEER':>8} {'APCER':>7} {'BPCER':>7} {'HTER':>7} {'vHTER':>7}"
        + (f" {'xHTER':>7} {'xvHTER':>7} {'xEER':>7} {'xvEER':>7}" if has_x else "")
    )
    for r in all_rows:
        line = (
            f"{r['dataset']:<7} {r['mode']:<14} {r['C']:>7g} {r['val_eer']:>8.4f} {r['frame_APCER']:>7.4f} "
            f"{r['frame_BPCER']:>7.4f} {r['frame_HTER']:>7.4f} {r['video_HTER']:>7.4f}"
        )
        if has_x:
            line += (
                f" {r['cross_frame_HTER']:>7.4f} {r['cross_video_HTER']:>7.4f}"
                f" {r['cross_frame_EER']:>7.4f} {r['cross_video_EER']:>7.4f}"
            )
        print(line)
    if all_attack:
        write_csv(Path(args.out) / f"{args.tag}_by_attack.csv", all_attack)
    print(f"\nĐã ghi CSV vào: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()