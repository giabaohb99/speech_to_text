"""Đánh giá hai kịch bản bằng kiểm tra chéo, chỉ hai tập train và test mỗi vòng.

- A, người đã có: bỏ từng mẫu ra làm test, train là toàn bộ phần còn lại (leave-one-sample-out).
- B, giọng mới: bỏ cả một người (hoặc một nhóm biến thể cùng giọng gốc) ra làm test,
  train là những người còn lại (leave-one-speaker-out).

Thống kê chuẩn hóa học lại trên tập train của từng vòng ở kịch bản B. Ở kịch bản A,
một mẫu bị bỏ ra không làm đổi thống kê đáng kể nên học một lần trên cả kho.

Mỗi lần chạy lưu vào results/evaluate/<thời điểm>/ để không ghi đè lần trước.
"""
import re
import time
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.classifier import load_summary_matrix, predict_classifier, single_query_ms  # noqa: E402
from src.dtw import cross_distances  # noqa: E402
from src.recognizer import FeatureTransform, Recognizer, vote  # noqa: E402
from src.store import load_bank  # noqa: E402

SCENARIOS = {"A": "Người đã có", "B": "Giọng mới"}
METHOD_SLUGS = {"DTW + F0": "dtw_f0", "DTW không F0": "dtw_khong_f0", "SVM": "svm",
                "Random Forest": "random_forest"}
CLASSIFIERS = ["SVM", "Random Forest"]
TTS_GROUP = re.compile(r"^(tts_[a-z0-9]+)_p\d+$")
SWEEP_K = [1, 3, 5, 7]
SWEEP_F0 = [0.0, 1.0, 2.0, 4.0]
SWEEP_CMN = [False, True]


def speaker_group(speaker):
    """Biến thể sinh từ cùng giọng gốc là một nhóm: tts_f1_p3 -> tts_f1. Người thật giữ nguyên tên."""
    match = TTS_GROUP.match(speaker)
    return match.group(1) if match else speaker


def fold_indices(bank):
    """Các vòng chia dùng chung cho mọi phương pháp: {kịch bản: [(chỉ số test, chỉ số train)]}."""
    n = len(bank.meta)
    everything = np.arange(n)
    groups = bank.meta["speaker"].map(speaker_group).to_numpy()
    folds = {"A": [(np.array([i]), np.delete(everything, i)) for i in range(n)], "B": []}
    for group in np.unique(groups):
        test, train = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
        if len(train):
            folds["B"].append((test, train))
    return folds


def compute_folds(bank, cfg, f0_weight, cmn):
    """Tính ma trận khoảng cách cho từng vòng. Trả về {kịch bản: [(chỉ số test, chỉ số train, D)]}."""
    dcfg = cfg["dtw"]
    seqs = bank.sequences

    # Kịch bản A gộp 1000 vòng bỏ-một-mẫu thành một ma trận, đường chéo vô cùng để không tự khớp
    transform = FeatureTransform(cfg, f0_weight, cmn).fit(seqs)
    prepared = [transform.apply(s) for s in seqs]
    everything = np.arange(len(seqs))
    dist = cross_distances(prepared, prepared, dcfg["band_ratio"], dcfg["band_min_frames"])
    np.fill_diagonal(dist, np.inf)
    folds = {"A": [(everything, everything, dist)], "B": []}

    for test, train in fold_indices(bank)["B"]:
        transform = FeatureTransform(cfg, f0_weight, cmn).fit([seqs[i] for i in train])
        test_seqs = [transform.apply(seqs[i]) for i in test]
        train_seqs = [transform.apply(seqs[i]) for i in train]
        folds["B"].append((test, train, cross_distances(test_seqs, train_seqs,
                                                        dcfg["band_ratio"], dcfg["band_min_frames"])))
    return folds


def predict_folds(bank, folds, k):
    """Bỏ phiếu cho mọi mẫu test. Trả về DataFrame một dòng một lần dự đoán."""
    meta = bank.meta
    words = meta["word"].to_numpy()
    file_ids = meta["file_id"].to_numpy()
    rows = []
    for scenario, fold_list in folds.items():
        for test, train, dist in fold_list:
            for row, i in enumerate(test):
                pred = vote(dist[row], words[train], file_ids[train], k)
                rows.append({"scenario": scenario, "file_id": file_ids[i], "word": words[i],
                             "speaker": meta.at[i, "speaker"], "group": speaker_group(meta.at[i, "speaker"]),
                             "predicted": pred.word, "confidence": round(pred.confidence, 4),
                             "nearest_file": pred.neighbors[0][0]})
    df = pd.DataFrame(rows)
    df["correct"] = df["word"] == df["predicted"]
    return df


def classifier_predictions(bank, predictions):
    """Đổi kết quả của classifier.predict_classifier sang cùng dạng bảng với predict_folds."""
    meta = bank.meta
    frames = []
    for scenario, predicted in predictions.items():
        tested = np.array([p is not None for p in predicted])
        part = meta.loc[tested, ["file_id", "word", "speaker"]].copy()
        part.insert(0, "scenario", scenario)
        part["group"] = part["speaker"].map(speaker_group)
        part["predicted"] = predicted[tested]
        part["confidence"] = np.nan
        part["nearest_file"] = ""
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["word"] == df["predicted"]
    return df


def accuracy_table(preds):
    rows = []
    for (method, scenario), part in preds.groupby(["method", "scenario"], sort=False):
        rows.append({"method": method, "scenario": scenario, "speaker": "TẤT CẢ",
                     "n": len(part), "correct": int(part["correct"].sum()), "accuracy": part["correct"].mean()})
        for speaker, sub in part.groupby("speaker"):
            rows.append({"method": method, "scenario": scenario, "speaker": speaker,
                         "n": len(sub), "correct": int(sub["correct"].sum()), "accuracy": sub["correct"].mean()})
    return pd.DataFrame(rows)


def confusion(part, word_codes):
    return pd.crosstab(part["word"], part["predicted"]).reindex(
        index=word_codes, columns=word_codes, fill_value=0)


def plot_confusion(matrix, words, title, out_path):
    labels = [words[c] for c in matrix.index]
    counts = matrix.to_numpy()
    rate = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9.5), constrained_layout=True)
    img = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j]:
                ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=8,
                        color="white" if rate[i, j] > 0.6 else ("#c1121f" if i != j else "#222"))
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Từ hệ thống đoán")
    ax.set_ylabel("Từ thật")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label="tỷ lệ theo hàng (từ thật)")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_by_speaker(acc, method, out_path):
    part = acc[(acc["method"] == method) & (acc["speaker"] != "TẤT CẢ")]
    table = part.pivot(index="speaker", columns="scenario", values="accuracy").reindex(columns=list(SCENARIOS))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(table))
    for offset, scenario, color in [(-0.2, "A", "#2a9d8f"), (0.2, "B", "#e76f51")]:
        values = table[scenario].fillna(0).to_numpy()
        bars = ax.bar(x + offset, values, 0.4, label=SCENARIOS[scenario], color=color)
        ax.bar_label(bars, [f"{v:.0%}" for v in values], fontsize=8)
    ax.set_xticks(x, table.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy theo từng người ({method})")
    ax.legend()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_compare(acc, timing, out_path):
    overall = acc[acc["speaker"] == "TẤT CẢ"].pivot(index="method", columns="scenario", values="accuracy")
    overall = overall.reindex(index=[m for m in METHOD_SLUGS if m in overall.index])
    fig, (ax, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(overall))
    for offset, scenario, color in [(-0.2, "A", "#2a9d8f"), (0.2, "B", "#e76f51")]:
        if scenario in overall:
            values = overall[scenario].to_numpy()
            bars = ax.bar(x + offset, values, 0.4, label=SCENARIOS[scenario], color=color)
            ax.bar_label(bars, [f"{v:.1%}" for v in values], fontsize=9)
    ax.set_xticks(x, overall.index)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy theo phương pháp và kịch bản")
    ax.legend(loc="upper center", ncol=2)

    times = [timing[m]["predict_ms"] for m in overall.index]
    bars = ax_time.barh(overall.index[::-1], times[::-1], color="#8d99ae")
    ax_time.bar_label(bars, [f"{t:.2f} ms" for t in times[::-1]], fontsize=9)
    ax_time.set_xscale("log")
    ax_time.set_xlabel("ms mỗi mẫu, đơn luồng (thang log)")
    ax_time.set_title("Thời gian nhận dạng một mẫu mới")
    ax_time.set_xlim(right=max(times) * 4)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def hard_pairs(preds):
    wrong = preds[~preds["correct"]]
    counts = Counter(zip(wrong["method"], wrong["scenario"], wrong["word"], wrong["predicted"]))
    rows = [{"method": m, "scenario": s, "word": w, "predicted": p, "count": c}
            for (m, s, w, p), c in counts.most_common()]
    return pd.DataFrame(rows, columns=["method", "scenario", "word", "predicted", "count"])


def run_sweep(bank, cfg, out_dir):
    """Quét k, trọng số F0, CMN. Trả về bảng accuracy theo kịch bản."""
    rows = []
    for cmn in SWEEP_CMN:
        for f0_weight in SWEEP_F0:
            started = time.perf_counter()
            folds = compute_folds(bank, cfg, f0_weight, cmn)
            for k in SWEEP_K:
                preds = predict_folds(bank, folds, k)
                acc = preds.groupby("scenario")["correct"].mean()
                rows.append({"cmn": cmn, "f0_weight": f0_weight, "k": k, "acc_A": acc["A"],
                             "acc_B": acc.get("B", np.nan)})
            print(f"  cmn={cmn!s:<5} f0_weight={f0_weight:<4} xong sau {time.perf_counter() - started:.0f} s")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "sweep.csv", index=False, float_format="%.4f", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, scenario in enumerate(["A", "B"]):
        for col, cmn in enumerate(SWEEP_CMN):
            table = sweep[sweep["cmn"] == cmn].pivot(index="f0_weight", columns="k", values=f"acc_{scenario}")
            ax = axes[row, col]
            img = ax.imshow(table.to_numpy(), cmap="viridis", vmin=sweep[f"acc_{scenario}"].min(),
                            vmax=sweep[f"acc_{scenario}"].max(), aspect="auto")
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    ax.text(j, i, f"{table.iat[i, j]:.1%}", ha="center", va="center", color="white", fontsize=9)
            ax.set_xticks(range(table.shape[1]), [f"k={c}" for c in table.columns])
            ax.set_yticks(range(table.shape[0]), [f"F0 x{v:g}" for v in table.index])
            ax.set_title(f"{SCENARIOS[scenario]}, CMN {'bật' if cmn else 'tắt'}")
            fig.colorbar(img, ax=ax)
    fig.suptitle("Quét tham số: accuracy theo k, trọng số F0 và CMN")
    fig.savefig(out_dir / "sweep.png", dpi=110)
    plt.close(fig)
    return sweep


def write_summary(out_dir, cfg, bank, acc, hard, timing, words, sweep):
    overall = acc[acc["speaker"] == "TẤT CẢ"].pivot(index="method", columns="scenario", values="accuracy")
    n_groups = bank.meta["speaker"].map(speaker_group).nunique()
    lines = [
        "# Kết quả đánh giá",
        "",
        f"Tự sinh bởi `python stt.py evaluate` lúc {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"Dữ liệu: {len(bank.meta)} mẫu, {bank.meta['word'].nunique()} từ, "
        f"{bank.meta['speaker'].nunique()} người, {n_groups} nhóm giọng cho kịch bản B.",
        "",
        f"Tham số: k = {cfg['knn']['k']}, trọng số F0 = {cfg['features']['f0_weight']:g}, "
        f"CMN = {'bật' if cfg['features'].get('cmn') else 'tắt'}, dải DTW {cfg['dtw']['band_ratio']:.0%}.",
        "",
        "## Accuracy",
        "",
        "| Phương pháp | " + " | ".join(SCENARIOS[s] for s in overall.columns)
        + " | Dự đoán mỗi mẫu | Huấn luyện mỗi vòng |",
        "|---|" + "---|" * len(overall.columns) + "---|---|",
    ]
    for method in [m for m in METHOD_SLUGS if m in overall.index]:
        row = overall.loc[method]
        cells = " | ".join(f"{row[s]:.1%}" for s in overall.columns)
        fit = timing[method]["fit_ms"]
        fit_text = f"{fit:.0f} ms" if fit is not None else "không cần"
        lines.append(f"| {method} | {cells} | {timing[method]['predict_ms']:.2f} ms | {fit_text} |")
    lines += ["", "Dự đoán mỗi mẫu: thời gian nhận dạng một mẫu mới, đơn luồng, trung vị trên 20 mẫu, "
              "chưa tính bước trích đặc trưng (giống nhau cho mọi phương pháp). "
              "Huấn luyện mỗi vòng: đơn luồng, trung bình trên mọi vòng; DTW không có bước huấn luyện.",
              "Đặc trưng của SVM và Random Forest: các cột số của `features_summary.csv`, "
              "bỏ `f0_hz_median` vì nó đặc trưng cho người nói chứ không cho từ.", "",
              "## Cặp từ hay nhầm nhất", "", "| Phương pháp | Kịch bản | Từ thật | Đoán thành | Số lần |",
              "|---|---|---|---|---|"]
    for row in hard.head(10).itertuples():
        lines.append(f"| {row.method} | {SCENARIOS[row.scenario]} | {words[row.word]} | "
                     f"{words[row.predicted]} | {row.count} |")
    if sweep is not None:
        best_b = sweep.sort_values(["acc_B", "acc_A"], ascending=False).iloc[0]
        lines += ["", "## Quét tham số", "",
                  f"Tốt nhất cho giọng mới: k = {int(best_b['k'])}, F0 x{best_b['f0_weight']:g}, "
                  f"CMN {'bật' if best_b['cmn'] else 'tắt'}, A = {best_b['acc_A']:.1%}, "
                  f"B = {best_b['acc_B']:.1%}. Chi tiết trong `sweep.csv`, `sweep.png`."]
    lines += ["", "## Tệp trong thư mục này", "",
              "- `accuracy.csv`: accuracy theo phương pháp, kịch bản, từng người.",
              "- `predictions.csv`: từng lần dự đoán.",
              "- `confusion_<phương pháp>_<A|B>.png` và `.csv`: ma trận nhầm lẫn.",
              "- `compare.png`: accuracy và thời gian dự đoán của các phương pháp.",
              "- `accuracy_by_speaker.png`: accuracy từng người, hai kịch bản, phương pháp DTW + F0.",
              "- `hard_words.csv`: các cặp từ bị nhầm, xếp theo số lần."]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dtw_single_query_ms(bank, cfg, f0_weight, cmn, n_queries=20, seed=0):
    """Thời gian nhận dạng một mẫu mới bằng DTW (một truy vấn chạy đơn luồng), trung vị trên n_queries mẫu."""
    recognizer = Recognizer(bank, cfg, FeatureTransform(cfg, f0_weight, cmn).fit(bank.sequences))
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(bank.meta), size=min(n_queries, len(bank.meta)), replace=False)
    recognizer.predict_sequence(bank.sequences[queries[0]])  # làm nóng numba
    times = [recognizer.predict_sequence(bank.sequences[i], exclude_files={bank.meta.at[i, "file_id"]}).elapsed_ms
             for i in queries]
    return float(np.median(times))


def print_scores(method, preds):
    print(f"  {method}: " + ", ".join(
        f"{SCENARIOS[s]} {preds[preds['scenario'] == s]['correct'].mean():.1%}"
        for s in preds["scenario"].unique()))


def run_evaluate(cfg, words, sweep=False, classifiers=True):
    bank = load_bank(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = cfg["paths"]["results"] / "evaluate" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    n_groups = bank.meta["speaker"].map(speaker_group).nunique()
    print(f"Kho: {len(bank.meta)} mẫu, {n_groups} nhóm giọng. Kết quả lưu vào {out_dir}")
    if n_groups < 2:
        print("Chỉ có một nhóm giọng nên không chạy được kịch bản B (giọng mới).")

    sweep_df = None
    if sweep:
        print("Quét tham số...")
        sweep_df = run_sweep(bank, cfg, out_dir)
        print(sweep_df.sort_values(["acc_B", "acc_A"], ascending=False).head(5).to_string(index=False))

    fcfg, k = cfg["features"], cfg["knn"]["k"]
    methods = {"DTW + F0": fcfg["f0_weight"], "DTW không F0": 0.0}
    all_preds, timing = [], {}
    for method, f0_weight in methods.items():
        cmn = fcfg.get("cmn", False)
        preds = predict_folds(bank, compute_folds(bank, cfg, f0_weight, cmn), k).assign(method=method)
        timing[method] = {"predict_ms": dtw_single_query_ms(bank, cfg, f0_weight, cmn), "fit_ms": None}
        all_preds.append(preds)
        print_scores(method, preds)

    if classifiers:
        X, _ = load_summary_matrix(cfg, bank.meta["file_id"])
        y = bank.meta["word"].to_numpy()
        splits = fold_indices(bank)
        for method in CLASSIFIERS:
            started = time.perf_counter()
            predicted, fit_ms = predict_classifier(method, X, y, splits)
            preds = classifier_predictions(bank, predicted).assign(method=method)
            timing[method] = {"predict_ms": single_query_ms(method, X, y), "fit_ms": fit_ms}
            all_preds.append(preds)
            print_scores(method, preds)
            print(f"    ({sum(len(f) for f in splits.values())} vòng, {time.perf_counter() - started:.0f} s)")

    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds[["method"] + [c for c in preds.columns if c != "method"]]
    acc = accuracy_table(preds)
    hard = hard_pairs(preds)
    preds.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    acc.to_csv(out_dir / "accuracy.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    hard.to_csv(out_dir / "hard_words.csv", index=False, encoding="utf-8-sig")

    codes = list(words)
    for (method, scenario), part in preds.groupby(["method", "scenario"], sort=False):
        slug = METHOD_SLUGS[method]
        matrix = confusion(part, codes)
        matrix.rename(index=words, columns=words).to_csv(out_dir / f"confusion_{slug}_{scenario}.csv",
                                                         encoding="utf-8-sig")
        plot_confusion(matrix, words, f"Ma trận nhầm lẫn: {method}, {SCENARIOS[scenario]} "
                       f"(accuracy {part['correct'].mean():.1%}, {len(part)} mẫu)",
                       out_dir / f"confusion_{slug}_{scenario}.png")
    plot_by_speaker(acc, "DTW + F0", out_dir / "accuracy_by_speaker.png")
    plot_compare(acc, timing, out_dir / "compare.png")
    write_summary(out_dir, cfg, bank, acc, hard, timing, words, sweep_df)
    print(f"Đã lưu bảng và hình vào {out_dir}")
    return out_dir
