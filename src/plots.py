"""Hình minh họa đặc trưng cho báo cáo: dạng sóng, năng lượng, ZCR, F0, spectrogram, MFCC."""
import matplotlib

matplotlib.use("Agg")
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common import parse_filename  # noqa: E402
from src.features import MFCC_COLS, extract  # noqa: E402
from src.preprocess import load_audio, normalize, trim_silence  # noqa: E402

DEFAULT_SAMPLES = ["mot_tts_f1_p0_01", "nam_tts_f1_p0_01", "tam_tts_f1_p0_01", "sau_tts_m1_p0_01"]


def plot_file(path, cfg, words, out_path):
    sr = cfg["audio"]["sample_rate"]
    y_full = normalize(load_audio(path, sr))
    y, cut = trim_silence(y_full, sr, cfg["audio"], cfg["preprocess"])
    frames, info = extract(y, sr, cfg)
    hop = cut["hop"]

    parsed = parse_filename(path.stem)
    title = f"{path.stem}" + (f"  ({words.get(parsed[0], parsed[0])})" if parsed else "")
    fig, axes = plt.subplots(6, 1, figsize=(10, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    t_full = np.arange(len(y_full)) / sr
    axes[0].plot(t_full, y_full, lw=0.6, color="#555")
    axes[0].axvspan(cut["start"] / sr, cut["end"] / sr, color="#2a9d8f", alpha=0.15, label="phần giữ lại")
    axes[0].set_title("Dạng sóng và vùng giữ lại sau cắt lặng")
    axes[0].legend(loc="upper right")

    t_frames = np.arange(len(cut["energy_db"])) * hop / sr
    axes[1].plot(t_frames, cut["energy_db"], color="#264653")
    axes[1].axhline(cut["threshold_db"], ls="--", color="#e76f51", label="ngưỡng cắt lặng")
    axes[1].set_title("Năng lượng ngắn hạn (dB)")
    axes[1].legend(loc="upper right")

    axes[2].plot(t_frames, cut["zcr"], color="#264653")
    axes[2].set_title("Zero Crossing Rate")

    t_kept = np.arange(len(frames)) * hop / sr
    axes[3].plot(t_kept, info["f0_hz"], "o-", ms=3, color="#264653")
    axes[3].set_title(f"Pitch F0 (Hz) trên phần giữ lại, trung vị {np.nanmedian(info['f0_hz']):.0f} Hz"
                      if np.isfinite(info["f0_hz"]).any() else "Pitch F0: không có khung hữu thanh")
    axes[3].set_xlim(0, len(y) / sr)

    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=cfg["features"]["n_fft"], hop_length=hop)),
                                      ref=np.max)
    img = librosa.display.specshow(spec_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=axes[4])
    fig.colorbar(img, ax=axes[4], format="%+.0f dB")
    axes[4].set_title("Spectrogram")
    axes[4].set_ylabel("Tần số (Hz)")

    mfcc = frames[MFCC_COLS].to_numpy()
    mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)  # chuẩn hóa từng hệ số để dễ nhìn
    img = axes[5].imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis",
                         extent=[0, len(y) / sr, 0.5, len(MFCC_COLS) + 0.5])
    fig.colorbar(img, ax=axes[5])
    axes[5].set_title("MFCC (13 hệ số, chuẩn hóa từng hệ số để hiển thị)")
    axes[5].set_ylabel("Hệ số")
    axes[5].set_xlabel("Thời gian (s)")

    for ax in axes[:5]:
        ax.set_xlabel("Thời gian (s)")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_recognition(path, cfg, words, recognizer, frames, pred, out_path):
    """Hình kết quả một lần nhận dạng: tín hiệu, MFCC, đường khớp DTW với mẫu gần nhất, khoảng cách từng từ."""
    from src.dtw import dtw_path
    from src.features import DTW_COLS
    from src.preprocess import load_and_clean

    sr = cfg["audio"]["sample_rate"]
    y, _ = load_and_clean(path, cfg)
    query = recognizer.normalize(frames[DTW_COLS].to_numpy(dtype=np.float32))
    nearest_id, nearest_word, nearest_dist = pred.neighbors[0]
    reference = recognizer.sequence_of(nearest_id)
    dcfg = cfg["dtw"]
    path_pairs = np.array(dtw_path(query, reference, dcfg["band_ratio"], dcfg["band_min_frames"]))
    local = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))

    text = words.get(pred.word, pred.word)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"{path.name}  ->  nhận dạng: {text}  (độ tin cậy {pred.confidence:.2f})", fontsize=13)

    ax = axes[0, 0]
    ax.plot(np.arange(len(y)) / sr, y, lw=0.6, color="#555")
    ax.set_title("Tín hiệu đầu vào sau cắt lặng")
    ax.set_xlabel("Thời gian (s)")

    ax = axes[0, 1]
    mfcc = frames[MFCC_COLS].to_numpy()
    mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)
    img = ax.imshow(mfcc.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, len(y) / sr, 0.5, len(MFCC_COLS) + 0.5])
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC đầu vào (chuẩn hóa từng hệ số để hiển thị)")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Hệ số")

    ax = axes[1, 0]
    img = ax.imshow(local.T, aspect="auto", origin="lower", cmap="magma_r")
    ax.plot(path_pairs[:, 0], path_pairs[:, 1], color="#2a9d8f", lw=2, label="đường khớp DTW")
    fig.colorbar(img, ax=ax, label="khoảng cách khung")
    ax.set_title(f"Khớp DTW với mẫu gần nhất: {nearest_id} ({words.get(nearest_word, nearest_word)}), "
                 f"khoảng cách {nearest_dist:.3f}", fontsize=10)
    ax.set_xlabel("Khung đầu vào")
    ax.set_ylabel("Khung mẫu trong kho")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ranked = sorted(pred.word_distances.items(), key=lambda item: item[1])[:10]
    labels = [words.get(w, w) for w, _ in ranked][::-1]
    values = [d for _, d in ranked][::-1]
    colors = ["#e76f51" if w == pred.word else "#8d99ae" for w, _ in ranked][::-1]
    ax.barh(labels, values, color=colors)
    ax.set_title("Khoảng cách DTW nhỏ nhất của 10 từ gần nhất (càng ngắn càng giống)")
    neighbors = ", ".join(words.get(w, w) for _, w, _ in pred.neighbors)
    ax.set_xlabel(f"Khoảng cách DTW\n{len(pred.neighbors)} láng giềng gần nhất bỏ phiếu: {neighbors}")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_block_diagram(out_path):
    """Sơ đồ khối hệ thống: luồng dựng kho (trên) và luồng nhận dạng (dưới)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="#264653", lw=1.4))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11, weight="bold")
        ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top", fontsize=8.6, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                     lw=1.5, color="#264653"))
        if label:
            ax.text((x1 + x2) / 2 + (0.1 if x1 == x2 else 0), (y1 + y2) / 2 + (0 if x1 == x2 else 0.2), label,
                    ha="left" if x1 == x2 else "center", va="center", fontsize=8, color="#555")

    data, step, store, match, out = "#e9f5f3", "#fdf0e6", "#eef0f7", "#fbe4e0", "#e6f2e6"
    ax.text(0.2, 6.3, "Luồng dựng kho (chạy một lần): build-features", fontsize=10, color="#555")
    ax.text(1.3, 3.05, "Luồng nhận dạng (mỗi file mới):\nrecognize / demo", fontsize=9.5, color="#555",
            ha="center", va="center")

    box(0.2, 3.6, 2.2, 2.4, "Kho âm thanh", "data/raw/*.wav\n<từ>_<người>_<số>.wav\n20 từ x nhiều người\nx 5 lần đọc", data)
    box(0.2, 0.2, 2.2, 2.3, "Âm thanh mới", "file wav / mp3\nhoặc ghi âm micro\nmột từ trong danh sách", data)
    box(3.0, 0.2, 2.6, 5.8, "[1] Tiền xử lý",
        "đổi về 16 kHz, 1 kênh\ntrừ thành phần một chiều\nchuẩn hóa đỉnh 0,95\n\ncắt lặng hai đầu:\nnăng lượng ngắn hạn\n> nền + 6 dB\nhoặc ZCR cao\n(giữ phụ âm xát)\nnới 50 ms mỗi đầu",
        step)
    box(6.2, 0.2, 2.7, 5.8, "[2] Trích đặc trưng",
        "khung 25 ms, bước 10 ms\n\npre-emphasis 0,97\n13 MFCC (40 bộ lọc Mel)\n+ delta + delta-delta\n\nF0 bằng pYIN\nlog2, lọc trung vị,\ntrừ trung vị file\n\nnăng lượng, ZCR,\nspectral centroid",
        step)
    box(9.5, 3.6, 2.5, 2.4, "[3] Kho đặc trưng",
        "features_frames.csv\n(một dòng một khung)\nfeatures_summary.csv\n(một dòng một file)", store)
    box(9.5, 0.2, 2.5, 2.3, "[4] So khớp DTW",
        "chuẩn hóa từng chiều\nF0 x 2\nDTW dải Sakoe-Chiba 10%\ntới mọi mẫu trong kho", match)
    box(12.6, 0.2, 2.2, 2.3, "[5] Quyết định",
        "5 láng giềng gần nhất\nbỏ phiếu 1/khoảng cách\n\n=> từ văn bản\n+ độ tin cậy + top 3", out)

    arrow(2.4, 4.8, 3.0, 4.8)
    arrow(2.4, 1.35, 3.0, 1.35)
    arrow(5.6, 3.1, 6.2, 3.1)
    arrow(8.9, 4.8, 9.5, 4.8, "ghi csv")
    arrow(8.9, 1.35, 9.5, 1.35, "chuỗi khung")
    arrow(10.75, 3.6, 10.75, 2.5, "nạp kho")
    arrow(12.0, 1.35, 12.6, 1.35)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_files(cfg, words, paths=None):
    raw_dir = cfg["paths"]["raw"]
    out_dir = cfg["paths"]["results"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not paths:
        paths = [raw_dir / f"{stem}.wav" for stem in DEFAULT_SAMPLES if (raw_dir / f"{stem}.wav").exists()]
        print("Đã vẽ", plot_block_diagram(out_dir / "so-do-khoi.png"))
    for path in paths:
        out = plot_file(path, cfg, words, out_dir / f"{path.stem}.png")
        print("Đã vẽ", out)
