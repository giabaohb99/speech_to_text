"""Điểm vào dòng lệnh của hệ thống nhận dạng từ đơn tiếng Việt.

Chạy `python stt.py --help` để xem các lệnh.
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def cmd_gen_data(args, cfg, words):
    from src.gen_data import generate

    generate(cfg, words, only_words=args.words)


def cmd_record(args, cfg, words):
    from src.record import record_session

    record_session(cfg, words, args.speaker, args.repeats, args.words, args.seconds)


def cmd_inventory(args, cfg, words):
    from src.inventory import run_inventory

    return run_inventory(cfg, words)


def cmd_build_features(args, cfg, words):
    from src.store import build_bank

    build_bank(cfg, words)


def cmd_plot(args, cfg, words):
    from pathlib import Path

    from src.plots import plot_files

    plot_files(cfg, words, [Path(p) for p in args.paths])


def cmd_recognize(args, cfg, words):
    from src.recognizer import run_recognize

    run_recognize(cfg, words, args.paths, args.exclude_speaker or (), args.k, not args.no_figure)


def cmd_evaluate(args, cfg, words):
    from src.evaluate import run_evaluate

    run_evaluate(cfg, words, sweep=args.sweep, classifiers=not args.dtw_only)


def cmd_package(args, cfg, words):
    from src.package import make_package

    make_package()


def cmd_demo(args, cfg, words):
    import subprocess
    import time
    import webbrowser
    from pathlib import Path

    app = Path(__file__).resolve().parent / "app.py"
    url = f"http://localhost:{args.port}"
    # headless: không hỏi email ở lần chạy đầu (lời hỏi đó chặn chờ nhập); tự mở trình duyệt bên dưới
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true",
                               "--server.port", str(args.port), "--browser.gatherUsageStats", "false"])
    print(f"Giao diện ở {url}  (Ctrl+C để dừng)", flush=True)
    time.sleep(3)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        return server.wait()
    except KeyboardInterrupt:
        server.terminate()
        return 0


def build_parser(cfg):
    parser = argparse.ArgumentParser(
        prog="stt", description="Nhận dạng từ đơn lẻ tiếng Việt (MFCC + F0 + DTW)"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="LỆNH")

    p = sub.add_parser("gen-data", help="Sinh dữ liệu mẫu bằng edge-tts vào data/raw")
    p.add_argument("--words", nargs="+", metavar="MA", help="chỉ sinh các từ này (mã không dấu)")
    p.set_defaults(handler=cmd_gen_data)

    p = sub.add_parser("record", help="Thu âm thật từ micro theo đúng quy ước tên file")
    p.add_argument("--speaker", required=True, help="mã người đọc, chữ thường không dấu, ví dụ an")
    p.add_argument("--repeats", type=int, default=cfg["record"]["repeats"], help="số lần đọc mỗi từ")
    p.add_argument("--seconds", type=float, default=cfg["record"]["seconds"], help="độ dài mỗi lần ghi")
    p.add_argument("--words", nargs="+", metavar="MA", help="chỉ thu các từ này")
    p.set_defaults(handler=cmd_record)

    p = sub.add_parser("inventory", help="Kiểm kê kho âm thanh: đếm theo từ, theo người, báo lỗi")
    p.set_defaults(handler=cmd_inventory)

    p = sub.add_parser("build-features", help="Trích đặc trưng toàn kho, ghi csv vào data/features")
    p.set_defaults(handler=cmd_build_features)

    p = sub.add_parser("plot", help="Vẽ hình minh họa đặc trưng vào results/figures")
    p.add_argument("paths", nargs="*", help="file wav cần vẽ; bỏ trống thì vẽ vài mẫu mặc định")
    p.set_defaults(handler=cmd_plot)

    p = sub.add_parser("recognize", help="Nhận dạng file wav, lưu hình kết quả vào results/recognize")
    p.add_argument("paths", nargs="+", metavar="FILE", help="một hoặc nhiều file wav")
    p.add_argument("--exclude-speaker", nargs="+", metavar="NGUOI",
                   help="loại người khỏi kho để giả lập giọng mới; 'tts_f1' loại cả nhóm tts_f1_p*")
    p.add_argument("--k", type=int, help=f"số láng giềng gần nhất (mặc định {cfg['knn']['k']})")
    p.add_argument("--no-figure", action="store_true", help="không lưu hình kết quả")
    p.set_defaults(handler=cmd_recognize)

    p = sub.add_parser("evaluate", help="Đánh giá hai kịch bản, lưu bảng và hình vào results/evaluate")
    p.add_argument("--sweep", action="store_true", help="quét thêm k, trọng số F0, CMN")
    p.add_argument("--dtw-only", action="store_true", help="bỏ SVM và Random Forest cho nhanh")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser("demo", help="Mở giao diện demo trên trình duyệt (Streamlit)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    p.set_defaults(handler=cmd_demo)

    p = sub.add_parser("package", help="Đóng gói dự án thành file zip trong dist/ để nộp")
    p.set_defaults(handler=cmd_package)
    return parser


def main():
    from src.common import load_config, load_words

    cfg = load_config()
    args = build_parser(cfg).parse_args()
    return args.handler(args, cfg, load_words()) or 0


if __name__ == "__main__":
    sys.exit(main())
