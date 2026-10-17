"""Đường phụ: SVM và Random Forest trên đặc trưng thống kê một dòng một mẫu (features_summary.csv).

Mỗi mẫu được nén thành một vector cố định (trung bình, độ lệch MFCC, độ dốc F0...), mất thứ tự
thời gian, khác với DTW so khớp cả chuỗi. Dùng để có bảng so sánh phương pháp trong báo cáo.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.store import CSV_ENCODING, SUMMARY_FILE

# Không đưa vào mô hình: khóa, chữ hiển thị, và cao độ trung vị (đặc trưng cho người nói, không cho từ)
NON_FEATURE_COLS = ["file_id", "word", "speaker", "sample_idx", "text", "f0_hz_median"]


def make_model(name):
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)
    raise ValueError(f"Không có mô hình {name}")


def load_summary_matrix(cfg, file_ids):
    """Đọc features_summary.csv, sắp đúng thứ tự file_ids. Trả về (X, tên cột)."""
    summary = pd.read_csv(cfg["paths"]["features"] / SUMMARY_FILE, encoding=CSV_ENCODING)
    summary = summary.set_index("file_id").reindex(file_ids)
    if summary.isna().any().any():
        raise SystemExit("features_summary.csv không khớp kho khung. Chạy lại 'python stt.py build-features'.")
    feature_cols = [c for c in summary.columns if c not in NON_FEATURE_COLS]
    return summary[feature_cols].to_numpy(dtype=np.float64), feature_cols


def _run_fold(name, X, y, test, train):
    model = make_model(name)
    started = time.perf_counter()
    model.fit(X[train], y[train])
    fitted = time.perf_counter()
    return test, model.predict(X[test]), fitted - started


def single_query_ms(name, X, y, n_queries=20, seed=0):
    """Thời gian nhận dạng một mẫu mới (đơn luồng), trung vị trên n_queries mẫu, mô hình học trên phần còn lại."""
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(y), size=min(n_queries, len(y)), replace=False)
    model = make_model(name).fit(np.delete(X, queries, axis=0), np.delete(y, queries))
    times = []
    for i in queries:
        started = time.perf_counter()
        model.predict(X[i:i + 1])
        times.append(time.perf_counter() - started)
    return 1000 * float(np.median(times))


def predict_classifier(name, X, y, splits):
    """Chạy mọi vòng song song trên nhiều nhân. splits: {kịch bản: [(chỉ số test, chỉ số train)]}.

    Trả về ({kịch bản: mảng từ dự đoán theo chỉ số mẫu}, thời gian huấn luyện trung bình mỗi vòng (ms)).
    """
    jobs = [(scenario, test, train) for scenario, folds in splits.items() for test, train in folds]
    results = Parallel(n_jobs=-1)(delayed(_run_fold)(name, X, y, test, train) for _, test, train in jobs)
    predictions = {scenario: np.empty(len(y), dtype=object) for scenario in splits}
    for (scenario, _, _), (test, predicted, _) in zip(jobs, results):
        predictions[scenario][test] = predicted
    return predictions, 1000 * float(np.mean([fit_s for _, _, fit_s in results]))
