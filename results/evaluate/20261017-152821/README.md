# Kết quả đánh giá

Tự sinh bởi `python stt.py evaluate` lúc 2026-10-17 15:30.

Dữ liệu: 1000 mẫu, 20 từ, 10 người, 2 nhóm giọng cho kịch bản B.

Tham số: k = 5, trọng số F0 = 2, CMN = tắt, dải DTW 10%.

## Accuracy

| Phương pháp | Người đã có | Giọng mới | Dự đoán mỗi mẫu | Huấn luyện mỗi vòng |
|---|---|---|---|---|
| DTW + F0 | 99.3% | 50.9% | 1.56 ms | không cần |
| DTW không F0 | 99.5% | 48.7% | 1.27 ms | không cần |
| SVM | 99.1% | 44.0% | 0.62 ms | 142 ms |
| Random Forest | 98.7% | 46.8% | 15.95 ms | 2471 ms |

DTW: thời gian tính cả ma trận khoảng cách song song nhiều nhân, chia đều cho số mẫu test. SVM, Random Forest: thời gian dự đoán đơn luồng mỗi mẫu; huấn luyện đơn luồng mỗi vòng.
Đặc trưng của SVM và Random Forest: các cột số của `features_summary.csv`, bỏ `f0_hz_median` vì nó đặc trưng cho người nói chứ không cho từ.

## Cặp từ hay nhầm nhất

| Phương pháp | Kịch bản | Từ thật | Đoán thành | Số lần |
|---|---|---|---|---|
| SVM | Giọng mới | trái | bật | 24 |
| DTW + F0 | Giọng mới | bảy | chạy | 21 |
| DTW + F0 | Giọng mới | trái | ba | 21 |
| DTW không F0 | Giọng mới | xuống | bốn | 21 |
| DTW không F0 | Giọng mới | trái | ba | 21 |
| DTW + F0 | Giọng mới | năm | chạy | 20 |
| DTW không F0 | Giọng mới | năm | chạy | 20 |
| DTW + F0 | Giọng mới | chạy | lên | 19 |
| DTW + F0 | Giọng mới | lên | chạy | 18 |
| DTW + F0 | Giọng mới | tám | đóng | 18 |

## Tệp trong thư mục này

- `accuracy.csv`: accuracy theo phương pháp, kịch bản, từng người.
- `predictions.csv`: từng lần dự đoán.
- `confusion_<phương pháp>_<A|B>.png` và `.csv`: ma trận nhầm lẫn.
- `compare.png`: accuracy và thời gian dự đoán của các phương pháp.
- `accuracy_by_speaker.png`: accuracy từng người, hai kịch bản, phương pháp DTW + F0.
- `hard_words.csv`: các cặp từ bị nhầm, xếp theo số lần.
