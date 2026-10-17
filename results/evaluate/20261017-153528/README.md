# Kết quả đánh giá

Tự sinh bởi `python stt.py evaluate` lúc 2026-10-17 15:35.

Dữ liệu: 1000 mẫu, 20 từ, 10 người, 2 nhóm giọng cho kịch bản B.

Tham số: k = 5, trọng số F0 = 2, CMN = tắt, dải DTW 10%.

## Accuracy

| Phương pháp | Người đã có | Giọng mới | Dự đoán mỗi mẫu | Huấn luyện mỗi vòng |
|---|---|---|---|---|
| DTW + F0 | 99.3% | 50.9% | 33.74 ms | không cần |
| DTW không F0 | 99.5% | 48.7% | 25.92 ms | không cần |

Dự đoán mỗi mẫu: thời gian nhận dạng một mẫu mới, đơn luồng, trung vị trên 20 mẫu, chưa tính bước trích đặc trưng (giống nhau cho mọi phương pháp). Huấn luyện mỗi vòng: đơn luồng, trung bình trên mọi vòng; DTW không có bước huấn luyện.
Đặc trưng của SVM và Random Forest: các cột số của `features_summary.csv`, bỏ `f0_hz_median` vì nó đặc trưng cho người nói chứ không cho từ.

## Cặp từ hay nhầm nhất

| Phương pháp | Kịch bản | Từ thật | Đoán thành | Số lần |
|---|---|---|---|---|
| DTW + F0 | Giọng mới | bảy | chạy | 21 |
| DTW + F0 | Giọng mới | trái | ba | 21 |
| DTW không F0 | Giọng mới | xuống | bốn | 21 |
| DTW không F0 | Giọng mới | trái | ba | 21 |
| DTW + F0 | Giọng mới | năm | chạy | 20 |
| DTW không F0 | Giọng mới | năm | chạy | 20 |
| DTW + F0 | Giọng mới | chạy | lên | 19 |
| DTW + F0 | Giọng mới | lên | chạy | 18 |
| DTW + F0 | Giọng mới | tám | đóng | 18 |
| DTW + F0 | Giọng mới | xuống | bốn | 18 |

## Tệp trong thư mục này

- `accuracy.csv`: accuracy theo phương pháp, kịch bản, từng người.
- `predictions.csv`: từng lần dự đoán.
- `confusion_<phương pháp>_<A|B>.png` và `.csv`: ma trận nhầm lẫn.
- `compare.png`: accuracy và thời gian dự đoán của các phương pháp.
- `accuracy_by_speaker.png`: accuracy từng người, hai kịch bản, phương pháp DTW + F0.
- `hard_words.csv`: các cặp từ bị nhầm, xếp theo số lần.
