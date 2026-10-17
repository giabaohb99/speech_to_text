# Kết quả đánh giá

Tự sinh bởi `python stt.py evaluate` lúc 2026-10-17 15:22.

Dữ liệu: 1000 mẫu, 20 từ, 10 người, 2 nhóm giọng cho kịch bản B.

Tham số: k = 5, trọng số F0 = 2, CMN = tắt, dải DTW 10%.

## Accuracy

| Phương pháp | Người đã có | Giọng mới | Thời gian so khớp mỗi mẫu |
|---|---|---|---|
| DTW + F0 | 99.3% | 50.9% | 1.1 ms |
| DTW không F0 | 99.5% | 48.7% | 1.2 ms |

Thời gian tính trên máy chạy song song nhiều nhân, chia đều cho số mẫu test.

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

## Quét tham số

Tốt nhất cho giọng mới: k = 5, F0 x1, CMN tắt, A = 99.4%, B = 51.3%. Chi tiết trong `sweep.csv`, `sweep.png`.

## Tệp trong thư mục này

- `accuracy.csv`: accuracy theo phương pháp, kịch bản, từng người.
- `predictions.csv`: từng lần dự đoán.
- `confusion_<phương pháp>_<A|B>.png` và `.csv`: ma trận nhầm lẫn.
- `accuracy_by_speaker.png`: accuracy từng người, hai kịch bản.
- `hard_words.csv`: các cặp từ bị nhầm, xếp theo số lần.
