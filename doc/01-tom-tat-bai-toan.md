# 01. Tóm tắt bài toán

Đề tài 2: Xây dựng hệ thống chuyển đổi âm thanh thành văn bản (Speech-to-Text) ở mức từ đơn lẻ cho tiếng Việt.

Ngày lập: 2026-10-17. Trạng thái: bản nháp chờ chốt.

## 1. Phát biểu bài toán

Cho trước một **danh sách từ tiếng Việt đóng** (ví dụ 20 từ) và một **kho file âm thanh** đã thu, mỗi file là một người đọc một từ trong danh sách. Hệ thống nhận vào **một file âm thanh mới** chứa đúng một từ thuộc danh sách, do người đã có trong kho hoặc người hoàn toàn mới phát âm, và trả ra **chuỗi văn bản** là từ được nhận dạng.

Đây là bài toán **nhận dạng từ đơn lẻ** (isolated word recognition), không phải nhận dạng câu liên tục. Tập từ đóng, số người nói ít, số mẫu mỗi từ ít. Vì vậy cách tiếp cận phải chạy tốt với dữ liệu nhỏ và phải **giải thích được từng bước**, đúng yêu cầu của đề.

## 2. Đầu vào, đầu ra

| Thành phần | Mô tả |
|---|---|
| Đầu vào huấn luyện | Danh sách từ `words.txt`; các file `.wav` đặt tên theo quy ước `<từ>_<người>_<số thứ tự>.wav` |
| Đầu vào nhận dạng | Một file `.wav` mới, hoặc tín hiệu ghi trực tiếp từ micro, dài 0,3 đến 2,5 giây |
| Đầu ra | Từ được nhận dạng, kèm độ tin cậy và 3 ứng viên gần nhất |
| Sản phẩm phụ | Kho đặc trưng dạng `.csv` theo từ và theo người; báo cáo accuracy theo hai kịch bản |

## 3. Yêu cầu của đề và cách đáp ứng

| Mục đề | Yêu cầu | Cách hệ thống đáp ứng |
|---|---|---|
| 1 | Dùng lại bộ dữ liệu từ và file thu âm của nhóm | Thư mục `data/raw/` theo quy ước tên; script kiểm kê. Khi chưa có dữ liệu thật, dùng bộ sinh tự động làm dữ liệu mẫu (xem mục 6) |
| 2 | Tìm hiểu kỹ thuật nhận dạng giọng nói tiếng Việt, tập trung từ đơn | Tài liệu 05 phần A: so khớp mẫu bằng DTW, mô hình thống kê HMM-GMM, phân loại vector cố định, mạng nơ-ron, mô hình lớn (Whisper, wav2vec). Giải thích vì sao chọn DTW cho quy mô này |
| 3 | Xây dựng hệ thống nhận file mới, xuất văn bản | Lệnh `recognize` và giao diện demo |
| 4 | Sơ đồ khối và quy trình | Mục 4 tài liệu này và tài liệu kỹ thuật 03 |
| 5 | Trình bày các đặc trưng âm thanh và kỹ thuật trích | Năng lượng, ZCR, pitch F0, spectrogram, MFCC. Có hình minh họa sinh tự động từ dữ liệu |
| 6 | Lưu trữ đặc trưng dạng `.csv` theo từ và theo người; cách so khớp | Hai file csv: theo khung và theo mẫu. So khớp bằng DTW trên chuỗi khung, bỏ phiếu k láng giềng gần nhất |
| 7 | Demo, đánh giá accuracy riêng cho người đã có và giọng mới | Kiểm tra chéo hai kiểu: theo mẫu cho người đã có, theo người cho giọng mới (nhóm 3 thành viên là 3 tập kiểm tra ngược nhau); ma trận nhầm lẫn; bảng kết quả |

## 4. Sơ đồ khối tổng thể

```
  File .wav mới / micro
          |
          v
  [1] Tiền xử lý
      - đưa về 16 kHz, mono
      - chuẩn hóa biên độ
      - pre-emphasis
      - cắt khoảng lặng đầu cuối (năng lượng + ZCR)
          |
          v
  [2] Trích đặc trưng theo khung 25 ms, bước 10 ms
      - 13 MFCC + delta + delta-delta  (39 chiều)
      - F0 (pitch) cho thanh điệu
      - năng lượng, ZCR, spectral centroid (phục vụ trình bày)
          |
          v
  [3] Kho đặc trưng (.csv)                 <--- xây một lần từ data/raw/
      - features_frames.csv  (theo khung)
      - features_summary.csv (theo mẫu)
          |
          v
  [4] So khớp
      - DTW giữa chuỗi khung đầu vào và từng mẫu trong kho
      - lấy k mẫu gần nhất, bỏ phiếu theo từ
      - độ tin cậy = tỷ lệ phiếu và khoảng cách chuẩn hóa
          |
          v
  [5] Xuất văn bản: từ + độ tin cậy + top 3
```

Đường phụ (để so sánh trong báo cáo): từ `features_summary.csv` huấn luyện SVM hoặc Random Forest, chạy cùng bộ đánh giá.

## 5. Phạm vi

Trong phạm vi:

- Tập từ đóng, cố định trong `words.txt`. Gợi ý 20 từ: 10 chữ số (không, một, hai, ba, bốn, năm, sáu, bảy, tám, chín) và 10 lệnh (bật, tắt, mở, đóng, trái, phải, lên, xuống, dừng, chạy).
- Mỗi file chứa đúng một từ, có khoảng lặng ngắn hai đầu.
- Chạy offline trên máy cá nhân, không cần GPU.
- Đánh giá hai kịch bản: người đã có trong kho và người mới.

Ngoài phạm vi:

- Nhận dạng câu liên tục, tách từ trong câu.
- Từ ngoài danh sách (hệ thống chỉ báo độ tin cậy thấp, không từ chối tuyệt đối).
- Môi trường nhiễu mạnh, nhiều người nói cùng lúc.

## 6. Dữ liệu mẫu khi chưa có thu âm thật

Nhóm hiện chưa có bộ thu âm. Để có dữ liệu chạy ngay, hệ thống kèm bộ sinh dữ liệu mẫu:

1. Dùng dịch vụ tổng hợp tiếng nói tiếng Việt để đọc từng từ bằng vài giọng gốc khác nhau (nam, nữ).
2. Từ mỗi giọng gốc, tạo thêm các **biến thể giọng** bằng dịch cao độ, kéo giãn thời gian, thêm nhiễu nền nhẹ và thay đổi âm lượng. Mỗi biến thể được coi là một "người" riêng trong kho.
3. Kèm script `record.py` để thành viên nhóm **thu âm thật** theo đúng quy ước tên. Khi có dữ liệu thật, chỉ cần chép vào `data/raw/` và dựng lại kho đặc trưng.

Lưu ý ghi vào báo cáo: dữ liệu sinh tự động chỉ là **dữ liệu thay thế** để phát triển và kiểm thử pipeline. Kết quả đánh giá chính thức nên chạy trên thu âm thật của thành viên nhóm, vì đề bài yêu cầu rõ điều đó.

## 7. Tiêu chí hoàn thành

- Chạy một lệnh dựng được kho đặc trưng từ `data/raw/` và sinh hai file csv.
- Chạy một lệnh nhận dạng một file `.wav` mới và in ra từ.
- Chạy một lệnh đánh giá, in accuracy và ma trận nhầm lẫn cho hai kịch bản. Mỗi vòng đánh giá chỉ có hai tập train và test; tham số chốt trên dữ liệu sinh tự động, đo điểm cuối trên thu âm thật.
- Có giao diện demo ghi âm từ micro và hiện kết quả.
- Có hướng dẫn sử dụng đủ để người khác cài và chạy trong 15 phút.
- Accuracy mục tiêu trên dữ liệu mẫu: trên 90 phần trăm với người đã có, trên 70 phần trăm với người mới. Con số này sẽ hiệu chỉnh lại khi có thu âm thật.
- Thực đo 2026-10-17 trên dữ liệu sinh: người đã có 99,3 phần trăm, đạt; người mới 50,9 phần trăm, chưa đạt vì dữ liệu sinh chỉ có hai giọng gốc khác giới. Xem `results/README.md`.

## 8. Rủi ro chính

| Rủi ro | Ảnh hưởng | Cách giảm |
|---|---|---|
| Dữ liệu sinh tự động quá "sạch", khác giọng thật | Accuracy trên demo thật thấp hơn báo cáo | Thu âm thật sớm nhất có thể, ít nhất 3 người, mỗi từ 5 lần |
| Cắt khoảng lặng sai làm mất phụ âm đầu | Nhầm từ hàng loạt | Ngưỡng cắt thích nghi theo nhiễu nền, kiểm tra bằng hình vẽ |
| Từ cùng âm khác thanh (ma, má, mà) | DTW trên MFCC khó phân biệt | Thêm đường F0 vào vector đặc trưng với trọng số riêng |
| DTW chậm khi kho lớn | Demo trễ | Ràng buộc dải Sakoe-Chiba, giới hạn số mẫu mỗi từ, cache kho trong bộ nhớ |
