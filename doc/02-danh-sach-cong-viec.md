# 02. Danh sách công việc theo phase

Ngày lập: 2026-10-17. Mỗi phase có đầu ra rõ và điều kiện nghiệm thu. Thứ tự phase là thứ tự phụ thuộc, không đổi chỗ được. Cột "Quyết" ghi thứ cần người phụ trách chốt trước khi làm.

Ký hiệu trạng thái: [ ] chưa làm, [~] đang làm, [x] xong.

## Tổng quan

| Phase | Tên | Đầu ra chính | Ước lượng |
|---|---|---|---|
| 0 | Khởi tạo dự án | Cấu trúc thư mục, cấu hình, môi trường | 0,5 ngày |
| 1 | Dữ liệu mẫu | `data/raw/` có file wav đủ 20 từ, nhiều giọng; script thu âm thật | 1 ngày |
| 2 | Tiền xử lý và trích đặc trưng | Hai file csv, hình minh họa đặc trưng | 1,5 ngày |
| 3 | Nhận dạng bằng DTW | Lệnh `recognize` chạy được | 1 ngày |
| 4 | Đánh giá hai kịch bản | Bảng accuracy, ma trận nhầm lẫn | 1 ngày |
| 5 | Đường phụ SVM và so sánh | Bảng so sánh ba phương pháp | 1 ngày |
| 6 | Demo giao diện và hướng dẫn sử dụng | Ứng dụng Streamlit, `HUONG-DAN.md` | 1 ngày |
| 7 | Đóng gói nộp | Gói zip, checklist đối chiếu đề | 0,5 ngày |

Tổng ước lượng: 7,5 ngày làm việc nếu một người làm tuần tự. Phase 5 và 6 có thể làm song song sau khi phase 4 xong.

## Phase 0. Khởi tạo dự án

Mục tiêu: có khung mã nguồn chạy được lệnh rỗng, cài đặt một lần là xong.

- [x] Tạo cấu trúc thư mục theo tài liệu kỹ thuật mục 2.
- [x] Viết `requirements.txt` và kiểm tra cài sạch trên môi trường ảo. Đã chạy trên Python 3.13, librosa 1.0.
- [x] Viết `config.yaml` với toàn bộ tham số: tần số lấy mẫu, kích thước khung, số MFCC, k của k-NN, dải DTW.
- [x] Viết `words.txt` với 20 từ.
- [x] Viết điểm vào `stt.py` với các lệnh con: `gen-data`, `record`, `inventory` chạy thật; `build-features`, `recognize`, `evaluate`, `demo` báo chưa làm.

Nghiệm thu: `python stt.py --help` liệt kê đủ lệnh; `pip install -r requirements.txt` không lỗi. Đạt 2026-10-17.

Đã chốt: Python 3.11 trở lên, môi trường ảo `.venv` trong thư mục dự án.

## Phase 1. Dữ liệu mẫu

Mục tiêu: có kho âm thanh đủ để phát triển ngay, và có đường cho dữ liệu thật thay vào sau.

- [x] Viết `gen_data.py`: đọc `words.txt`, gọi edge-tts với 2 giọng gốc, lưu bản gốc vào `data/tts_cache/`, cắt lặng đệm sẵn, lưu `.wav` 16 kHz mono.
- [x] Viết bộ tăng cường: dịch cao độ (âm 3 đến dương 4 nửa cung), đổi độ dài (0,92 đến 1,08), thêm nhiễu trắng tỷ số tín hiệu trên nhiễu 20 đến 35 dB, đổi âm lượng. Mỗi giọng gốc sinh 5 biến thể, mỗi biến thể là một "người".
- [x] Số lượng: 20 từ, 10 người (2 giọng gốc x 5 biến thể), 5 mẫu mỗi từ mỗi người, tổng 1000 file, khoảng 24 MB, sinh trong khoảng 20 giây. Mẫu trong cùng một người khác nhau bằng nhiễu, lệch thời gian, âm lượng và lệch nhẹ cao độ, tốc độ.
- [x] Viết `record.py`: hiện từ cần đọc, nhấn Enter rồi ghi 2 giây từ micro, cảnh báo tiếng quá nhỏ, lưu đúng quy ước tên, cho nghe lại hoặc thu lại, số thứ tự nối tiếp file đã có.
- [x] Viết `inventory.py`: đếm số file theo từ và theo người, báo từ thiếu, báo file không đúng tên hoặc không đúng định dạng.
- [x] Lưu `data/README.md` ghi rõ nguồn dữ liệu, phần nào là sinh tự động.
- [x] Kiểm thử `tests/test_common.py`: quy ước tên file, đọc `words.txt`, số thứ tự thu âm.

Nghiệm thu: `python stt.py gen-data` tạo đủ kho; `python stt.py inventory` báo không thiếu từ. Đạt 2026-10-17. Riêng `record`: đã kiểm micro ghi được ở 16 kHz, nhưng chưa thử trọn một phiên đọc và nghe lại vì cần người đọc thật.

Đã chốt: edge-tts (cần mạng lần đầu, sau đó dùng bản lưu); giữ 20 từ như tài liệu kỹ thuật mục 3.3.

## Phase 2. Tiền xử lý và trích đặc trưng

Mục tiêu: biến mọi file wav thành chuỗi vector đặc trưng và lưu ra csv.

- [x] `preprocess.py`: nạp, đổi về 16 kHz mono, chuẩn hóa đỉnh, cắt khoảng lặng bằng ngưỡng năng lượng thích nghi kết hợp ZCR, đệm 50 ms hai đầu. Pre-emphasis chuyển sang ngay trước MFCC để F0 tính trên tín hiệu gốc.
- [x] `features.py`: 13 MFCC, delta, delta-delta; F0 bằng pYIN, nội suy khung vô thanh, lọc trung vị, trừ trung vị của file; năng lượng, ZCR, spectral centroid; hàm tóm tắt một mẫu.
- [x] `store.py`: ghi `features_frames.csv`, `features_summary.csv`; đọc lại thành kho trong bộ nhớ (`load_bank`) cho phase 3. Phase 4 bỏ `features_stats.csv`, thống kê chuẩn hóa tính theo tập train của từng vòng.
- [x] `plots.py` và lệnh `plot`: hình 6 ô dạng sóng, năng lượng, ZCR, F0, spectrogram, MFCC ra png trong `results/figures/`.
- [x] Kiểm thử `tests/test_features.py`: cắt lặng mất không quá 20 ms tiếng, giữ phụ âm xát, file toàn lặng không lỗi, số khung đúng, F0 đúng, dựng và đọc lại kho khớp.
- [x] Sửa thêm: hạt giống sinh dữ liệu theo từng cặp từ-giọng để sinh lẻ một từ ra file giống hệt sinh cả kho.

Nghiệm thu: `python stt.py build-features` chạy hết 1000 file trong khoảng 56 giây; ba csv ghi `utf-8-sig` cho Excel; ảnh minh họa đủ 6 loại. Đạt 2026-10-17. Chưa mở thử csv bằng Excel thật.

Đã chốt: F0 nằm trong vector so khớp, bật tắt bằng `f0_weight` trong cấu hình, mặc định 2,0.

## Phase 3. Nhận dạng bằng DTW

Mục tiêu: nhận một file mới và trả ra từ.

- [x] `dtw.py`: khoảng cách DTW với dải Sakoe-Chiba nghiêng, chia cho tổng độ dài, viết bằng numba; có `dtw_path` để vẽ; kiểm thử so với cài đặt tham chiếu.
- [x] `recognizer.py`: nạp kho từ csv, chuẩn hóa đặc trưng đầu vào theo thống kê kho, nhân trọng số F0, tính khoảng cách tới mọi mẫu, lấy k gần nhất, bỏ phiếu theo từ, tính độ tin cậy.
- [x] Lệnh `recognize FILE...`: in từ, độ tin cậy, top 3, mẫu gần nhất, thời gian; tự loại chính file nếu nó đã có trong kho.
- [x] Tùy chọn `--exclude-speaker` để giả lập người mới khi thử tay, nhận cả tên nhóm như `tts_f1`.
- [x] Lưu hình kết quả mỗi lần thử vào `results/recognize/` và ghi `log.csv` (đại ca yêu cầu thêm 2026-10-17).
- [x] Đo thời gian: phần so khớp 17 đến 60 ms với kho 1000 mẫu, không cần tối ưu.
- [x] Kiểm thử `tests/test_dtw.py`: 9 bài cho DTW và bộ nhận dạng.

Nghiệm thu 2026-10-17, 20 file chọn ngẫu nhiên (hạt giống 7), đã lưu 40 hình trong `results/recognize/`:

| Lượt | Kho dùng để so khớp | Đúng |
|---|---|---|
| Người đã có | cả kho trừ chính file đó | 20/20 |
| Giọng mới, thử sớm | loại cả nhóm giọng gốc của file, chỉ còn một giọng khác giới | 9/20 |

Đạt nghiệm thu phase 3. Lượt giọng mới chỉ là tín hiệu sớm cho phase 4: kho sinh tự động chỉ có hai giọng gốc, nên khi loại một nhóm thì chỉ còn một giọng khác giới để so.

Đã chốt: k = 5, bán kính dải 10 phần trăm độ dài, tối thiểu 5 khung.

## Phase 4. Đánh giá hai kịch bản

Mục tiêu: có số liệu đúng yêu cầu mục 7 của đề.

- [x] `evaluate.py` kịch bản A, người đã có: kiểm tra chéo theo mẫu, bỏ lần lượt từng mẫu ra làm test, train là phần còn lại.
- [x] Kịch bản B, giọng mới: kiểm tra chéo theo người, bỏ lần lượt từng người ra làm test, train là các người còn lại. Trên dữ liệu sinh, một "người" là cả nhóm biến thể của một giọng gốc. Thống kê chuẩn hóa học lại trên train của từng vòng để không rò rỉ.
- [x] Chỉ có hai tập train và test, không có tập kiểm định. Tham số chốt trên dữ liệu sinh bằng lượt quét `--sweep` (k, trọng số F0, CMN); ghi vào `results/README.md`.
- [x] Mỗi lần chạy lưu vào `results/evaluate/<thời điểm>/`: `accuracy.csv`, `predictions.csv`, ma trận nhầm lẫn png và csv cho từng phương pháp và kịch bản, `accuracy_by_speaker.png`, `hard_words.csv`, `README.md` tự sinh; có `--sweep` thì thêm `sweep.csv`, `sweep.png`.
- [x] Chạy thêm với F0 tắt để có số liệu cho bàn luận về thanh điệu.
- [x] Viết `results/README.md` tóm tắt số liệu và nhận xét.
- [x] Thêm tùy chọn CMN (trừ trung bình MFCC theo từng file) để thử cải thiện giọng mới; mặc định tắt vì không giúp trên dữ liệu sinh.
- [x] Kiểm thử `tests/test_evaluate.py`: nhóm giọng, không tự khớp và không khớp người cùng nhóm ở kịch bản B, bảng accuracy đủ dòng.

Nghiệm thu: lệnh `python stt.py evaluate --sweep` chạy hết trong 42 giây, không can thiệp tay; đủ bảng và hình. Đạt 2026-10-17, lần chạy chính thức ở `results/evaluate/20261017-152220/`.

| Phương pháp | Người đã có | Giọng mới |
|---|---|---|
| DTW + F0 | 99,3% | 50,9% |
| DTW không F0 | 99,5% | 48,7% |

Chưa đạt mục tiêu 70% cho giọng mới ghi ở tài liệu 01 mục 7. Nguyên nhân: dữ liệu sinh chỉ có hai giọng gốc khác giới, nên mỗi vòng kịch bản B chỉ còn một giọng tham chiếu. Nhận xét chi tiết ở `results/README.md`. Cần đo lại trên thu âm thật.

Đã chốt: k = 5, trọng số F0 = 2, CMN tắt, dải DTW 10 phần trăm. Lượt quét cho thấy các phương án khác không hơn đáng kể.

## Phase 5. Đường phụ SVM và so sánh

Mục tiêu: có phương pháp thứ hai để so sánh trong báo cáo.

- [x] `classifier.py`: đọc `features_summary.csv` (bỏ cột cao độ trung vị vì đặc trưng cho người nói), chuẩn hóa, huấn luyện SVM nhân RBF và Random Forest bằng scikit-learn.
- [x] Gắn vào `evaluate.py`, chạy trên đúng các vòng chia của DTW, kể cả 1000 vòng bỏ-một-mẫu ở kịch bản A. Các vòng chạy song song trên mọi nhân, mỗi mô hình đơn luồng.
- [x] Bảng so sánh: DTW có F0, DTW không F0, SVM, Random Forest, theo hai kịch bản, thời gian nhận dạng một mẫu và thời gian huấn luyện. Thời gian đo cùng một cách: đơn luồng, trung vị trên 20 mẫu.
- [x] Lưu hình: `compare.png` và ma trận nhầm lẫn của từng phương pháp, từng kịch bản.
- [x] Tùy chọn `evaluate --dtw-only` bỏ hai mô hình cho nhanh.
- [x] Kiểm thử thêm trong `tests/test_evaluate.py`: các vòng chia phủ mỗi mẫu đúng một lần, hai mô hình chạy đúng trên các vòng đó, cột cao độ bị loại.
- [ ] Nếu dữ liệu thật đạt 20 mẫu mỗi từ trở lên, cân nhắc thêm CNN nhỏ trên mel-spectrogram. Mặc định không làm.

Nghiệm thu: bảng so sánh đầy đủ trong `results/evaluate/20261017-153204/`, nhận xét trong `results/README.md`. Đạt 2026-10-17. Lệnh `evaluate` đầy đủ chạy khoảng 2,5 phút, phần lớn là Random Forest.

| Phương pháp | Người đã có | Giọng mới | Nhận dạng một mẫu |
|---|---|---|---|
| DTW + F0 | 99,3% | 50,9% | 27 ms |
| DTW không F0 | 99,5% | 48,7% | 32 ms |
| SVM | 99,1% | 44,0% | 0,3 ms |
| Random Forest | 98,7% | 46,8% | 13 ms |

Kết luận: DTW + F0 là phương pháp chính, tốt nhất ở kịch bản giọng mới.

Quyết: có làm CNN hay không, quyết sau khi biết lượng dữ liệu thật.

## Phase 6. Demo giao diện và hướng dẫn sử dụng

Mục tiêu: người chấm thao tác được mà không đọc mã.

- [x] `app.py` bằng Streamlit: ba nguồn âm thanh (ghi âm micro, tải file lên, chọn file mẫu trong kho), hiện từ nhận dạng, độ tin cậy, top 3, láng giềng, hình kết quả 4 ô gồm dạng sóng, MFCC, đường khớp DTW, khoảng cách từng từ.
- [x] Thanh bên: thông tin kho, chỉnh k và trọng số F0, chế độ "giả lập người mới" loại người hoặc cả nhóm giọng khỏi kho.
- [x] Lưu mỗi lần thử trên giao diện vào `results/demo/`: hình, `log.csv`, và file âm thanh đầu vào nếu ghi âm hoặc tải lên.
- [x] Lệnh `demo` chạy Streamlit ở chế độ không hỏi email lần đầu (lời hỏi đó chặn chờ nhập), tự mở trình duyệt.
- [x] Tách hàm `recognize_file` dùng chung cho lệnh `recognize` và giao diện.
- [x] Kiểm thử `tests/test_app.py` bằng Streamlit AppTest: chọn file mẫu nhận đúng, lưu hình và log; loại nhóm giọng thì láng giềng không còn ai của nhóm đó.
- [x] `HUONG-DAN.md`: cài đặt, chạy nhanh, các lệnh, giao diện, quy trình thu âm thật của nhóm, đổi danh sách từ, đọc kết quả, tham số, lỗi thường gặp.
- [ ] Thử theo hướng dẫn trên một máy sạch.

Nghiệm thu 2026-10-17: đã chạy giao diện thật trên trình duyệt, chọn file mẫu chữ chạy của giọng nữ, loại cả nhóm giọng nữ, nhận đúng chạy với độ tin cậy 80%, hình lưu ở `results/demo/`. File 44,1 kHz hai kênh được tự đổi định dạng và nhận đúng. Chưa thử: ghi âm micro và tải file qua trình duyệt (cần người thao tác), và cài trên máy sạch trong 15 phút.

Đã chốt: Streamlit.

## Phase 7. Đóng gói nộp

- [x] Checklist đối chiếu 7 mục của đề với vị trí trong sản phẩm: `doc/04-doi-chieu-de-bai.md`.
- [x] Bổ sung phần trình bày còn thiếu khi đối chiếu: `doc/05-co-so-ly-thuyet.md` cho mục 2 (tìm hiểu kỹ thuật), mục 5 (đặc trưng và cách trích), mục 6 (lưu trữ và so khớp); hình sơ đồ khối `results/figures/so-do-khoi.png` cho mục 4, vẽ lại bằng lệnh `plot`.
- [x] Dữ liệu: giữ nguyên kho mẫu 1000 file (24 MB) và csv đã dựng để người chấm không phải chạy lại; giữ toàn bộ hình đã thử theo yêu cầu.
- [x] Lệnh `package` tạo `dist/speech-to-text_<thời điểm>.zip`, gồm mã nguồn, dữ liệu mẫu, csv, kết quả, hướng dẫn, năm tài liệu; bỏ `.venv` và thư mục đệm. Gói khoảng 36 MB.
- [x] Ghi rõ phần nào dùng dữ liệu sinh tự động: `data/README.md`, `results/README.md`, `doc/04` mục 1 và 7. Hiện toàn bộ số liệu là trên dữ liệu sinh.

Nghiệm thu 2026-10-17: giải nén gói vào thư mục tạm trên cùng máy, tạo môi trường ảo mới, cài `requirements.txt` mất 3 phút, rồi chạy theo hướng dẫn: `--help`, 33 bài kiểm thử đạt, `inventory`, `recognize`, `evaluate --dtw-only` ra đúng số liệu như bản gốc, `gen-data` sinh lại file giống từng byte. Chưa thử trên máy khác hẳn hoặc hệ điều hành khác.

## Việc phụ thuộc vào nhóm, không nằm trong mã

- Thu âm thật: tối thiểu 3 thành viên, mỗi người đọc 20 từ, mỗi từ 5 lần, trong phòng yên tĩnh, cùng một micro nếu được. Nên làm ngay sau phase 1 khi đã có `record.py`.
- Sau khi thu âm: chạy lại `build-features`, `evaluate`, cập nhật `results/README.md`, rồi `package`.
- Viết báo cáo học thuật và slide: dựa trên tài liệu 01, 04, 05 cộng số liệu và hình trong `results/`.
