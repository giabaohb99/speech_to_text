# 04. Đối chiếu 7 mục của đề bài

Ngày lập: 2026-10-17. Mỗi mục ghi yêu cầu của đề, sản phẩm đáp ứng, nơi xem, và phần còn phụ thuộc nhóm.

Ký hiệu: **Đạt** là đã có và đã kiểm; **Đạt có điều kiện** là đã có nhưng cần dữ liệu hoặc thao tác của nhóm để hoàn chỉnh.

| Mục | Yêu cầu tóm tắt | Trạng thái |
|---|---|---|
| 1 | Dùng bộ dữ liệu từ và file thu âm của nhóm | Đạt có điều kiện |
| 2 | Tìm hiểu kỹ thuật nhận dạng tiếng Việt, tập trung từ đơn | Đạt |
| 3 | Hệ thống nhận file mới, xuất từ văn bản | Đạt |
| 4 | Sơ đồ khối và quy trình | Đạt |
| 5 | Trình bày đặc trưng âm thanh và cách trích | Đạt |
| 6 | Lưu đặc trưng dạng csv theo từ và người, cách so khớp | Đạt |
| 7 | Demo, đánh giá accuracy riêng người đã có và giọng mới | Đạt có điều kiện |

## Mục 1. Bộ dữ liệu

**Đề yêu cầu:** dùng lại danh sách từ tiếng Việt và các file âm thanh nhóm đã thu theo giọng từng thành viên, làm tập huấn luyện và tham chiếu mức từ.

**Đã có:**

- Danh sách 20 từ: `words.txt` (10 chữ số, 10 lệnh điều khiển).
- Quy ước tên file `<từ>_<người>_<số>.wav` và kho `data/raw/`.
- Công cụ thu âm thật cho thành viên: `python stt.py record --speaker <mã>`.
- Công cụ kiểm kê kho: `python stt.py inventory`.
- Dữ liệu mẫu thay thế khi chưa có thu âm: 1000 file sinh bằng edge-tts, 2 giọng gốc x 5 biến thể x 20 từ x 5 lần. Nguồn gốc ghi ở `data/README.md`.

**Nhóm cần làm:** thu âm thật, ít nhất 3 thành viên, mỗi từ 5 lần, theo `HUONG-DAN.md` mục 6. Hiện mọi số liệu đều trên dữ liệu sinh tự động.

## Mục 2. Tìm hiểu kỹ thuật

**Đề yêu cầu:** tìm hiểu kỹ thuật xử lý và nhận dạng giọng nói tiếng Việt hiện hành, tập trung nhận dạng từ đơn thay vì câu liên tục, phù hợp quy mô dữ liệu.

**Đã có:** `doc/05-co-so-ly-thuyet.md` phần A.

- So sánh nhận dạng từ đơn và câu liên tục.
- Năm hướng kỹ thuật: DTW, HMM-GMM, phân loại trên vector cố định, mạng nơ-ron trên spectrogram, mô hình đầu cuối và mô hình nền lớn cho tiếng Việt.
- Đặc thù tiếng Việt: đơn âm tiết, sáu thanh, phương ngữ.
- Lý do chọn DTW + MFCC + F0 cho quy mô dữ liệu nhỏ.

## Mục 3. Hệ thống chuyển âm thanh thành văn bản

**Đề yêu cầu:** đầu vào là file âm thanh mới, phát âm một từ trong danh sách, do người đã thu hoặc người mới; đầu ra là từ văn bản.

**Đã có:**

- Dòng lệnh: `python stt.py recognize <file>` in từ, độ tin cậy, top 3. Nhận wav, flac, ogg, mp3, mọi tần số lấy mẫu và số kênh.
- Giao diện: `python stt.py demo`, ghi âm micro, tải file, hoặc chọn file mẫu.
- Giả lập người mới: `--exclude-speaker` trên dòng lệnh, ô "Giả lập giọng mới" trên giao diện.
- Mã nguồn: `src/recognizer.py`, `src/dtw.py`, `src/features.py`, `src/preprocess.py`.

**Đã kiểm:** 20 trên 20 file ngẫu nhiên đúng ở phase 3; giao diện chạy thật trên trình duyệt ở phase 6; 33 bài kiểm thử tự động đạt.

## Mục 4. Sơ đồ khối và quy trình

**Đề yêu cầu:** sơ đồ khối và quy trình từ nhận âm thanh đầu vào đến xuất văn bản.

**Đã có:**

- Hình sơ đồ khối: `results/figures/so-do-khoi.png`, vẽ lại bằng `python stt.py plot`. Hai luồng: dựng kho một lần và nhận dạng mỗi file mới, năm khối tiền xử lý, trích đặc trưng, kho đặc trưng, so khớp DTW, quyết định.
- Sơ đồ dạng chữ và mô tả từng khối: `doc/01-tom-tat-bai-toan.md` mục 4, `doc/03-tai-lieu-ky-thuat.md` mục 1.
- Hình minh họa quy trình trên một file cụ thể: `results/recognize/*.png`, gồm tín hiệu sau cắt lặng, MFCC, đường khớp DTW với mẫu gần nhất, khoảng cách tới các từ.

## Mục 5. Đặc trưng âm thanh

**Đề yêu cầu:** trình bày các thuộc tính âm thanh dùng để nhận dạng như năng lượng trung bình, ZCR, pitch, spectrogram, MFCC, và kỹ thuật trích rút.

**Đã có:**

- Trình bày lý thuyết, công thức, các bước trích: `doc/05-co-so-ly-thuyet.md` phần B, đủ năm đặc trưng đề nêu cộng delta MFCC và spectral centroid; bảng đặc trưng nào dùng vào việc gì.
- Hình minh họa 6 ô trên file thật: `results/figures/mot_*.png`, `nam_*.png`, `tam_*.png`, `sau_*.png`. Vẽ thêm cho file bất kỳ: `python stt.py plot <file>`.
- Mã trích: `src/preprocess.py` (năng lượng, ZCR, cắt lặng), `src/features.py` (MFCC, delta, F0, centroid).
- Kiểm chứng: độ dốc F0 tách được thanh sắc với thanh nặng, huyền (`doc/03-tai-lieu-ky-thuat.md` mục 6.3).

## Mục 6. Lưu trữ đặc trưng và so khớp

**Đề yêu cầu:** lưu và quản lý đặc trưng theo từ và theo người, ví dụ file csv; cách hệ thống so khớp và phân loại âm thanh đầu vào dựa trên các đặc trưng đó.

**Đã có:**

- `data/features/features_frames.csv`: một dòng một khung, cột `word`, `speaker`, 39 cột MFCC, F0, năng lượng, ZCR, centroid. Khoảng 44 nghìn dòng.
- `data/features/features_summary.csv`: một dòng một file, có cột từ có dấu `text`, thống kê từng đặc trưng. Mở bằng Excel, lọc theo từ hoặc người.
- Dựng lại: `python stt.py build-features`.
- Cách so khớp: `doc/05-co-so-ly-thuyet.md` phần C, chi tiết ở `doc/03-tai-lieu-ky-thuat.md` mục 7. DTW trên chuỗi khung, bỏ phiếu 5 láng giềng gần nhất.
- Hai cách phân loại so sánh dùng `features_summary.csv`: SVM, Random Forest.

## Mục 7. Demo và đánh giá

**Đề yêu cầu:** demo nhập file mới, xuất từ; đánh giá accuracy, riêng với người đã có trong tập huấn luyện và với giọng mới chưa xuất hiện.

**Đã có:**

- Demo: `python stt.py demo` và `python stt.py recognize`.
- Đánh giá: `python stt.py evaluate`. Chỉ hai tập train và test mỗi vòng, kiểm tra chéo:
  - Người đã có: giấu từng file một, train là phần còn lại.
  - Giọng mới: giấu toàn bộ một người, train là những người còn lại.
- Kết quả chính thức: `results/evaluate/20261017-153204/`, có accuracy tổng và từng người, ma trận nhầm lẫn từng phương pháp, hình so sánh, cặp từ hay nhầm. Nhận xét ở `results/README.md`.

| Phương pháp | Người đã có | Giọng mới |
|---|---|---|
| DTW + F0 | 99,3% | 50,9% |
| DTW không F0 | 99,5% | 48,7% |
| SVM | 99,1% | 44,0% |
| Random Forest | 98,7% | 46,8% |

**Nhóm cần làm:** chạy lại `evaluate` trên thu âm thật. Con số giọng mới hiện tại là cận dưới bi quan vì dữ liệu sinh chỉ có hai giọng gốc khác giới.

## Danh sách sản phẩm trong gói nộp

| Sản phẩm | Vị trí |
|---|---|
| Mã nguồn chạy được | `stt.py`, `app.py`, `src/`, `config.yaml`, `words.txt`, `requirements.txt` |
| Kiểm thử | `tests/`, `pytest.ini` |
| Dữ liệu mẫu | `data/raw/`, `data/tts_cache/`, `data/README.md` |
| Kho đặc trưng đã dựng | `data/features/` |
| Kết quả và hình | `results/` |
| Hướng dẫn sử dụng | `HUONG-DAN.md`, `README.md` |
| Tài liệu | `doc/01` tóm tắt bài toán, `doc/02` danh sách công việc, `doc/03` tài liệu kỹ thuật, `doc/04` đối chiếu đề, `doc/05` cơ sở lý thuyết |

Tạo gói: `python stt.py package`, xem `HUONG-DAN.md`.
