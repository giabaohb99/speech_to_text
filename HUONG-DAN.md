# Hướng dẫn sử dụng

Hệ thống nhận dạng từ đơn lẻ tiếng Việt: đầu vào là một file âm thanh chứa một từ trong danh sách 20 từ, đầu ra là từ đó dạng văn bản. Phương pháp chính là MFCC kết hợp F0, so khớp DTW với kho mẫu và bỏ phiếu k láng giềng gần nhất.

Mọi lệnh chạy từ thư mục gốc của dự án, nơi có tệp `stt.py`. Từ mục 3 trở đi, `python` nghĩa là Python trong môi trường ảo của dự án: trên Windows là `.venv\Scripts\python`, trên macOS và Linux là `.venv/bin/python`.

## 1. Cần gì

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10, 11; macOS; Linux. Đã chạy thật trên Windows 11 |
| Python | 3.11, 3.12 hoặc 3.13. Đã chạy thật trên 3.13 |
| Đĩa trống | Khoảng 1 GB cho thư viện, 60 MB cho mã và dữ liệu mẫu |
| Mạng | Cần khi cài thư viện. Dữ liệu mẫu đã có sẵn trong kho mã, không cần mạng để chạy |
| Micro | Chỉ cần khi thu âm thật hoặc ghi âm trên giao diện |
| GPU | Không cần |

## 2. Cài đặt chi tiết

### 2.1 Cài Python

**Windows**

1. Vào https://www.python.org/downloads/windows/, tải bản Python 3.13 "Windows installer (64-bit)".
2. Chạy tệp cài. Ở màn hình đầu tiên, **tích ô "Add python.exe to PATH"**, rồi bấm Install Now.
3. Mở PowerShell mới, kiểm tra:

   ```powershell
   python --version
   ```

   Phải hiện `Python 3.13.x`. Nếu máy có nhiều bản Python, có thể gọi đúng bản bằng `py -3.13` thay cho `python` ở bước 2.3.

**macOS**

```bash
brew install python@3.13
python3.13 --version
```

Chưa có Homebrew thì tải bộ cài macOS ở https://www.python.org/downloads/macos/.

**Linux (Ubuntu, Debian)**

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip libportaudio2
python3 --version
```

`libportaudio2` cần cho chức năng thu âm từ micro. Nếu `python3 --version` cho bản cũ hơn 3.11, cài thêm bản mới qua kho deadsnakes hoặc pyenv.

### 2.2 Lấy mã nguồn

Cách 1, dùng Git (cài Git ở https://git-scm.com nếu chưa có):

```bash
git clone https://github.com/giabaohb99/speech_to_text.git
cd speech_to_text
```

Cách 2, không cần Git: mở https://github.com/giabaohb99/speech_to_text, bấm nút **Code**, chọn **Download ZIP**, giải nén, rồi mở cửa sổ dòng lệnh tại thư mục vừa giải nén.

Kho mã đã kèm sẵn dữ liệu mẫu (`data/raw`, 1000 file), kho đặc trưng đã dựng (`data/features`) và kết quả đánh giá (`results`).

### 2.3 Tạo môi trường ảo và cài thư viện

Môi trường ảo giữ thư viện của dự án riêng trong thư mục `.venv`, không ảnh hưởng Python chung của máy. Không cần "activate" môi trường; mọi lệnh gọi thẳng Python trong `.venv`.

**Windows, PowerShell hoặc cmd:**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS, Linux:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Bước cài mất khoảng 3 đến 5 phút tùy mạng. Các thư viện chính: librosa (xử lý âm thanh), numba (tăng tốc DTW), scikit-learn (SVM, Random Forest), streamlit (giao diện), edge-tts (sinh dữ liệu mẫu), sounddevice (thu âm).

### 2.4 Kiểm tra cài đặt

**Windows:**

```powershell
.venv\Scripts\python stt.py --help
.venv\Scripts\python -m pytest -q
.venv\Scripts\python stt.py recognize data/raw/mot_tts_f1_p2_03.wav
```

**macOS, Linux:**

```bash
.venv/bin/python stt.py --help
.venv/bin/python -m pytest -q
.venv/bin/python stt.py recognize data/raw/mot_tts_f1_p2_03.wav
```

Kết quả đúng:

- Lệnh đầu liệt kê các lệnh `gen-data`, `record`, `inventory`, `build-features`, `plot`, `recognize`, `evaluate`, `demo`, `package`.
- Lệnh kiểm thử báo toàn bộ đạt, ví dụ `33 passed`. Lần đầu mất khoảng 40 giây vì numba biên dịch.
- Lệnh cuối in `Từ nhận dạng: một`.

### 2.5 Lỗi khi cài

| Hiện tượng | Cách xử lý |
|---|---|
| `python` không được nhận ra trên Windows | Cài lại Python và tích "Add python.exe to PATH", hoặc dùng `py -3.13` thay cho `python` |
| PowerShell báo không chạy được script khi activate | Không cần activate; gọi thẳng `.venv\Scripts\python` như hướng dẫn |
| pip báo lỗi khi cài `numba` hoặc `llvmlite` | Phiên bản Python quá mới hoặc quá cũ; dùng Python 3.11 đến 3.13 rồi tạo lại `.venv` |
| `OSError: PortAudio library not found` trên Linux | `sudo apt install libportaudio2` |
| `sndfile library not found` trên Linux | `sudo apt install libsndfile1` |
| pip chậm hoặc ngắt giữa chừng | Chạy lại đúng lệnh cài, pip tiếp tục từ chỗ đã tải |
| Muốn cài lại từ đầu | Xóa thư mục `.venv` rồi làm lại bước 2.3 |

## 3. Chạy nhanh từ đầu đến cuối

```bash
python stt.py gen-data          # sinh 1000 file mẫu vào data/raw, khoảng 20 giây
python stt.py build-features    # trích đặc trưng ra data/features, khoảng 1 phút
python stt.py recognize data/raw/mot_tts_f1_p2_03.wav
python stt.py evaluate          # đánh giá 4 phương pháp, khoảng 2,5 phút
python stt.py demo              # mở giao diện tại http://localhost:8501
```

Kho mã đã có sẵn `data/raw` và `data/features`, nên có thể bỏ qua hai lệnh đầu. Chạy `gen-data` cần mạng để gọi edge-tts nếu thiếu bản giọng gốc trong `data/tts_cache`.

## 4. Các lệnh

| Lệnh | Làm gì | Kết quả ở đâu |
|---|---|---|
| `gen-data [--words MA ...]` | Sinh dữ liệu mẫu bằng edge-tts, 2 giọng gốc x 5 biến thể x 20 từ x 5 lần | `data/raw/`, bản gốc ở `data/tts_cache/` |
| `record --speaker MA [--repeats 5] [--words ...]` | Thu âm thật từ micro theo quy ước tên file | `data/raw/` |
| `inventory` | Đếm file theo người và từ, báo thiếu, sai tên, sai định dạng | màn hình |
| `build-features` | Tiền xử lý, trích đặc trưng mọi file | `data/features/features_frames.csv`, `features_summary.csv` |
| `plot [FILE ...]` | Vẽ hình 6 ô: dạng sóng, năng lượng, ZCR, F0, spectrogram, MFCC | `results/figures/` |
| `recognize FILE ... [--exclude-speaker ...] [--k K] [--no-figure]` | Nhận dạng file, in từ, độ tin cậy, top 3 | màn hình, hình và `log.csv` ở `results/recognize/` |
| `evaluate [--sweep] [--dtw-only]` | Đánh giá hai kịch bản người đã có và giọng mới | thư mục mới `results/evaluate/<thời điểm>/` |
| `demo [--port 8501] [--no-browser]` | Mở giao diện trên trình duyệt | hình, file đầu vào, `log.csv` ở `results/demo/` |
| `package` | Nén toàn bộ dự án để nộp, bỏ `.venv` và thư mục đệm | `dist/speech-to-text_<thời điểm>.zip` |

Không lệnh nào ghi đè kết quả của lần chạy trước, trừ `build-features` dựng lại kho đặc trưng.

### 4.1 Quy ước tên file

`<mã từ>_<mã người>_<số thứ tự>.wav`, ví dụ `mot_an_03.wav`. Mã từ lấy ở cột đầu của `words.txt`. Mã người viết thường, không dấu, không khoảng trắng. Dữ liệu sinh tự động có mã người bắt đầu bằng `tts_`.

### 4.2 Nhận dạng một file

```bash
python stt.py recognize duong/dan/file.wav
```

File có thể là wav, flac, ogg hoặc mp3, tần số và số kênh bất kỳ; hệ thống tự đổi về 16 kHz một kênh. Nếu file nằm sẵn trong kho, hệ thống tự loại chính nó khỏi kho khi so khớp.

Giả lập người chưa từng có trong kho bằng cách loại người đó, hoặc cả nhóm giọng:

```bash
python stt.py recognize data/raw/mot_tts_f1_p2_03.wav --exclude-speaker tts_f1
```

## 5. Giao diện demo

```bash
python stt.py demo
```

Trình duyệt tự mở `http://localhost:8501` sau khoảng 3 giây. Nếu không tự mở, dán địa chỉ đó vào trình duyệt. Dừng bằng Ctrl+C ở cửa sổ dòng lệnh.

1. Chọn **nguồn âm thanh**:
   - **Ghi âm từ micro:** bấm biểu tượng micro, đọc rõ một từ, bấm dừng. Trình duyệt sẽ hỏi quyền dùng micro ở lần đầu.
   - **Tải file lên:** chọn file wav, flac, ogg hoặc mp3.
   - **Chọn file mẫu trong kho:** chọn một file có sẵn, tiện khi không có micro.
2. Bấm **Nhận dạng**.
3. Kết quả: từ nhận dạng, độ tin cậy, bảng ba ứng viên, năm láng giềng gần nhất, hình kết quả 4 ô.

Thanh bên trái:

- **Tham số:** số láng giềng k, trọng số F0. Đổi xong bấm Nhận dạng lại.
- **Giả lập giọng mới:** chọn người hoặc nhóm giọng cần loại khỏi kho, ví dụ `tts_f1` để loại toàn bộ giọng nữ sinh tự động.

Mỗi lần bấm Nhận dạng lưu vào `results/demo/`: hình kết quả, một dòng `log.csv`, và với micro hoặc file tải lên thì lưu cả file âm thanh đầu vào trong `results/demo/inputs/`.

## 6. Dùng với thu âm thật của nhóm

Kết quả trên dữ liệu sinh tự động chỉ để thử hệ thống. Đề bài yêu cầu đánh giá trên giọng của thành viên.

1. **Thu âm:** mỗi thành viên chạy một lần, mã người là tên viết thường không dấu. Nên thu ít nhất 3 người, mỗi từ 5 lần, phòng yên tĩnh, cùng một micro.

   ```bash
   python stt.py record --speaker an
   ```

   Mỗi lần: nhấn Enter, đọc ngay một từ trong 2 giây. Sau đó nhấn Enter để lưu, `n` để nghe lại, `r` để thu lại, `q` để dừng. Thu tiếp lần sau thì số thứ tự nối tiếp, không ghi đè.

2. **Kiểm kê:**

   ```bash
   python stt.py inventory
   ```

3. **Quyết định giữ hay bỏ dữ liệu sinh.** Muốn đánh giá chỉ trên giọng thật thì xóa các file có `_tts_` trong tên ở `data/raw/`. Có thể chạy lại `gen-data` bất cứ lúc nào để sinh lại y hệt.
4. **Dựng lại kho và đánh giá:**

   ```bash
   python stt.py build-features
   python stt.py evaluate
   ```

   Với 3 người, kịch bản giọng mới có 3 vòng: mỗi vòng giấu một người làm test, hai người còn lại làm train.

5. **Cập nhật nhận xét và đóng gói lại:** sửa `results/README.md` theo số liệu mới, rồi chạy `python stt.py package`.

## 7. Đổi danh sách từ

Sửa `words.txt`, mỗi dòng `mã,từ có dấu`, mã chỉ gồm chữ thường không dấu và số. Sau đó sinh lại dữ liệu hoặc thu âm cho từ mới, rồi chạy `build-features`.

## 8. Đọc kết quả đánh giá

Mỗi thư mục `results/evaluate/<thời điểm>/` có `README.md` tự sinh tóm tắt số liệu. Các tệp chính:

- `compare.png`: accuracy của DTW + F0, DTW không F0, SVM, Random Forest ở hai kịch bản, và thời gian nhận dạng.
- `confusion_<phương pháp>_<A|B>.png`: ma trận nhầm lẫn. Hàng là từ thật, cột là từ đoán; ô trên đường chéo là đúng, số đỏ ngoài đường chéo là nhầm.
- `accuracy_by_speaker.png`: accuracy từng người.
- `hard_words.csv`: các cặp từ hay bị nhầm.
- Có `--sweep` thì thêm `sweep.png`: accuracy khi đổi k, trọng số F0, CMN.

Kịch bản A, **người đã có**: giấu từng file một, kho là phần còn lại. Kịch bản B, **giọng mới**: giấu toàn bộ một người, kho là những người còn lại. Mỗi vòng chỉ có hai tập train và test. Nhận xét kết quả hiện tại ở `results/README.md`.

## 9. Tham số hay chỉnh

Trong `config.yaml`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `knn.k` | 5 | số láng giềng bỏ phiếu |
| `features.f0_weight` | 2.0 | trọng số đường F0 khi so khớp, 0 là bỏ F0 |
| `features.cmn` | false | trừ trung bình MFCC theo từng file |
| `dtw.band_ratio` | 0.10 | độ rộng dải DTW so với độ dài chuỗi |
| `preprocess.energy_margin_db` | 6 | ngưỡng cắt lặng cao hơn nền bao nhiêu dB |
| `record.seconds` | 2.0 | độ dài mỗi lần thu âm |

Đổi tham số đặc trưng hoặc tiền xử lý thì phải chạy lại `build-features`. Đổi `knn`, `dtw`, `f0_weight`, `cmn` thì không cần.

## 10. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Chưa có kho đặc trưng` | Chạy `python stt.py build-features` |
| `gen-data` báo lỗi mạng | Cần mạng ở lần đầu để tải giọng từ edge-tts; kiểm tra kết nối rồi chạy lại |
| `record` không ghi được hoặc tiếng quá nhỏ | Kiểm tra micro mặc định trong cài đặt âm thanh của hệ điều hành; nói gần micro hơn |
| Giao diện không hỏi quyền micro | Mở bằng `http://localhost:8501`, không dùng địa chỉ IP; cho phép micro trong cài đặt trình duyệt |
| Lần nhận dạng đầu chậm vài giây | Bình thường: numba biên dịch hàm DTW lần đầu, các lần sau nhanh |
| `features_summary.csv không khớp kho khung` | Kho bị dựng dở; chạy lại `build-features` |
| Excel hiện sai tiếng Việt khi mở csv | Các csv ghi `utf-8-sig`; mở bằng Excel bản mới hoặc dùng Data, From Text và chọn UTF-8 |
| `build-features` báo không cắt được lặng | File gần như không có tiếng hoặc toàn tiếng ồn; nghe lại và thu lại file đó |
| Muốn file mp3 tải lên nhận được | Đã hỗ trợ wav, flac, ogg, mp3; định dạng m4a cần đổi sang wav trước |

## 11. Cấu trúc thư mục

```
speech_to_text/           thư mục dự án (tên khi clone từ GitHub)
  stt.py, app.py          dòng lệnh, giao diện
  config.yaml, words.txt  cấu hình, danh sách từ
  src/                    mã nguồn
  tests/                  kiểm thử
  data/raw/               file âm thanh
  data/features/          kho đặc trưng csv
  results/                hình và bảng kết quả
  doc/                    tài liệu 01 đến 05
```

Chi tiết kỹ thuật ở `doc/03-tai-lieu-ky-thuat.md`. Đối chiếu từng mục của đề bài ở `doc/04-doi-chieu-de-bai.md`.
