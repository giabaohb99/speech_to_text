# 03. Tài liệu kỹ thuật

Ngày lập: 2026-10-17. Mô tả thiết kế để lập trình theo. Tham số cụ thể nằm trong `config.yaml`, tài liệu này ghi giá trị mặc định đề xuất.

## 1. Kiến trúc và luồng dữ liệu

```
data/raw/*.wav
    |  build-features
    v
preprocess.load_and_clean  ->  features.extract  ->  store.write_csv
                                                          |
                                      data/features/features_frames.csv
                                      data/features/features_summary.csv
                                                          |
new.wav -> preprocess -> features -> recognizer.predict (DTW + k-NN)  -> word, confidence, top3
                                          ^
                                   store.load_bank (nạp csv vào bộ nhớ một lần)
```

Ba nguyên tắc:

- Mọi tham số số học đọc từ `config.yaml`, không viết cứng trong mã.
- Kho đặc trưng là **file csv**, mã nguồn không phụ thuộc định dạng nhị phân nào.
- Đường nhận dạng và đường đánh giá dùng chung bước biến đổi đặc trưng (`FeatureTransform`), hàm DTW và hàm bỏ phiếu (`vote`), tránh lệch kết quả. Đánh giá tính sẵn cả ma trận khoảng cách rồi bỏ phiếu, nhận dạng tính một hàng.

## 2. Cấu trúc thư mục

```
speech-to-text/
  stt.py                  điểm vào dòng lệnh, các lệnh con
  config.yaml             tham số
  words.txt               danh sách từ, mỗi dòng một từ
  requirements.txt
  HUONG-DAN.md            hướng dẫn sử dụng (viết ở phase 6)
  pytest.ini              cấu hình kiểm thử
  src/
    __init__.py           chặn hai cảnh báo vô hại của librosa 1.0
    common.py             nạp cấu hình, words.txt, quy ước tên file
    preprocess.py         nạp, chuẩn hóa, cắt khoảng lặng
    features.py           MFCC, F0, năng lượng, ZCR, centroid
    store.py              ghi và đọc csv, cache
    dtw.py                khoảng cách DTW
    recognizer.py         k-NN trên DTW, độ tin cậy
    classifier.py         SVM, Random Forest
    evaluate.py           hai kịch bản, bảng, hình
    gen_data.py           sinh dữ liệu mẫu
    record.py             thu âm thật
    inventory.py          kiểm kê kho
    plots.py              hình minh họa đặc trưng, hình nhận dạng, sơ đồ khối
    package.py            đóng gói zip để nộp
  app.py                  giao diện Streamlit
  data/
    raw/                  file wav gốc, tên theo quy ước
    tts_cache/            mp3 giọng gốc tải từ edge-tts
    features/             hai csv và cache
    README.md             nguồn gốc dữ liệu
  results/                accuracy.csv, ma trận nhầm lẫn, bảng so sánh
    figures/              hình minh họa đặc trưng (lệnh plot)
    recognize/            hình kết quả từng lần nhận dạng thử và log.csv
    demo/                 hình, log.csv và inputs/ từ giao diện demo
    evaluate/<thời điểm>/ bảng và hình của từng lần đánh giá
    README.md             nhận xét kết quả, viết tay
  tests/                  kiểm thử đơn vị cho preprocess, features, dtw, store
  doc/                    ba tài liệu này
```

## 3. Dữ liệu

### 3.1 Quy ước tên file

`<word>_<speaker>_<index>.wav`

- `word`: từ viết không dấu, chữ thường, không khoảng trắng, ánh xạ về từ có dấu qua `words.txt` (cột thứ hai). Ví dụ `mot`, `bon`, `xuong`.
- `speaker`: mã người, chữ thường không dấu, ví dụ `an`, `binh`, `tts_f1_p2` cho giọng sinh.
- `index`: số thứ tự hai chữ số, bắt đầu từ `01`.

Ví dụ: `mot_an_03.wav`, `xuong_tts_m1_p0_05.wav`.

`words.txt` dạng hai cột cách nhau bằng dấu phẩy: `mot,một`. Mã không dấu là khóa trong toàn bộ hệ thống, chữ có dấu chỉ dùng khi hiển thị.

### 3.2 Định dạng âm thanh

- WAV PCM 16 bit, 16 kHz, mono. File khác định dạng được đổi khi nạp, nhưng `inventory` cảnh báo.
- Độ dài 0,3 đến 2,5 giây. Ngoài khoảng này `inventory` cảnh báo. Từ một âm tiết đọc nhanh có thể chỉ khoảng 0,5 giây kể cả khoảng lặng.

### 3.3 Danh sách từ đề xuất

| Mã | Từ | Mã | Từ |
|---|---|---|---|
| khong | không | bat | bật |
| mot | một | tat | tắt |
| hai | hai | mo | mở |
| ba | ba | dong | đóng |
| bon | bốn | trai | trái |
| nam | năm | phai | phải |
| sau | sáu | len | lên |
| bay | bảy | xuong | xuống |
| tam | tám | dung | dừng |
| chin | chín | chay | chạy |

Danh sách có vài cặp gần nhau về âm để bài đánh giá có ý nghĩa: `nam` và `tam`, `bay` và `hai`, `dong` và `dung`.

## 4. Sinh dữ liệu mẫu

### 4.1 Nguồn giọng gốc

Đã chốt dùng **edge-tts** với giọng `vi-VN-HoaiMyNeural` (mã `f1`, nữ) và `vi-VN-NamMinhNeural` (mã `m1`, nam). Đây là hai giọng tiếng Việt duy nhất edge-tts có. Chất lượng tốt, miễn phí, cần mạng ở lần sinh đầu.

- Bộ sinh gọi từng từ riêng lẻ, mỗi lần một từ, để file chứa đúng một từ.
- Bản mp3 gốc lưu ở `data/tts_cache/<giọng>_<từ>.mp3`. Lần chạy sau dùng lại, không cần mạng.
- mp3 được đọc thẳng bằng `soundfile` (libsndfile 1.1 trở lên), không cần ffmpeg.
- edge-tts đệm mọi file thành khoảng 1,87 giây, phần có tiếng chỉ 0,2 đến 0,3 giây. Bộ sinh cắt lặng bằng `librosa.effects.trim(top_db=40)` rồi chừa 150 ms mỗi đầu trước khi tạo biến thể.

Không cài nguồn dự phòng (Windows SAPI, gTTS). Thêm khi cần chạy hoàn toàn offline từ đầu.

### 4.2 Tạo biến thể giọng

Từ mỗi file giọng gốc tạo 4 "người" biến thể, mỗi biến thể là một tổ hợp cố định:

| Biến thể | Dịch cao độ (nửa cung) | Kéo giãn thời gian | Mục đích |
|---|---|---|---|
| p0 | 0 | 1,00 | giọng gốc |
| p1 | +2 | 0,92 | cao hơn, nhanh hơn |
| p2 | -2 | 1,08 | trầm hơn, chậm hơn |
| p3 | +4 | 1,00 | giọng rất cao |
| p4 | -3 | 0,95 | giọng trầm nhanh |

Trong mỗi biến thể tạo 5 mẫu, mỗi mẫu lệch ngẫu nhiên: cao độ thêm trong khoảng cộng trừ 0,3 nửa cung, tốc độ cộng trừ 3 phần trăm, thêm nhiễu trắng với tỷ số tín hiệu trên nhiễu 20 đến 35 dB, dịch thời gian 0 đến 80 ms, nhân âm lượng 0,7 đến 1,0. Lệch cao độ và tốc độ giúp các lần đọc của cùng một người không chỉ khác nhau ở nhiễu. Hạt giống ngẫu nhiên cố định để sinh lại được.

Tổng: 2 giọng gốc, 5 biến thể mỗi giọng, 20 từ, 5 mẫu, bằng 1000 file, khoảng 24 MB, độ dài 0,49 đến 0,96 giây, sinh trong khoảng 20 giây.

Mỗi cặp từ và giọng gốc có hạt giống ngẫu nhiên riêng, nên `gen-data --words mot` sinh ra file giống hệt khi sinh cả kho.

librosa 1.0 phát cảnh báo FutureWarning từ `time_stretch` và "invalid value encountered in cast" từ numba (hàm `util.abs2` cấp phát mảng bằng `np.empty` rồi để ufunc ghi đè, numba kiểm ép kiểu trên dữ liệu rác chưa khởi tạo). Kết quả không bị ảnh hưởng, đã kiểm không có NaN. Hai cảnh báo được chặn đúng nội dung trong `src/__init__.py` và `pytest.ini`.

### 4.3 Giới hạn cần ghi vào báo cáo

Các biến thể từ cùng một giọng gốc **không độc lập** như người thật. Kịch bản "giọng mới" trên dữ liệu sinh sẽ lạc quan hơn thực tế. Khi đánh giá kịch bản B trên dữ liệu sinh, nên loại cả nhóm biến thể của một giọng gốc ra cùng lúc, không loại từng biến thể.

## 5. Tiền xử lý

Thứ tự trong `preprocess.load_and_clean(path, cfg) -> (y, info)`:

1. Nạp bằng `librosa.load(sr=16000, mono=True)`.
2. Bỏ thành phần một chiều: trừ trung bình.
3. Chuẩn hóa đỉnh về 0,95.
4. Cắt khoảng lặng:
   - Chia khung 25 ms bước 10 ms, tính năng lượng log và ZCR mỗi khung.
   - Năng lượng nền = phân vị 10 của năng lượng mọi khung trong file. Cách này vẫn đúng khi file không có đủ khoảng lặng ở hai đầu, khác với cách lấy trung vị 10 khung đầu và cuối trong bản thiết kế đầu.
   - Ngưỡng = lớn hơn trong hai giá trị: nền cộng 6 dB, và đỉnh trừ 35 dB. Vế sau chặn trường hợp nền quá sạch, ví dụ lặng số tuyệt đối.
   - Khung có tiếng khi năng lượng trên ngưỡng, hoặc năng lượng trên ngưỡng trừ 3 dB và ZCR trên 0,25 (giữ phụ âm xát đầu từ như `s`, `x`, `ch`).
   - Lấy từ khung có tiếng đầu tiên tới khung có tiếng cuối cùng, nới mỗi đầu 50 ms.
   - Nếu đoạn có tiếng dưới 80 ms, hoặc cả file không có khung nào có tiếng, giữ nguyên file và đặt `info["trimmed"] = False`. Lệnh `build-features` in tên các file này để nghe lại. Ngưỡng 80 ms vì chữ "tắt" đọc nhanh chỉ khoảng 115 ms có tiếng.
5. `info` chứa năng lượng, ZCR, ngưỡng và vị trí cắt để vẽ hình minh họa.

**Pre-emphasis** (`y[n] = x[n] - 0.97 * x[n-1]`) không làm ở bước này mà làm ngay trước khi tính MFCC. F0, năng lượng, ZCR và centroid tính trên tín hiệu chưa pre-emphasis, vì pre-emphasis làm méo đường F0 và năng lượng.

Tham số trong `config.yaml`: `frame_ms=25`, `hop_ms=10`, `energy_margin_db=6`, `pad_ms=50`, `min_voiced_ms=80`. Các hằng số phân vị 10, khoảng động 35 dB, ZCR 0,25 nằm đầu `src/preprocess.py`.

## 6. Trích đặc trưng và lược đồ csv

### 6.1 Đặc trưng theo khung

Dùng `librosa` với `n_fft=512`, `hop_length=160`, `win_length=400`, cửa sổ Hamming, `n_mels=40`.

| Nhóm | Chiều | Cách tính |
|---|---|---|
| MFCC | 13 | `librosa.feature.mfcc(n_mfcc=13)` trên tín hiệu đã pre-emphasis, giữ hệ số 0 |
| Delta | 13 | `librosa.feature.delta(order=1, width=9)`; file quá ngắn thì thu nhỏ `width` theo số khung |
| Delta-delta | 13 | `librosa.feature.delta(order=2, width=9)` |
| F0 | 1 | `librosa.pyin(fmin=70, fmax=400, frame_length=1024)`; khung vô thanh nội suy tuyến tính; chuyển sang log2, lọc trung vị 5 khung để khử lỗi nhảy quãng tám, trừ trung vị của file để bỏ khác biệt cao độ giữa người mà vẫn giữ hình dạng thanh điệu |
| Năng lượng | 1 | dB của trung bình bình phương khung |
| ZCR | 1 | `librosa.feature.zero_crossing_rate` |
| Spectral centroid | 1 | `librosa.feature.spectral_centroid` |

Vector dùng cho DTW: 39 chiều MFCC cộng F0 nhân trọng số `f0_weight` (mặc định 2,0, tắt bằng 0). Ba đặc trưng cuối chỉ lưu để trình bày và cho đường phụ.

Chuẩn hóa do `recognizer.FeatureTransform` làm, theo thứ tự:

1. CMN tùy chọn (`features.cmn`, mặc định tắt): trừ trung bình theo thời gian của 13 MFCC trong chính file đó.
2. Trừ trung bình, chia độ lệch chuẩn từng chiều, học bằng `fit` trên đúng tập train đang dùng. Chiều có std bằng 0 thì chia cho 1.
3. Nhân chiều F0 với `f0_weight`.

Thống kê không lưu ra file. Bản thiết kế đầu có `features_stats.csv` tính một lần trên toàn kho, nhưng ở kịch bản giọng mới như vậy là rò rỉ dữ liệu test vào thống kê, nên đã bỏ. Lệnh `recognize` học thống kê trên cả kho khi nạp, mất không đáng kể.

### 6.2 `features_frames.csv`

Một dòng một khung.

| Cột | Kiểu | Ghi chú |
|---|---|---|
| file_id | str | tên file không đuôi |
| word | str | mã không dấu |
| speaker | str | |
| sample_idx | int | |
| frame_idx | int | từ 0 |
| mfcc_01 .. mfcc_13 | float | |
| d_01 .. d_13 | float | delta |
| dd_01 .. dd_13 | float | delta-delta |
| f0 | float | log2 đã trừ trung vị, 0 nếu toàn file vô thanh |
| energy | float | |
| zcr | float | |
| centroid | float | |

Đo thực tế với 1000 file mẫu: khoảng 44 nghìn dòng (trung bình 44 khung mỗi file), 48 cột, 18 MB. Đọc lại vào bộ nhớ bằng `store.load_bank` mất 1,3 giây, nên không làm bộ đệm `.npz`.

Cả hai file csv ghi mã hóa `utf-8-sig` để Excel hiện đúng tiếng Việt.

### 6.3 `features_summary.csv`

Một dòng một mẫu, phục vụ mục 6 của đề (xem theo từ và theo người bằng Excel) và đường phụ SVM.

| Cột | Ghi chú |
|---|---|
| file_id, word, speaker, sample_idx | khóa |
| text | từ có dấu, để đọc bằng Excel |
| duration_ms, n_frames | sau cắt lặng |
| mean_mfcc_01 .. mean_mfcc_13 | trung bình theo thời gian |
| std_mfcc_01 .. std_mfcc_13 | độ lệch chuẩn |
| mean_d_01 .. mean_d_13 | |
| f0_hz_median | cao độ trung vị (Hz) của các khung hữu thanh, đặc trưng cho giọng từng người |
| voiced_ratio | tỷ lệ khung hữu thanh |
| f0_std, f0_slope | trên đường F0 đã trừ trung vị; slope là hệ số góc hồi quy tuyến tính theo khung, phản ánh thanh điệu |
| energy_mean, energy_max, zcr_mean, centroid_mean | |

Bản thiết kế đầu có `f0_mean`, nhưng F0 đã trừ trung vị nên trung bình luôn gần 0, vô nghĩa. Thay bằng `f0_hz_median` lấy từ F0 gốc.

Kiểm tra trên dữ liệu mẫu: `f0_slope` dương ở các từ thanh sắc (sáu, tám, chín, bốn, đóng, xuống) và âm ở các từ thanh nặng, huyền (chạy, một, bật, dừng). Đặc trưng F0 mang được thông tin thanh điệu.


### 6.4 Hình minh họa

Lệnh `python stt.py plot [file.wav ...]` vẽ một hình 6 ô cho mỗi file vào `results/figures/`: dạng sóng kèm vùng giữ lại sau cắt lặng, năng lượng kèm ngưỡng, ZCR, F0 (Hz), spectrogram, MFCC. Không truyền file thì vẽ 4 mẫu mặc định: một, năm, tám, sáu, kèm hình sơ đồ khối hệ thống `so-do-khoi.png`. Ô MFCC chuẩn hóa từng hệ số chỉ để hiển thị, vì hệ số 0 lớn hơn hẳn các hệ số khác.

## 7. So khớp bằng DTW và k-NN

### 7.1 DTW

Đầu vào hai ma trận `A (n x d)` và `B (m x d)` đã chuẩn hóa. Chi phí cục bộ là khoảng cách Euclid. Quy hoạch động chuẩn với ba bước đi (ngang, dọc, chéo), ràng buộc dải Sakoe-Chiba bán kính `r = max(5, 0.1 * max(n, m))` khung. Khoảng cách cuối chia cho `n + m` để không thiên vị mẫu ngắn.

Dải là **dải nghiêng** bám theo đường chéo nối hai điểm đầu cuối, không phải dải quanh `i = j`, nên hai chuỗi dài ngắn khác nhau vẫn luôn có đường đi. Điều kiện `|(i-1)(m-1) - (j-1)(n-1)| <= r (n-1)(m-1) / max(n, m)` đối xứng khi đổi chỗ A và B.

Cài trong `src/dtw.py` bằng numba (đi kèm librosa, không thêm thư viện), biên dịch có cache nên chỉ lần chạy đầu chậm vài trăm mili giây. Có thêm `dtw_path` truy vết đường khớp để vẽ hình. Kiểm thử trong `tests/test_dtw.py`: khoảng cách với chính nó bằng 0, đối xứng khi khác độ dài, khớp cài đặt tham chiếu không dải trên 20 cặp ngẫu nhiên, dải không làm giảm khoảng cách, bản co giãn thời gian gần hơn chuỗi khác hẳn, đường khớp đơn điệu từ đầu tới cuối.

Đo thực tế: kho 1000 mẫu, trung bình 44 khung, 40 chiều, mỗi lần nhận dạng mất 17 đến 60 ms phần so khớp. Không cần lọc sơ bộ.

### 7.2 Quyết định

- Lấy `k = 5` mẫu gần nhất.
- Điểm mỗi từ = tổng `1 / (dist + eps)` của các mẫu thuộc từ đó trong k.
- Từ dự đoán = từ có điểm cao nhất.
- Độ tin cậy = điểm từ thắng chia tổng điểm k mẫu, nằm trong 0 đến 1. Dưới `0,4` hiển thị cảnh báo "không chắc".
- Top 3 = xếp các từ theo điểm giảm dần, từ không có phiếu xếp sau theo khoảng cách nhỏ nhất tăng dần, lấy ba từ đầu.

Lớp `Recognizer(bank, cfg)` trong `src/recognizer.py` chuẩn hóa kho một lần khi khởi tạo. Hàm `predict(frames, exclude_speakers=(), exclude_files=(), k=None) -> Prediction` với `Prediction(word, confidence, top3, neighbors, word_distances, elapsed_ms)`.

- `exclude_speakers`: loại người trùng tên, hoặc cả nhóm khi truyền tiền tố, ví dụ `tts_f1` loại mọi `tts_f1_p*`. Dùng cho kịch bản B và chế độ giả lập người mới.
- `exclude_files`: loại file cụ thể. Lệnh `recognize` tự loại chính file đầu vào nếu nó đã nằm trong kho, để không tự khớp với chính mình.

### 7.3 Lệnh `recognize` và hình kết quả

```bash
python stt.py recognize FILE [FILE ...] [--exclude-speaker NGUOI ...] [--k K] [--no-figure]
```

In từ nhận dạng, độ tin cậy, top 3, mẫu gần nhất, thời gian so khớp. Mặc định mỗi file lưu một hình vào `results/recognize/<tên file>_<thời điểm>.png` gồm 4 ô: tín hiệu sau cắt lặng, MFCC, ma trận khoảng cách khung với mẫu gần nhất kèm đường khớp DTW, khoảng cách nhỏ nhất của 10 từ gần nhất. Mỗi lần chạy ghi thêm một dòng vào `results/recognize/log.csv`: thời điểm, file, từ nhận dạng, độ tin cậy, top 2, top 3, mẫu gần nhất, người bị loại, thời gian, tên hình.

## 8. Đường phụ: SVM và Random Forest

Đầu vào là các cột số của `features_summary.csv`, trừ `f0_hz_median` vì cột này đặc trưng cho người nói chứ không cho từ, còn lại 48 cột. SVM nhân RBF với `C=10`, `gamma='scale'`, chuẩn hóa bằng `StandardScaler` trong cùng pipeline; Random Forest 300 cây, `random_state=0`. Cài trong `src/classifier.py`.

Huấn luyện lại trong từng vòng chia tập của đánh giá, không huấn luyện một lần rồi dùng chung, để không rò rỉ dữ liệu. Các vòng lấy từ `evaluate.fold_indices`, dùng chung với DTW. Vì kịch bản A có 1000 vòng, các vòng chạy song song bằng joblib trên mọi nhân, mỗi mô hình đơn luồng; huấn luyện đa luồng từng mô hình chậm hơn nhiều vì chi phí điều phối.

Thời gian ghi trong bảng so sánh đo cùng một cách cho mọi phương pháp: nhận dạng một mẫu mới, đơn luồng, trung vị trên 20 mẫu, chưa tính trích đặc trưng.

## 9. Đánh giá

### 9.0 Nguyên tắc chia dữ liệu

Chỉ có **hai tập: train và test**. Dùng **kiểm tra chéo** (cross-validation), không chia cố định một lần. Mỗi vòng, tập train là kho dùng để so khớp hoặc huấn luyện, tập test là phần đem thử. Các vòng đổi vai cho nhau, mọi mẫu đều được làm test đúng một lần, kết quả gộp lại thành accuracy. Không có tập kiểm định riêng, vì dữ liệu ít và DTW không có bước huấn luyện.

Tham số như `k`, `f0_weight`, dải DTW được **chốt trên dữ liệu sinh tự động**, sau đó giữ nguyên khi đo trên thu âm thật, không cắt nhỏ bộ thu âm thật ra để chỉnh tham số.

### 9.1 Kịch bản A, người đã có

Kiểm tra chéo **theo mẫu** (leave-one-sample-out). Với mỗi mẫu `s` của người `p`: train = toàn bộ kho trừ `s`; test = `s`. Với 10 người, 20 từ, 5 mẫu là 1000 vòng. Accuracy báo chung và theo từng người.

### 9.2 Kịch bản B, giọng mới

Kiểm tra chéo **theo người** (leave-one-speaker-out). Số tập bằng số người: với nhóm 3 thành viên là 3 tập kiểm tra ngược nhau. Vòng 1 train là người 2 và 3, test người 1; vòng 2 train 1 và 3, test 2; vòng 3 train 1 và 2, test 3. Trên dữ liệu sinh, "người" ở đây là **nhóm biến thể của một giọng gốc** (xem 4.3), nên chỉ có 2 tập. Trên dữ liệu thật, là từng thành viên.

Nhóm giọng xác định bằng `evaluate.speaker_group`: mã `tts_<giọng>_p<số>` gộp về `tts_<giọng>`, người thật giữ nguyên mã. Chỉ có một nhóm thì bỏ qua kịch bản B và báo ra màn hình.

Ở kịch bản B, thống kê chuẩn hóa học lại trên train của từng vòng. Ở kịch bản A, bỏ một mẫu ra không làm đổi thống kê đáng kể nên học một lần trên cả kho.

Với SVM và Random Forest, huấn luyện lại trong từng vòng trên đúng tập train của vòng đó.

### 9.3 Cách tính

`compute_folds` tính ma trận khoảng cách DTW một lần cho mỗi cấu hình: kịch bản A là ma trận 1000 x 1000 với đường chéo đặt vô cùng để không tự khớp; kịch bản B là ma trận test x train cho từng nhóm. `cross_distances` chạy song song trên mọi nhân CPU. `predict_folds` bỏ phiếu bằng đúng hàm `vote` của bộ nhận dạng, nên đổi k không cần tính lại DTW.

Đo thực tế trên máy 22 nhân: mỗi cấu hình mất khoảng 4 giây cho cả hai kịch bản, lượt quét 8 cấu hình cộng đánh giá chính hết 42 giây.

### 9.4 Lệnh và đầu ra

```bash
python stt.py evaluate [--sweep] [--dtw-only]
```

Mỗi lần chạy tạo thư mục mới `results/evaluate/<thời điểm>/`, không ghi đè lần trước:

- `accuracy.csv`: cột method, scenario, speaker, n, correct, accuracy; dòng speaker `TẤT CẢ` là tổng.
- `predictions.csv`: từng lần dự đoán, có từ thật, từ đoán, độ tin cậy, mẫu gần nhất.
- `confusion_<dtw_f0|dtw_khong_f0|svm|random_forest>_<A|B>.png` và `.csv`: ma trận nhầm lẫn, hàng là từ thật, cột là từ đoán, tô màu theo tỷ lệ hàng, số đỏ là nhầm.
- `compare.png`: accuracy của các phương pháp theo hai kịch bản, và thời gian nhận dạng một mẫu.
- `accuracy_by_speaker.png`: accuracy từng người ở hai kịch bản.
- `hard_words.csv`: các cặp từ bị nhầm, xếp theo số lần.
- `README.md`: tóm tắt tự sinh gồm tham số, bảng accuracy, thời gian, 10 cặp nhầm nhiều nhất.
- Có `--sweep`: thêm `sweep.csv`, `sweep.png` quét k thuộc 1, 3, 5, 7; trọng số F0 thuộc 0, 1, 2, 4; CMN bật và tắt.

Bốn phương pháp: DTW với F0 theo cấu hình, DTW không F0 để bàn luận vai trò thanh điệu, SVM và Random Forest (mục 8). `--dtw-only` chỉ chạy hai phương pháp DTW, mất khoảng 13 giây thay vì 2,5 phút. Nhận xét kết quả viết tay ở `results/README.md`.

## 10. Giao diện demo

`app.py`, Streamlit 1.66, một trang. Chạy bằng `python stt.py demo`.

- Nguồn âm thanh: ghi âm bằng `st.audio_input` có sẵn của Streamlit, tải file wav, flac, ogg, mp3, hoặc chọn file mẫu trong `data/raw`. Lựa chọn thứ ba để thử khi không có micro.
- Kết quả: cột trái có từ dự đoán cỡ chữ lớn, độ tin cậy, cảnh báo khi dưới ngưỡng, bảng top 3, láng giềng gần nhất, ghi chú tham số; cột phải là hình kết quả 4 ô như lệnh `recognize`.
- Thanh bên: thông tin kho (số từ, số người, số mẫu), danh sách từ, chỉnh `k` và `f0_weight`, ô "giả lập giọng mới" chọn người hoặc nhóm giọng cần loại khỏi kho.
- Nhận dạng gọi `recognizer.recognize_file`, đúng hàm của lệnh `recognize`, lưu hình và `log.csv` vào `results/demo/`. Âm thanh ghi âm hoặc tải lên được lưu vào `results/demo/inputs/` trước khi nhận dạng. Biến môi trường `STT_DEMO_DIR` đổi thư mục lưu, dùng cho kiểm thử.
- Kho nạp một lần bằng `st.cache_resource`; mỗi trọng số F0 có một bộ nhận dạng được cache riêng.
- Lệnh `demo` chạy Streamlit với `--server.headless true` vì ở lần chạy đầu, chế độ thường hỏi email trên dòng lệnh và chặn chờ nhập; sau 3 giây tự mở trình duyệt bằng `webbrowser`, trừ khi có `--no-browser`.

Kiểm thử `tests/test_app.py` dùng `streamlit.testing.v1.AppTest` trên kho thật, tự bỏ qua nếu chưa dựng kho.

## 11. Cấu hình mặc định

Nguồn sự thật là tệp `config.yaml` ở gốc dự án, có chú thích từng tham số. Các nhóm: `audio`, `preprocess`, `features`, `dtw`, `knn`, `gen_data`, `record`, `paths`. Giá trị mặc định đề xuất ở các mục 5 đến 7 của tài liệu này đã được ghi vào đó.

## 12. Phụ thuộc

Python 3.11 trở lên, đã chạy trên 3.13 với librosa 1.0. `numpy`, `scipy`, `pandas`, `librosa`, `soundfile`, `scikit-learn`, `matplotlib`, `pyyaml`, `streamlit`, `sounddevice` (thu âm), `edge-tts` (sinh dữ liệu), `pytest`. DTW dùng `numba`, cài sẵn theo librosa.

Không dùng PyTorch hay TensorFlow trừ khi chốt làm CNN ở phase 5.

## 13. Kiểm thử tối thiểu

- `tests/test_common.py` (đã có): quy ước tên file, đọc `words.txt`, số thứ tự thu âm.
- `tests/test_features.py` (đã có, gộp cả tiền xử lý): file có 300 ms lặng hai đầu được cắt còn phần tiếng cộng đệm và mất không quá 20 ms tiếng; phụ âm xát nhỏ đầu từ được giữ; file toàn lặng không gây lỗi; số khung đúng công thức; F0 của âm 150 Hz đo đúng; F0 đi lên cho `f0_slope` dương; dựng kho rồi đọc lại khớp số khung, std khác 0.
- `tests/test_dtw.py` (đã có, gộp cả bộ nhận dạng): các tính chất DTW ở mục 7.1; ma trận song song khớp cách tính từng cặp; CMN xóa được lệch hằng số trên MFCC; quy tắc loại người theo tên và theo nhóm; kho giả 3 từ dự đoán đúng từ khi đã loại một người; loại file cụ thể khỏi kho.
- `tests/test_evaluate.py` (đã có): nhóm giọng; các vòng chia phủ mỗi mẫu đúng một lần ở mỗi kịch bản; kịch bản A không tự khớp, kịch bản B không khớp người cùng nhóm; SVM và Random Forest chạy đúng trên các vòng chia; cột cao độ bị loại khỏi đặc trưng; bảng accuracy đủ dòng tổng và dòng từng người.

## 14. Các điểm còn mở

| Vấn đề | Lựa chọn | Đề xuất |
|---|---|---|
| Nguồn tổng hợp tiếng nói | edge-tts cần mạng; SAPI cần gói ngôn ngữ | edge-tts, dự phòng SAPI |
| Giữ hệ số MFCC 0 | giữ mang thông tin năng lượng; bỏ bền hơn với âm lượng | giữ, vì đã chuẩn hóa đỉnh |
| F0 trong DTW | bật mặc định hay tắt | đã chốt: bật, trọng số 2,0; giúp giọng mới khoảng 2 điểm |
| CMN | bật hay tắt | đã chốt: tắt, không giúp trên dữ liệu sinh; thử lại trên thu âm thật |
| Lọc sơ bộ trước DTW | chỉ cần khi kho lớn | không cần, so khớp chỉ vài chục mili giây |
