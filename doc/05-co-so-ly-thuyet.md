# 05. Cơ sở lý thuyết

Ngày lập: 2026-10-17. Tài liệu này phục vụ trình bày trong báo cáo cho mục 2, mục 5 và mục 6 của đề. Phần thiết kế cụ thể và tham số nằm ở tài liệu kỹ thuật 03.

## Phần A. Các kỹ thuật nhận dạng giọng nói (mục 2)

### A.1 Hai dạng bài toán

| | Nhận dạng từ đơn lẻ | Nhận dạng câu liên tục |
|---|---|---|
| Đầu vào | Một từ, có khoảng lặng hai đầu | Câu nói liền, ranh giới từ không rõ |
| Tập từ | Đóng, vài chục đến vài trăm từ | Mở, hàng chục nghìn từ |
| Thành phần cần có | So khớp hoặc phân loại cả đoạn âm | Mô hình âm học, từ điển phát âm, mô hình ngôn ngữ, bộ giải mã tìm chuỗi từ |
| Lượng dữ liệu cần | Vài mẫu mỗi từ mỗi người là chạy được | Hàng trăm đến hàng nghìn giờ ghi âm có phiên âm |
| Ứng dụng | Lệnh điều khiển, đọc số, menu bằng giọng | Đọc chính tả, phụ đề, trợ lý ảo |

Đề bài có một danh sách từ cố định và dữ liệu do nhóm tự thu, nên thuộc dạng thứ nhất.

### A.2 Các hướng kỹ thuật chính

1. **So khớp mẫu bằng DTW** (Dynamic Time Warping). Lưu các lần đọc mẫu của từng từ. Khi có âm mới, co giãn trục thời gian để căn hai chuỗi đặc trưng cho khớp nhau nhất rồi đo khoảng cách, chọn từ của mẫu gần nhất. Không cần huấn luyện, chạy tốt với rất ít dữ liệu, giải thích được. Yếu khi người nói mới khác nhiều so với người trong kho, và chậm dần khi kho lớn.
2. **Mô hình thống kê HMM-GMM** (Hidden Markov Model với Gaussian Mixture Model). Mỗi từ hoặc âm vị là một chuỗi trạng thái ẩn, mỗi trạng thái phát ra vector đặc trưng theo phân bố Gauss hỗn hợp. Là nền tảng của nhận dạng giọng nói trong nhiều thập kỷ, các bộ công cụ như HTK và Kaldi. Cần nhiều dữ liệu hơn DTW để ước lượng tham số.
3. **Phân loại trên vector đặc trưng cố định** như SVM, Random Forest. Nén cả đoạn âm thành một vector thống kê rồi dùng bộ phân loại học máy. Đơn giản, nhanh, nhưng mất thông tin diễn biến theo thời gian.
4. **Mạng nơ-ron trên spectrogram.** CNN coi mel-spectrogram như một bức ảnh; RNN hoặc LSTM đọc chuỗi khung. Hiệu quả cao cho nhận dạng từ khóa khi có đủ dữ liệu, thường vài chục mẫu mỗi từ trở lên.
5. **Mô hình đầu cuối và mô hình nền lớn.** Mạng học thẳng từ âm thanh ra chữ, ví dụ huấn luyện bằng CTC, wav2vec 2.0, Whisper. Với tiếng Việt có các bản tinh chỉnh như PhoWhisper và các mô hình wav2vec 2.0 tiếng Việt của cộng đồng, cùng các bộ dữ liệu công khai như VIVOS, VLSP, Common Voice. Đây là kỹ thuật hiện hành cho nhận dạng câu liên tục, nhưng cần GPU, dữ liệu lớn, và là hộp đen: không trích được các đặc trưng như năng lượng, ZCR, MFCC để trình bày theo yêu cầu mục 5 và 6.

### A.3 Đặc thù tiếng Việt

- **Đơn âm tiết:** mỗi tiếng là một âm tiết, thuận lợi cho nhận dạng từ đơn vì một từ trong danh sách thường là một khối âm liền.
- **Có thanh điệu:** sáu thanh ngang, huyền, sắc, hỏi, ngã, nặng. Cùng phụ âm và vần, khác thanh là khác nghĩa, ví dụ ma, má, mà. Thanh điệu thể hiện chủ yếu qua đường cao độ F0 theo thời gian. MFCC mô tả hình dạng phổ nên phản ánh thanh điệu yếu, cần bổ sung F0.
- **Phương ngữ:** giọng Bắc, Trung, Nam khác nhau về thanh và cách phát âm phụ âm, làm khó kịch bản người nói mới.

### A.4 Lựa chọn cho đề tài và lý do

Chọn **DTW trên MFCC kết hợp F0, bỏ phiếu k láng giềng gần nhất**, kèm SVM và Random Forest để so sánh.

- Phù hợp quy mô dữ liệu: vài người, mỗi từ vài lần đọc.
- Đáp ứng mục 5 và 6: đặc trưng được trích ra, lưu thành csv, so khớp dựa trên chính các đặc trưng đó.
- Giải thích được kết quả: chỉ ra mẫu nào trong kho giống nhất, vẽ được đường khớp.
- Bổ sung F0 để xử lý thanh điệu của tiếng Việt.

Kết quả thực nghiệm ở `results/README.md` xác nhận DTW + F0 tốt nhất ở kịch bản giọng mới trong bốn phương pháp đã thử.

## Phần B. Đặc trưng âm thanh và cách trích (mục 5)

Tín hiệu được chia thành **khung** dài 25 ms, bước dịch 10 ms, vì trong khoảng ngắn này tiếng nói coi như ổn định. Mọi đặc trưng dưới đây tính theo từng khung. Ký hiệu x[n] là mẫu thứ n trong khung, N là số mẫu một khung, ở 16 kHz thì N = 400.

### B.1 Năng lượng ngắn hạn

```
E = 10 * log10( (1/N) * tổng x[n]^2 )   (đơn vị dB)
```

Khung có tiếng nói có năng lượng cao hơn hẳn khung lặng. Hệ thống dùng năng lượng để **cắt khoảng lặng** hai đầu: ngưỡng bằng mức nền của chính file cộng 6 dB. Hình minh họa: ô thứ hai trong `results/figures/*.png`.

### B.2 Zero Crossing Rate

```
ZCR = (1/(N-1)) * số lần x[n] và x[n-1] trái dấu
```

Tỷ lệ tín hiệu đổi dấu. Âm hữu thanh như nguyên âm có ZCR thấp; âm xát vô thanh như s, x và tiếng ồn có ZCR cao. Hệ thống dùng ZCR cùng năng lượng để **giữ lại phụ âm xát đầu từ**, vốn nhỏ tiếng nên dễ bị cắt nhầm. Ô thứ ba trong hình minh họa.

### B.3 Cao độ F0 (pitch)

Tần số dao động của dây thanh, quyết định giọng cao hay trầm và mang thanh điệu tiếng Việt.

- Trích bằng thuật toán **pYIN**: dựa trên hàm sai khác giữa tín hiệu và bản dịch của chính nó, chu kỳ là độ trễ làm hàm này nhỏ nhất; pYIN thêm mô hình xác suất để chọn đường cao độ trơn và đánh dấu khung vô thanh.
- Xử lý: khung vô thanh được nội suy từ khung hai bên; chuyển sang thang log2; lọc trung vị 5 khung khử lỗi nhảy quãng tám; **trừ trung vị của cả file** để bỏ khác biệt giọng cao trầm giữa người, chỉ giữ hình dạng thanh điệu.
- Kiểm chứng trên dữ liệu: độ dốc F0 dương ở các từ thanh sắc như sáu, tám, chín và âm ở các từ thanh nặng, huyền như chạy, một, dừng.

Ô thứ tư trong hình minh họa.

### B.4 Spectrogram

Biến đổi Fourier thời gian ngắn (STFT) cho từng khung, nhân cửa sổ Hamming, độ dài FFT 512 điểm. Biên độ theo thang dB vẽ thành ảnh: trục ngang là thời gian, trục dọc là tần số, độ sáng là năng lượng. Thấy rõ các **formant**, tức dải tần cộng hưởng của khoang miệng quyết định nguyên âm, và vùng tần cao của phụ âm xát. Ô thứ năm trong hình minh họa.

### B.5 MFCC

Mel-Frequency Cepstral Coefficients, đặc trưng phổ biến nhất trong nhận dạng giọng nói. Các bước:

1. **Pre-emphasis** y[n] = x[n] - 0,97 x[n-1]: tăng phần tần cao vốn yếu trong tiếng nói.
2. **Chia khung và nhân cửa sổ Hamming** để giảm rò phổ ở mép khung.
3. **FFT** lấy phổ công suất từng khung.
4. **Bộ lọc thang Mel:** 40 bộ lọc tam giác dày ở tần thấp, thưa ở tần cao, mô phỏng tai người nhạy với thay đổi tần thấp hơn. Thang Mel: mel = 2595 log10(1 + f/700).
5. **Lấy log** năng lượng mỗi bộ lọc, mô phỏng cảm nhận độ to theo thang log.
6. **Biến đổi cosin rời rạc (DCT)**, giữ 13 hệ số đầu. DCT tách đường bao phổ, mang thông tin khoang miệng và âm vị, khỏi phần chi tiết do cao độ.

Thêm **delta** và **delta-delta**, là đạo hàm bậc một và bậc hai theo thời gian của 13 hệ số, mô tả phổ đang biến đổi ra sao. Tổng cộng 39 chiều mỗi khung. Ô cuối trong hình minh họa.

### B.6 Spectral centroid

Trọng tâm của phổ, trung bình tần số có trọng số theo biên độ. Âm sáng, nhiều tần cao có centroid cao. Lưu để trình bày và làm đặc trưng cho SVM.

### B.7 Đặc trưng dùng ở đâu

| Đặc trưng | Cắt lặng | So khớp DTW | SVM, Random Forest | Trình bày |
|---|---|---|---|---|
| Năng lượng | có | | có, trung bình và cực đại | có |
| ZCR | có | | có, trung bình | có |
| F0 | | có, trọng số 2 | có, độ lệch và độ dốc | có |
| Spectrogram | | | | có |
| MFCC, delta, delta-delta | | có | có, trung bình và độ lệch | có |
| Spectral centroid | | | có, trung bình | có |

## Phần C. Lưu trữ và so khớp (mục 6)

### C.1 Lưu trữ

Hai file csv trong `data/features/`, mở được bằng Excel:

- `features_frames.csv`: một dòng một khung, có cột từ, người, lần đọc, số thứ tự khung, 39 cột MFCC, F0, năng lượng, ZCR, centroid. Lọc theo cột `word` để xem một từ, theo cột `speaker` để xem một người.
- `features_summary.csv`: một dòng một file, gồm thống kê của cả file. Dùng để so sánh nhanh giữa các từ và các người, và làm đầu vào cho SVM, Random Forest.

### C.2 So khớp bằng DTW

Hai lần đọc cùng một từ thường dài ngắn khác nhau và nhanh chậm khác nhau ở từng đoạn. DTW tìm cách ghép từng khung của chuỗi này với một khung của chuỗi kia sao cho tổng khoảng cách nhỏ nhất, với điều kiện đi theo thứ tự thời gian, không quay lui.

```
D[i, j] = d(a_i, b_j) + min( D[i-1, j-1], D[i-1, j], D[i, j-1] )
```

Trong đó d là khoảng cách Euclid giữa hai vector đặc trưng đã chuẩn hóa. Khoảng cách cuối D[n, m] chia cho n + m để không thiên vị mẫu ngắn. Dải Sakoe-Chiba giới hạn đường đi không lệch quá 10 phần trăm độ dài khỏi đường chéo, vừa nhanh hơn vừa tránh ghép vô lý.

### C.3 Quyết định

1. Trích đặc trưng file mới theo đúng các bước trên.
2. Tính khoảng cách DTW tới mọi mẫu trong kho.
3. Lấy 5 mẫu gần nhất. Mỗi mẫu bỏ phiếu cho từ của nó với trọng số nghịch đảo khoảng cách.
4. Từ có tổng phiếu cao nhất là kết quả. Độ tin cậy bằng phiếu của từ thắng chia tổng phiếu.

Hình `results/recognize/*.png` minh họa trọn quá trình cho từng lần thử: tín hiệu, MFCC, ma trận khoảng cách khung với đường khớp DTW, và khoảng cách tới các từ.
