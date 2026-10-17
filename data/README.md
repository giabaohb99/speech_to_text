# Dữ liệu

## Nguồn gốc

Toàn bộ file trong `raw/` có mã người bắt đầu bằng `tts_` là **dữ liệu sinh tự động**, không phải giọng người thật. Chúng dùng để phát triển và kiểm thử hệ thống khi nhóm chưa thu âm.

- Giọng gốc: edge-tts, `vi-VN-HoaiMyNeural` (mã `f1`, nữ) và `vi-VN-NamMinhNeural` (mã `m1`, nam). File gốc lưu trong `tts_cache/`.
- Mỗi giọng gốc có 5 biến thể `p0` đến `p4` (dịch cao độ, đổi tốc độ), mỗi biến thể coi là một người. Ví dụ `tts_f1_p3`.
- Mỗi người đọc mỗi từ 5 lần, các lần khác nhau ở nhiễu nền, lệch thời gian, âm lượng và lệch nhẹ cao độ, tốc độ.
- Tổng: 20 từ, 10 người, 5 lần, bằng 1000 file WAV 16 kHz mono 16 bit.
- Sinh lại y hệt bằng `python stt.py gen-data` (hạt giống cố định trong `config.yaml`).

Các biến thể cùng một giọng gốc **không độc lập** như người thật. Khi đánh giá kịch bản giọng mới trên dữ liệu này, cả nhóm `tts_f1_*` hoặc `tts_m1_*` phải được bỏ ra cùng lúc.

## Thu âm thật

File thu bằng `python stt.py record --speaker <mã>` có mã người là tên thành viên, ví dụ `mot_an_03.wav`. Dữ liệu thật và dữ liệu sinh nằm chung trong `raw/`, phân biệt bằng tiền tố `tts_`.

Muốn chỉ dùng dữ liệu thật: xóa các file `*_tts_*.wav` trong `raw/`.
