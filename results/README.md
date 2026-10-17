# Kết quả

| Thư mục | Sinh bởi lệnh | Nội dung |
|---|---|---|
| `figures/` | `plot` | Hình minh họa đặc trưng 6 ô cho từng file |
| `recognize/` | `recognize` | Hình kết quả mỗi lần nhận dạng thử, kèm `log.csv` |
| `demo/` | `demo` | Hình và `log.csv` mỗi lần bấm Nhận dạng trên giao diện; `inputs/` giữ file âm thanh ghi âm hoặc tải lên |
| `evaluate/<thời điểm>/` | `evaluate` | Một thư mục cho mỗi lần đánh giá, có `README.md` tự sinh tóm tắt số liệu |

Không lần chạy nào ghi đè lần trước. Các lần đánh giá đã có:

| Thư mục | Nội dung | Dùng cho báo cáo |
|---|---|---|
| `evaluate/20261017-152220/` | Phase 4: hai biến thể DTW, kèm lượt quét tham số `sweep.png` | Phần chọn tham số |
| `evaluate/20261017-152821/` | Phase 5 lần đầu, cột thời gian đo sai cách (DTW song song, SVM đơn luồng) | Không dùng |
| `evaluate/20261017-153204/` | Phase 5 chính thức: bốn phương pháp, thời gian đo cùng cách | Bảng so sánh chính |
| `evaluate/20261017-153528/` | Thử chế độ `--dtw-only`, số liệu DTW trùng lần chính thức | Không cần |

## Kết quả chính thức trên dữ liệu sinh tự động

Kho 1000 mẫu, 20 từ, 10 người biến thể từ 2 giọng gốc edge-tts (một nữ, một nam). Tham số DTW đã chốt: k = 5, trọng số F0 = 2, CMN tắt, dải DTW 10 phần trăm. SVM nhân RBF C = 10; Random Forest 300 cây. Bốn phương pháp chạy trên đúng cùng các vòng chia.

| Phương pháp | Người đã có | Giọng mới | Nhận dạng một mẫu | Huấn luyện mỗi vòng |
|---|---|---|---|---|
| DTW + F0 | 99,3% | 50,9% | 27 ms | không cần |
| DTW không F0 | 99,5% | 48,7% | 32 ms | không cần |
| SVM | 99,1% | 44,0% | 0,3 ms | 144 ms |
| Random Forest | 98,7% | 46,8% | 13 ms | 2,4 s |

Thời gian đo đơn luồng, trung vị trên 20 mẫu, chưa tính bước trích đặc trưng vốn giống nhau cho mọi phương pháp. Hình tổng hợp: `evaluate/20261017-153204/compare.png`.

## Nhận xét

1. **Người đã có: bốn phương pháp ngang nhau, đều trên 98%.** Khi kho có sẵn giọng đó, cả so khớp chuỗi lẫn vector thống kê đều đủ phân biệt 20 từ.
2. **Giọng mới: DTW + F0 tốt nhất, hơn SVM gần 7 điểm và Random Forest 4 điểm.** DTW giữ thứ tự thời gian của chuỗi đặc trưng, còn SVM và Random Forest chỉ thấy trung bình và độ lệch nên mất thông tin diễn biến âm. Thông tin diễn biến này bền hơn khi đổi người nói.
3. **Giọng mới thấp ở mọi phương pháp vì dữ liệu sinh chỉ có hai giọng gốc.** Giấu nhóm giọng nữ thì kho chỉ còn giọng nam, và ngược lại. Đây là bài nhận dạng khác giới với đúng một giọng tham chiếu, trường hợp khó nhất.
4. **Giọng càng gần giọng trong kho càng dễ nhận.** Ở kịch bản giọng mới với DTW + F0, biến thể nữ trầm nhất đạt 69%, nữ cao nhất 25%; biến thể nam cao nhất đạt 67%, nam trầm nhất 29%. Xem `accuracy_by_speaker.png`.
5. **F0 giúp cho giọng mới** khoảng 2 điểm phần trăm, đều ở cả hai chiều nữ sang nam và nam sang nữ.
6. **CMN không giúp trên dữ liệu này** theo lượt quét ở `evaluate/20261017-152220/sweep.png`. Cần thử lại trên thu âm thật, nơi khác biệt micro và phòng thu đáng kể hơn.
7. **Đổi tham số gần như không đổi kết quả.** Lượt quét k từ 1 đến 7, F0 từ 0 đến 4 cho giọng mới chênh nhau trong khoảng 43 đến 51%. Phương án tốt nhất chỉ hơn phương án đã chốt 4 mẫu trên 1000.
8. **Từ hay bị nhầm ở giọng mới:** DTW hay đoán thành chạy, ba, lên; SVM nhầm nhiều nhất trái thành bật. Xem `hard_words.csv` và các ma trận nhầm lẫn.
9. **Tốc độ:** SVM nhanh nhất khi nhận dạng nhưng phải huấn luyện lại mỗi khi thêm mẫu. DTW chậm hơn, khoảng 27 ms với kho 1000 mẫu, vẫn dư cho dùng tương tác, và thêm mẫu là dùng được ngay.

Kết luận cho báo cáo: chọn **DTW + F0** làm phương pháp chính. Con số giọng mới trên dữ liệu sinh là **cận dưới bi quan**. Với thu âm thật của ít nhất ba thành viên, mỗi vòng sẽ còn hai giọng khác nhau trong kho thay vì một, nên kỳ vọng cao hơn. Cần chạy lại `evaluate` sau khi nhóm thu âm.
