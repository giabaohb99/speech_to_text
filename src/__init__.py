import warnings

# librosa 1.0 cấp phát mảng bằng np.empty rồi để ufunc của numba ghi đè (util.abs2, phase vocoder);
# numba kiểm ép kiểu trên dữ liệu rác chưa khởi tạo nên cảnh báo. Kết quả không bị ảnh hưởng.
warnings.filterwarnings("ignore", "invalid value encountered in cast", RuntimeWarning, module="numba")
# librosa 1.0 tự truyền tham số đã lỗi thời vào phase_vocoder bên trong time_stretch.
warnings.filterwarnings("ignore", "The `(hop_length|n_fft)` parameter is deprecated", FutureWarning)
