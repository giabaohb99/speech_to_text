"""Khoảng cách DTW (Dynamic Time Warping) có dải Sakoe-Chiba, biên dịch bằng numba."""
import numba
import numpy as np


@numba.njit(cache=True)
def _accumulate(a, b, radius):
    """Ma trận chi phí tích lũy (n+1) x (m+1). Ô ngoài dải giữ giá trị vô cùng."""
    n, m = a.shape[0], b.shape[0]
    dn, dm = max(n - 1, 1), max(m - 1, 1)
    limit = radius * dn * dm / max(n, m)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # dải nghiêng theo đường chéo nối hai đầu, bán kính tính theo số khung của chuỗi dài hơn;
            # điều kiện đối xứng khi đổi chỗ a và b
            if abs((i - 1) * dm - (j - 1) * dn) > limit:
                continue
            cost = 0.0
            for d in range(a.shape[1]):
                diff = a[i - 1, d] - b[j - 1, d]
                cost += diff * diff
            best = min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            acc[i, j] = np.sqrt(cost) + best
    return acc


@numba.njit(parallel=True, cache=True)
def _cross(flat_a, off_a, flat_b, off_b, band_ratio, band_min_frames):
    rows, cols = len(off_a) - 1, len(off_b) - 1
    out = np.empty((rows, cols))
    for i in numba.prange(rows):
        a = flat_a[off_a[i]:off_a[i + 1]]
        for j in range(cols):
            b = flat_b[off_b[j]:off_b[j + 1]]
            radius = max(band_min_frames, band_ratio * max(a.shape[0], b.shape[0]))
            out[i, j] = _accumulate(a, b, radius)[-1, -1] / (a.shape[0] + b.shape[0])
    return out


def _flatten(seqs):
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in seqs])
    return np.ascontiguousarray(np.vstack(seqs), dtype=np.float64), offsets


def cross_distances(seqs_a, seqs_b, band_ratio=0.10, band_min_frames=5):
    """Ma trận khoảng cách DTW len(seqs_a) x len(seqs_b), chạy song song trên nhiều nhân CPU."""
    flat_a, off_a = _flatten(seqs_a)
    flat_b, off_b = _flatten(seqs_b)
    return _cross(flat_a, off_a, flat_b, off_b, float(band_ratio), float(band_min_frames))


def _prepare(a, b, band_ratio, band_min_frames):
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    radius = max(band_min_frames, band_ratio * max(len(a), len(b)))
    return a, b, radius


def dtw_distance(a, b, band_ratio=0.10, band_min_frames=5):
    """Khoảng cách DTW giữa hai chuỗi (số khung x số chiều), chia cho tổng độ dài.

    Chi phí cục bộ là khoảng cách Euclid giữa hai khung. Đặt band_ratio rất lớn để tắt dải.
    """
    a, b, radius = _prepare(a, b, band_ratio, band_min_frames)
    return float(_accumulate(a, b, radius)[-1, -1] / (len(a) + len(b)))


def dtw_path(a, b, band_ratio=0.10, band_min_frames=5):
    """Đường khớp tối ưu dạng danh sách cặp (khung của a, khung của b), dùng để vẽ hình."""
    a, b, radius = _prepare(a, b, band_ratio, band_min_frames)
    acc = _accumulate(a, b, radius)
    i, j, path = len(a), len(b), []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min(steps)
    return path[::-1]
