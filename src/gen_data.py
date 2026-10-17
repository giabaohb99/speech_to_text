"""Sinh dữ liệu mẫu: đọc từng từ bằng edge-tts rồi tạo biến thể giọng.

Mỗi giọng gốc sinh nhiều biến thể (dịch cao độ, đổi tốc độ), mỗi biến thể là một
"người" có mã dạng tts_<giọng>_p<số>. Trong một người, các lần đọc khác nhau ở
nhiễu nền, lệch thời gian, âm lượng và một chút cao độ, tốc độ.
"""
import asyncio
import zlib

import edge_tts
import librosa
import numpy as np
import soundfile as sf

from src.common import make_filename

TTS_CONCURRENCY = 4
BASE_PAD_S = 0.15


async def _fetch_one(text, voice, out_path, semaphore):
    async with semaphore:
        tmp = out_path.with_suffix(".part")
        await edge_tts.Communicate(text, voice).save(str(tmp))
        tmp.replace(out_path)


async def _fetch_all(jobs):
    semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
    await asyncio.gather(*(_fetch_one(*job, semaphore) for job in jobs))


def fetch_base_voices(words, voices, cache_dir):
    """Tải giọng gốc về cache mp3; file đã có thì bỏ qua để chạy lại không cần mạng."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        (voice_key, code): cache_dir / f"{voice_key}_{code}.mp3"
        for voice_key in voices
        for code in words
    }
    missing = [
        (words[code], voices[voice_key], path)
        for (voice_key, code), path in paths.items()
        if not path.exists()
    ]
    if missing:
        print(f"Tải {len(missing)} file giọng gốc từ edge-tts...")
        asyncio.run(_fetch_all(missing))
    return paths


def add_noise(y, snr_db, rng):
    voiced = y[np.abs(y) > 0.05 * np.max(np.abs(y))]
    signal_power = np.mean(voiced**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return y + rng.normal(0.0, np.sqrt(noise_power), len(y))


def load_base(path, sr):
    """Nạp giọng gốc, bỏ phần lặng edge-tts đệm sẵn, chừa lại 150 ms mỗi đầu."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    _, (start, end) = librosa.effects.trim(y, top_db=40)
    pad = int(BASE_PAD_S * sr)
    return y[max(0, start - pad): end + pad]


def make_sample(base, sr, semitones, time_scale, rng, gcfg):
    """Tạo một lần đọc từ giọng gốc theo tham số của biến thể cộng lệch ngẫu nhiên."""
    n_steps = semitones + rng.uniform(-gcfg["jitter_semitones"], gcfg["jitter_semitones"])
    scale = time_scale * (1 + rng.uniform(-gcfg["jitter_time"], gcfg["jitter_time"]))
    y = librosa.effects.pitch_shift(base, sr=sr, n_steps=n_steps)
    y = librosa.effects.time_stretch(y, rate=1.0 / scale)
    shift = int(rng.uniform(*gcfg["shift_ms"]) / 1000 * sr)
    y = np.concatenate([np.zeros(shift, dtype=y.dtype), y])
    y = y / np.max(np.abs(y)) * 0.9
    y = add_noise(y, rng.uniform(*gcfg["snr_db"]), rng)
    y = y * rng.uniform(*gcfg["gain"])
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def generate(cfg, words, only_words=None):
    gcfg = cfg["gen_data"]
    sr = cfg["audio"]["sample_rate"]
    raw_dir = cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if only_words:
        unknown = set(only_words) - set(words)
        if unknown:
            raise SystemExit(f"Mã từ không có trong words.txt: {', '.join(sorted(unknown))}")
        words = {code: text for code, text in words.items() if code in only_words}

    bases = fetch_base_voices(words, gcfg["voices"], cfg["paths"]["tts_cache"])
    n_written = 0
    for code in words:
        for voice_key in gcfg["voices"]:
            # hạt giống riêng cho từng cặp từ-giọng: sinh lẻ một từ hay cả kho đều ra file y hệt
            rng = np.random.default_rng([gcfg["seed"], zlib.crc32(f"{code}/{voice_key}".encode())])
            base = load_base(bases[(voice_key, code)], sr)
            for variant_idx, (semitones, time_scale) in enumerate(gcfg["variants"]):
                speaker = f"tts_{voice_key}_p{variant_idx}"
                for index in range(1, gcfg["samples_per_variant"] + 1):
                    y = make_sample(base, sr, semitones, time_scale, rng, gcfg)
                    sf.write(raw_dir / make_filename(code, speaker, index), y, sr, subtype="PCM_16")
                    n_written += 1
        print(f"  {code:<6} {words[code]}")
    print(f"Đã ghi {n_written} file vào {raw_dir}")
