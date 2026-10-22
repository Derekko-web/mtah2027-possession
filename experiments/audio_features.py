"""Local whistle/crowd spectral features aligned to tactical-video frame time.

No audio is uploaded. ffmpeg decodes mono 16 kHz PCM in memory; only derived
features are saved under ignored data/. Frame i is centered at i/25 seconds.
Observation context is at most 9 seconds plus the 64 ms FFT half-window.
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import asset_paths

SAMPLE_RATE = 16000
FFT_SIZE = 2048
FPS = 25
MAX_CONTEXT_SECONDS = 9.064


def spectral_features(samples, n_frames):
    samples = np.asarray(samples, dtype=np.float32)
    hop = SAMPLE_RATE//FPS
    expected = (n_frames-1)*hop+FFT_SIZE//2+1
    if len(samples) < expected:
        raise ValueError("Audio ended before requested feature context")
    frequencies, times, spectrum = stft(samples, fs=SAMPLE_RATE, nperseg=FFT_SIZE,
                                        noverlap=FFT_SIZE-hop, boundary="zeros", padded=False)
    if len(times) < n_frames or not np.allclose(times[:n_frames], np.arange(n_frames)/FPS):
        raise ValueError("Audio/video feature alignment failed")
    power = np.abs(spectrum[:, :n_frames]).astype(np.float64)**2
    total = power.sum(axis=0)+1e-20
    result = {"audio_log_energy": np.log(total)}
    for low, high in ((100, 1000), (1000, 2000), (2000, 4000), (4000, 7000)):
        band = power[(frequencies >= low) & (frequencies < high)]
        result[f"audio_fraction_{low}_{high}"] = band.sum(axis=0)/total
    whistle = power[(frequencies >= 1500) & (frequencies <= 4500)]
    mass = whistle.sum(axis=0)+1e-20
    result["audio_whistle_peak_share"] = whistle.max(axis=0)/mass
    result["audio_whistle_concentration"] = np.sort(whistle, axis=0)[-5:].sum(axis=0)/mass
    result["audio_whistle_flatness"] = np.exp(np.log(whistle+1e-20).mean(axis=0))/(whistle.mean(axis=0)+1e-20)
    result["audio_whistle_energy_fraction"] = mass/total
    base = pd.DataFrame(result)
    extra = {}
    for name in ("audio_log_energy", "audio_whistle_peak_share", "audio_whistle_concentration",
                 "audio_whistle_flatness", "audio_whistle_energy_fraction"):
        column = base[name]
        for seconds in (1, 3, 9):
            extra[f"{name}_past_{seconds}s_mean"] = column.rolling(seconds*FPS, min_periods=seconds*FPS).mean()
            extra[f"{name}_past_{seconds}s_max"] = column.rolling(seconds*FPS, min_periods=seconds*FPS).max()
        extra[f"{name}_future_1s_mean"] = column[::-1].rolling(FPS, min_periods=FPS).mean()[::-1]
    return base.join(pd.DataFrame(extra)).astype(np.float32)


def build_audio_features(n_frames, video=None):
    video = Path(video or asset_paths()["video"])
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(video)]))
    streams = metadata["streams"]
    if not any(s["codec_type"] == "audio" for s in streams):
        raise ValueError("Video has no audio")
    # This dataset starts both streams at zero; refuse silent misalignment on a
    # different source. aresample also preserves initial timestamps explicitly.
    if any(abs(float(s.get("start_time", 0))) > 1/SAMPLE_RATE for s in streams):
        raise ValueError("Nonzero media timestamps need explicit audio/video alignment")
    seconds = n_frames/FPS+1
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(video), "-t", str(seconds),
        "-vn", "-af", f"aresample={SAMPLE_RATE}:async=1:first_pts=0", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"], check=True, capture_output=True)
    return spectral_features(np.frombuffer(result.stdout, dtype="<f4"), n_frames)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=18685)
    parser.add_argument("--out", type=Path, default=Path("data/stoppage_research/audio_inputs.pkl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features = build_audio_features(args.frames)
    features.to_pickle(args.out)
    print("Saved local audio features", features.shape, "to", args.out)
