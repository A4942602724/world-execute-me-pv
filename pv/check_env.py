"""Environment check for the audio-analysis side: GPU, beat tracking, Whisper, Demucs.

    .venv/Scripts/python.exe pv/check_env.py
Uses a synthetic click track (120 BPM), so no song is needed. Downloads whisper `tiny` and demucs `htdemucs` once.
"""
import env  # noqa: F401  (cache locations)

import numpy as np
import torch

SR = 44100


def click_track(bpm=120.0, seconds=8.0):
    y = np.zeros(int(SR * seconds), np.float32)
    burst = (np.random.default_rng(0).standard_normal(int(0.02 * SR)) * np.hanning(int(0.02 * SR))).astype(np.float32)
    for t in np.arange(0.5, seconds, 60 / bpm):
        i = int(t * SR)
        y[i:i + burst.size] += burst[: y.size - i]
    return y


def main() -> None:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}")

    import librosa
    y = click_track()
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR)
    print(f"librosa {librosa.__version__}  tempo={float(np.atleast_1d(tempo)[0]):.1f} BPM (expect ~120)  beats={len(beats)}")

    import stable_whisper
    model = stable_whisper.load_model("tiny", device="cuda")
    res = model.transcribe(y, verbose=None)
    print(f"stable-ts {stable_whisper.__version__}  whisper tiny on {next(model.parameters()).device}: ok "
          f"({len(res.segments)} segments on a click track)")

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    sep = get_model("htdemucs").cuda().eval()
    mix = torch.from_numpy(np.stack([y, y])[None]).cuda()
    with torch.no_grad():
        stems = apply_model(sep, mix, device="cuda", progress=False)
    print(f"demucs htdemucs: stems {sep.sources} shape {tuple(stems.shape)}")

    import torchaudio
    has_mms = hasattr(torchaudio.pipelines, "MMS_FA")
    print(f"torchaudio {torchaudio.__version__}  MMS_FA pipeline: {'yes' if has_mms else 'no (use stable-ts alignment)'}")
    print("check_env: ok")


if __name__ == "__main__":
    main()
