"""Audio analysis: stems, beat grid, onsets, energy envelopes, sections and vocal phrases.

    python pv/analyze.py     -> work/song.wav, work/stems/{drums,bass,other,vocals}.wav,
                                work/analysis.json (events), work/envelopes.npz (per-stem RMS and onset strength)

Time zero is the first sample ffmpeg decodes from the song in input/; the final mux decodes the same file.
"""
import env

import json
import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch

INPUT = env.ROOT / "input"
WORK = env.ROOT / "work"
STEMS = WORK / "stems"
SR = 22050
HOP = 256                                  # envelope rate: SR / HOP = 86.13 frames per second
NAMES = ("drums", "bass", "other", "vocals")


def song_path():
    songs = sorted(p for p in INPUT.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"})
    if len(songs) != 1:
        raise SystemExit(f"expected exactly one song in {INPUT}, found {[p.name for p in songs]}")
    return songs[0]


def decode(src):
    out = WORK / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le",
                    str(out)], check=True)
    return out


def separate(wav_path):
    """htdemucs_ft (a bag of four fine-tuned models): slower than htdemucs, cleaner vocals for the lyric alignment."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_ft").eval()
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    assert sr == model.samplerate, (sr, model.samplerate)
    mix = torch.from_numpy(wav.T.copy()).cuda()
    ref = mix.mean(0)
    mix = (mix - ref.mean()) / ref.std()
    with torch.no_grad():
        stems = apply_model(model, mix[None], device="cuda", shifts=2, split=True, overlap=0.25, progress=False)[0]
    stems = (stems * ref.std() + ref.mean()).cpu().numpy()
    STEMS.mkdir(parents=True, exist_ok=True)
    for name, s in zip(model.sources, stems):
        sf.write(STEMS / f"{name}.wav", s.T, sr, subtype="FLOAT")


def runs(mask, fps, merge_gap=0.25, min_len=0.2):
    """[start, end] seconds of the True runs in a frame mask, short gaps merged, short runs dropped."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    spans = [[s / fps, e / fps] for s, e in zip(edges[::2], edges[1::2])]
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in merged if e - s >= min_len]


def analyse():
    fps = SR / HOP
    y = {n: librosa.load(STEMS / f"{n}.wav", sr=SR, mono=True)[0] for n in NAMES}
    y["mix"] = librosa.load(WORK / "song.wav", sr=SR, mono=True)[0]
    duration = len(y["mix"]) / SR

    rms = {n: librosa.feature.rms(y=v, frame_length=1024, hop_length=HOP)[0] for n, v in y.items()}
    oenv = {n: librosa.onset.onset_strength(y=v, sr=SR, hop_length=HOP) for n, v in y.items()}

    # kick and snare/hat from the drum stem's low and high mel bands
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y["drums"], sr=SR, hop_length=HOP, n_mels=64))
    freqs = librosa.mel_frequencies(n_mels=64, fmax=SR / 2)
    oenv["kick"] = librosa.onset.onset_strength(S=mel[freqs < 150], sr=SR, hop_length=HOP)
    oenv["snare"] = librosa.onset.onset_strength(S=mel[freqs > 2000], sr=SR, hop_length=HOP)

    onsets = {n: np.round(librosa.onset.onset_detect(onset_envelope=e, sr=SR, hop_length=HOP, units="time"), 3).tolist()
              for n, e in oenv.items()}

    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv["mix"], sr=SR, hop_length=HOP)
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    # downbeat phase: the beat position (mod 4) carrying the most kick + bass onset energy
    low = oenv["kick"] / (oenv["kick"].max() + 1e-9) + oenv["bass"] / (oenv["bass"].max() + 1e-9)
    phase_energy = [float(low[beat_frames[p::4]].mean()) for p in range(4)]
    phase = int(np.argmax(phase_energy))

    # rough sections: agglomerative clustering of beat-synchronous chroma + MFCC
    chroma = librosa.feature.chroma_cqt(y=y["mix"], sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y["mix"], sr=SR, n_mfcc=20, hop_length=HOP)
    feats = np.vstack([librosa.util.sync(chroma, beat_frames, aggregate=np.median),
                       librosa.util.normalize(librosa.util.sync(mfcc, beat_frames), axis=1)])
    bounds = librosa.segment.agglomerative(feats, k=6)
    bound_frames = librosa.util.fix_frames(beat_frames[np.clip(bounds - 1, 0, len(beat_frames) - 1)][1:], x_min=0)
    section_starts = sorted(set(np.round(librosa.frames_to_time(bound_frames, sr=SR, hop_length=HOP), 2).tolist()))
    sections = [[s, e] for s, e in zip(section_starts, section_starts[1:] + [round(duration, 2)])]

    vdb = librosa.amplitude_to_db(rms["vocals"], ref=np.max)
    phrases = runs(vdb > -30, fps)

    analysis = dict(
        song=song_path().name, duration=round(duration, 3), tempo=round(float(np.atleast_1d(tempo)[0]), 2),
        beats=np.round(beats, 3).tolist(), downbeat_phase=phase, phase_energy=np.round(phase_energy, 3).tolist(),
        downbeats=np.round(beats[phase::4], 3).tolist(), sections=sections, vocal_phrases=phrases, onsets=onsets,
        stem_rms_mean={n: round(float(r.mean()), 4) for n, r in rms.items()},
    )
    (WORK / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(WORK / "envelopes.npz", fps=fps, **{f"rms_{n}": r for n, r in rms.items()},
                        **{f"onset_{n}": e for n, e in oenv.items()})
    return analysis


def main():
    WORK.mkdir(exist_ok=True)
    wav = decode(song_path())
    if not (STEMS / "vocals.wav").exists():
        separate(wav)
    a = analyse()
    print(f"{a['song']}: {a['duration']} s, tempo {a['tempo']} BPM, {len(a['beats'])} beats, "
          f"downbeat phase {a['downbeat_phase']} {a['phase_energy']}")
    print("stem mean RMS:", a["stem_rms_mean"])
    print("sections:", a["sections"])
    print(f"vocal phrases ({len(a['vocal_phrases'])}):", a["vocal_phrases"])
    print("onset counts:", {n: len(v) for n, v in a["onsets"].items()})


if __name__ == "__main__":
    main()
