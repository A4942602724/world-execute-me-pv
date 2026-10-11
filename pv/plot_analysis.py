"""Timeline picture of the analysis: energy per stem, kick/snare onsets, beat grid, lyric characters, sections.

    python pv/plot_analysis.py      -> work/analysis.png
Lyric lines are labelled by number only.
"""
import env

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = env.ROOT / "work"


def main():
    a = json.loads((WORK / "analysis.json").read_text(encoding="utf-8"))
    lyr = json.loads((WORK / "lyrics_timing.json").read_text(encoding="utf-8"))["lines"]
    e = np.load(WORK / "envelopes.npz")
    fps = float(e["fps"])
    t = np.arange(len(e["rms_mix"])) / fps
    db = lambda x: np.clip(20 * np.log10(x / x.max() + 1e-9), -45, 0)

    halves = [(0, a["duration"] / 2 + 1), (a["duration"] / 2 - 1, a["duration"])]
    fig, axes = plt.subplots(2, 1, figsize=(26, 11), dpi=80)
    for ax, (t0, t1) in zip(axes, halves):
        for s, _ in a["sections"]:
            ax.axvline(s, color="#c03", lw=2, alpha=0.6)
        for b in a["beats"]:
            ax.axvline(b, color="#999", lw=0.5, alpha=0.5)
        for b in a["downbeats"]:
            ax.axvline(b, color="#555", lw=1.2, alpha=0.7)
        ax.plot(t, db(e["rms_drums"]) + 45, color="#e08a00", lw=1, label="drums dB")
        ax.plot(t, db(e["rms_vocals"]) + 45, color="#2a6fdb", lw=1.2, label="vocals dB")
        ax.plot(t, db(e["rms_bass"]) + 45, color="#3a3", lw=0.8, alpha=0.7, label="bass dB")
        for k, y, c in (("kick", -6, "#a40"), ("snare", -10, "#d70")):
            on = np.array(a["onsets"][k])
            ax.scatter(on, np.full(len(on), y), marker="|", s=80, color=c, label=k)
        von = np.array(a["onsets"]["vocals"])
        ax.scatter(von, np.full(len(von), -14), marker="|", s=80, color="#2a6fdb", label="vocal onsets")
        for ln in lyr:
            ax.hlines(-20, ln["start"], ln["end"], color="#123", lw=6)
            ax.text(ln["start"], -25, str(ln["i"]), fontsize=13, weight="bold")
            ax.scatter([c["s"] for c in ln["chars"]], [-20] * len(ln["chars"]), marker="|", s=160, color="#fff",
                       zorder=3)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-28, 47)
        ax.set_xticks(np.arange(np.ceil(t0), t1, 1))
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(loc="upper right", ncol=7)
    axes[0].set_title(f"{a['song']}  {a['tempo']} BPM  (grey: beats, dark: downbeats, red: rough sections; "
                      f"bars: lyric lines, white ticks: characters)")
    fig.tight_layout()
    fig.savefig(WORK / "analysis.png")

    # how far each sung character lands from the nearest vocal onset
    von = np.array(a["onsets"]["vocals"])
    d = np.array([np.min(np.abs(von - c["s"])) for ln in lyr for c in ln["chars"]])
    print(f"char start to nearest vocal onset: median {np.median(d) * 1000:.0f} ms, "
          f"p75 {np.percentile(d, 75) * 1000:.0f} ms, p90 {np.percentile(d, 90) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
