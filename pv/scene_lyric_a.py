"""The ink-and-code pond with lyric style A, "cinema" (see lyrics.py)."""
import scene_pond2
from scene_pond2 import T0, T1, static  # noqa: F401  (render.py reads these)


def frame(t):
    return scene_pond2.frame(t, style="cinema")
