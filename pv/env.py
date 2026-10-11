"""Import first in every pipeline script: keeps model and browser caches inside the project (C: is nearly full)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"

for var, sub in {"TORCH_HOME": "torch", "HF_HOME": "hf", "PLAYWRIGHT_BROWSERS_PATH": "ms-playwright"}.items():
    os.environ.setdefault(var, str(CACHE / sub))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))   # openai-whisper -> .cache/whisper
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # exFAT has no symlinks; the HF cache copies files
