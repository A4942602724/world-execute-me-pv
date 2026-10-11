# Dot-source before working in this project:  . .\env.ps1
# Activates .venv and keeps every cache on this drive (C: is nearly full).
$P = $PSScriptRoot
$env:PIP_CACHE_DIR = "$P\.cache\pip"
$env:TMP = "$P\.cache\tmp"
$env:TEMP = "$P\.cache\tmp"
$env:TORCH_HOME = "$P\.cache\torch"
$env:HF_HOME = "$P\.cache\hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"   # exFAT has no symlinks; the HF cache copies files instead
$env:XDG_CACHE_HOME = "$P\.cache"          # openai-whisper models -> .cache\whisper
$env:PLAYWRIGHT_BROWSERS_PATH = "$P\.cache\ms-playwright"
$env:PYTHONUTF8 = "1"
. "$P\.venv\Scripts\Activate.ps1"
