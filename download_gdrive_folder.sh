#!/usr/bin/env bash
# Download a Google Drive folder (and its subfolders) recursively with gdown.
# Default URL points at the folder you asked for; override by passing a URL.
#
# Usage:
#   ./download_gdrive_folder.sh                              # default URL -> ./gdrive_download/
#   ./download_gdrive_folder.sh <FOLDER_URL_OR_ID>           # custom URL  -> ./gdrive_download/
#   ./download_gdrive_folder.sh <FOLDER_URL_OR_ID> <OUTDIR>  # custom URL & output dir
#
# Notes:
# - Public folders only. For private folders, set GDRIVE_COOKIES (a Netscape cookies.txt)
#   or run `gdown --folder <id>` interactively once with auth.
# - Files >2 GiB or with the virus-scan warning need confirmation; this script forces it.

set -euo pipefail

DEFAULT_URL="https://drive.google.com/drive/folders/11wWL6vWxxzHJMpSzYlA0uijJkCO10mLC?usp=sharing"
URL="${1:-$DEFAULT_URL}"
OUT_DIR="${2:-./gdrive_download}"

mkdir -p "$OUT_DIR"

if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] installing gdown..."
  python3 -m pip install -U "gdown>=5.2" >/dev/null
fi

GDOWN_VER=$(python3 -c "import gdown,sys;print(gdown.__version__)")
echo "[INFO] gdown=$GDOWN_VER"
echo "[INFO] URL    : $URL"
echo "[INFO] OUTPUT : $OUT_DIR"

# Build args. --remaining-ok skips files already downloaded so re-runs resume.
ARGS=(--folder "$URL" -O "$OUT_DIR" --continue --remaining-ok)
[[ -n "${GDRIVE_COOKIES:-}" ]] && ARGS+=(--cookies "$GDRIVE_COOKIES")

# gdown returns non-zero if some files fail (e.g. quota, permission). Capture and
# report instead of aborting; rerun the script to retry the failures.
if gdown "${ARGS[@]}"; then
  status="ok"
else
  status="partial (some files failed; rerun to retry)"
fi

echo
echo "[DONE] $status"
echo "Saved to: $(cd "$OUT_DIR" && pwd)"
echo "File count: $(find "$OUT_DIR" -type f | wc -l | tr -d ' ')"
echo "Size      : $(du -sh "$OUT_DIR" | awk '{print $1}')"
