#!/usr/bin/env bash
# Download COCO 2017 images + keypoint annotations into data/coco/.
# Resumable. Uses aria2c (16 parallel segments) when installed, since single
# connections to the COCO server are often throttled; falls back to wget -c.
# Archive integrity is verified by unzip's CRC check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/coco}"
BASE="http://images.cocodataset.org"
FILES=(
  "annotations/annotations_trainval2017.zip"
  "zips/val2017.zip"
  "zips/train2017.zip"
)

mkdir -p "$DEST/zips"
for f in "${FILES[@]}"; do
  name="$(basename "$f")"
  echo "==> $name"
  if command -v aria2c >/dev/null; then
    aria2c -c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=60 \
      --console-log-level=warn -d "$DEST/zips" -o "$name" "$BASE/$f"
  else
    wget -c -q --show-progress -O "$DEST/zips/$name" "$BASE/$f"
  fi
done

for f in "${FILES[@]}"; do
  name="$(basename "$f")"
  marker="$DEST/.${name%.zip}.done"
  if [[ -f "$marker" ]]; then
    echo "==> $name already extracted"
    continue
  fi
  echo "==> extracting $name"
  unzip -q -o "$DEST/zips/$name" -d "$DEST"
  touch "$marker"
done

echo "COCO ready in $DEST"
echo "You may delete $DEST/zips to reclaim ~19 GB."
