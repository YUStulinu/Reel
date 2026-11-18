#!/usr/bin/env bash
# Downloads the Xiph test clips (media.xiph.org/video/derf) into data/clips/.
#   scripts/fetch_clips.sh [clip ...]     (default: all six used by eval/evaluate.py)
set -e
cd "$(dirname "$0")/.."
mkdir -p data/clips
clips=("$@")
[ ${#clips[@]} -eq 0 ] && clips=(foreman_cif akiyo_cif mobile_cif news_cif park_joy_420_720p50 in_to_tree_420_720p50)
for c in "${clips[@]}"; do
  [ -s "data/clips/$c.y4m" ] && continue
  curl -fsSL -o "data/clips/$c.y4m" "https://media.xiph.org/video/derf/y4m/$c.y4m"
  echo "data/clips/$c.y4m"
done
