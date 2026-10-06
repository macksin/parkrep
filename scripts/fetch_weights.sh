#!/usr/bin/env bash
# Download LocoTrack-S weights (Apache-2.0, ~33 MB) and verify the checksum.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
url="https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt"
sha="da023594e6d6c05ecad9644efc1467545481cfa899e20730bd9fdce778ffa5ac"
[ -f weights/locotrack_small.ckpt ] || curl -L --fail -o weights/locotrack_small.ckpt "$url"
echo "$sha  weights/locotrack_small.ckpt" | shasum -a 256 -c -
