#!/bin/bash
# fetch.sh <dir>   (needs internet; ~12.4 GB)
# 1. Blender source at the pinned v5.3.0 commit with its Linux libraries (lib/linux_x64, git LFS) -> <dir>/blender
# 2. The official 5.3.0 Linux binary (the stock baseline)                                         -> <dir>/blender-5.3.0-linux-x64
# 3. Open Data benchmark launcher 3.3.0, its Blender 5.3.0 scenes and benchmark script             -> <dir>/bench
# 4. The baked neural radiance caches (t24, rule c0) from Hugging Face                             -> <dir>/bakes/<scene>/bake_c0.nrc
#    and the nrc-hq caches (same model, baked offline by nrc/bake/stream/bake_stream.py)           -> <dir>/bakes_hq/<scene>/bake_c0.nrc
#    and the nrc-permuto caches (permutohedral lattice grid, bake_stream.py --enc permuto5)        -> <dir>/bakes_permuto/<scene>/bake_c0.nrc
#    and the nrc-scatter caches (scattered query, averaged targets, bake_stream.py --nrcinf)         -> <dir>/bakes_scatter/<scene>/bake_c0.nrc
set -euo pipefail
D=${1:?dir}; mkdir -p "$D"; D=$(cd "$D" && pwd)
HERE=$(cd "$(dirname "$0")/.." && pwd)
PIN=02f0899d78e374b63e7181d9061e1f7ab03419cc
[ -d "$D/blender/.git" ] || git clone --depth 1 --branch v5.3.0 https://projects.blender.org/blender/blender.git "$D/blender"
[ "$(git -C "$D/blender" rev-parse HEAD)" = $PIN ] || { echo "v5.3.0 is not $PIN"; exit 1; }
git -C "$D/blender" -c submodule.lib/linux_x64.update=checkout submodule update --init --depth 1 lib/linux_x64
(cd "$D/blender/lib/linux_x64" && git lfs install --local && git lfs pull)
if [ ! -x "$D/blender-5.3.0-linux-x64/blender" ]; then
  cd "$D"
  curl -fsSL -O https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz
  curl -fsSL -O https://download.blender.org/release/Blender5.3/blender-5.3.0.sha256
  grep linux-x64 blender-5.3.0.sha256 | sha256sum -c - || exit 1
  tar -xf blender-5.3.0-linux-x64.tar.xz; rm blender-5.3.0-linux-x64.tar.xz
fi
if [ ! -d "$D/bench/blender-benchmark-launcher/scenes" ]; then
  mkdir -p "$D/bench"; cd "$D/bench"
  curl -fsSL https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-3.3.0-linux.tar.gz | tar -xz
  XDG_CACHE_HOME=$D/bench ./benchmark-launcher-cli blender download 5.3.0
  XDG_CACHE_HOME=$D/bench ./benchmark-launcher-cli scenes download --blender-version 5.3.0 monster junkshop classroom
fi
for s in monster junkshop classroom; do
  f=$s/bake_c0.nrc; mkdir -p "$D/bakes/$s"
  (cd "$D/bakes" && grep " $f\$" "$HERE/configs/bakes.sha256" | sha256sum -c --status 2>/dev/null) && continue   # present and intact
  curl -fsSL -o "$D/bakes/$f" "https://huggingface.co/wchai/cycles-h100-nrc-bake/resolve/main/nrc_bakehi/${s}_t24/bake_c0.nrc"
done
(cd "$D/bakes" && sha256sum -c "$HERE/configs/bakes.sha256")
for s in monster junkshop classroom; do
  f=$s/bake_c0.nrc; mkdir -p "$D/bakes_hq/$s"
  (cd "$D/bakes_hq" && grep " $f\$" "$HERE/configs/bakes_hq.sha256" | sha256sum -c --status 2>/dev/null) && continue   # present and intact
  curl -fsSL -o "$D/bakes_hq/$f" "https://huggingface.co/wchai/cycles-h100-nrc-bake/resolve/main/nrc_hq/${s}_t24/bake_c0.nrc"
done
(cd "$D/bakes_hq" && sha256sum -c "$HERE/configs/bakes_hq.sha256")
for s in monster junkshop classroom; do
  f=$s/bake_c0.nrc; mkdir -p "$D/bakes_permuto/$s"
  (cd "$D/bakes_permuto" && grep " $f\$" "$HERE/configs/bakes_permuto.sha256" | sha256sum -c --status 2>/dev/null) && continue   # present and intact
  curl -fsSL -o "$D/bakes_permuto/$f" "https://huggingface.co/wchai/cycles-h100-nrc-bake/resolve/main/nrc_permuto/${s}_t24/bake_c0.nrc"
done
(cd "$D/bakes_permuto" && sha256sum -c "$HERE/configs/bakes_permuto.sha256")
for s in monster junkshop classroom; do
  f=$s/bake_c0.nrc; mkdir -p "$D/bakes_scatter/$s"
  (cd "$D/bakes_scatter" && grep " $f\$" "$HERE/configs/bakes_scatter.sha256" | sha256sum -c --status 2>/dev/null) && continue   # present and intact
  curl -fsSL -o "$D/bakes_scatter/$f" "https://huggingface.co/wchai/cycles-h100-nrc-bake/resolve/main/nrc_scatter/${s}_t24/bake_c0.nrc"
done
(cd "$D/bakes_scatter" && sha256sum -c "$HERE/configs/bakes_scatter.sha256")
echo "BENCH=$D/bench/blender-benchmark-launcher  BAKES=$D/bakes  BAKES_HQ=$D/bakes_hq  BAKES_PERMUTO=$D/bakes_permuto  BAKES_SCATTER=$D/bakes_scatter  stock=$D/blender-5.3.0-linux-x64/blender"
