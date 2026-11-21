#!/bin/bash
# build_host.sh <blender checkout> <out dir>      PATCH=patches/bake/host.diff builds the bake host instead
# Headless Blender 5.3.0 with patches/host.diff, no CUDA binaries (the cubins come from build_cubins.sh).
# <blender checkout>: a clone at the pinned commit with its precompiled libraries in lib/linux_x64
# (scripts/fetch.sh). Result: <out dir>/install/blender. ~1 h on 32 cores, ~30 GB of disk.
# Needs GCC >= 14 (GCC_TOOLSET_DIR, default /opt/rh/gcc-toolset-14/root/usr) and GL/EGL headers (CPATH).
set -euo pipefail
SRC=$(cd "${1:?blender checkout}" && pwd); OUT=${2:?out dir}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")/.." && pwd)
PIN=02f0899d78e374b63e7181d9061e1f7ab03419cc
[ -d "$SRC/lib/linux_x64" ] || { echo "no $SRC/lib/linux_x64"; exit 1; }
GT=${GCC_TOOLSET_DIR:-/opt/rh/gcc-toolset-14/root/usr}
# CC/CXX override the toolset's gcc/g++ (e.g. an unpacked gcc-14 whose binaries are named gcc-14/g++-14);
# GCC_RUNTIME: a directory with that GCC's libstdc++.so.6, put on LD_LIBRARY_PATH and copied into the install.
export PATH=$GT/bin:$PATH LD_LIBRARY_PATH=${GCC_RUNTIME:+$GCC_RUNTIME:}$GT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CC=${CC:-$GT/bin/gcc} CXX=${CXX:-$GT/bin/g++}
$CXX --version | head -1

# 1. source: the checkout without .git and lib (linked), intern/cycles reset to the pinned commit, then host.diff
S=$OUT/src
if [ ! -e "$S/.patched" ]; then
  rm -rf "$S"; mkdir -p "$S"
  rsync -a --exclude /.git --exclude /lib --exclude /build --exclude /logs "$SRC/" "$S/"
  ln -s "$SRC/lib" "$S/lib"
  rm -rf "$S/intern/cycles"; git -C "$SRC" archive "$PIN" intern/cycles | tar -x -C "$S"
  # inside another git work tree `git apply` skips every file and exits 0: stop discovery, then check the reverse
  P=$HERE/${PATCH:-patches/host.diff}
  (cd "$S" && export GIT_CEILING_DIRECTORIES=$OUT && git apply --whitespace=nowarn "$P" && git apply --check -R "$P") \
    || { echo "patch did not apply: $P"; exit 1; }
  # PATCH2: a second patch on top (patches/optix/host.diff for the OptiX route)
  for P2 in ${PATCH2:-}; do
    P2=$HERE/$P2
    (cd "$S" && export GIT_CEILING_DIRECTORIES=$OUT && git apply --whitespace=nowarn "$P2" && git apply --check -R "$P2") \
      || { echo "patch did not apply: $P2"; exit 1; }
  done
  touch "$S/.patched"
fi

# 2. configure and build
# OPTIX_ROOT=<OptiX SDK>: also build the OptiX device (RTX GPUs); its kernels then come from CYCLES_OPTIX_PTX or are
# compiled at run time. Off by default, as on the H100, which has no RT cores.
B=$OUT/build
if [ -n "${OPTIX_ROOT:-}" ]; then
  OPTIX_ARGS="-DWITH_CYCLES_DEVICE_OPTIX=ON -DOPTIX_ROOT_DIR=$OPTIX_ROOT -DCYCLES_RUNTIME_OPTIX_ROOT_DIR=$OPTIX_ROOT -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-/usr/local/cuda-12.9}"
else OPTIX_ARGS="-DWITH_CYCLES_DEVICE_OPTIX=OFF"; fi
cmake -S "$S" -B "$B" -C "$S/build_files/cmake/config/blender_headless.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CYCLES_CUDA_BINARIES=OFF $OPTIX_ARGS \
  -DWITH_CYCLES_DEVICE_HIP=OFF -DWITH_CYCLES_DEVICE_ONEAPI=OFF -DWITH_CYCLES_EMBREE=ON \
  -DWITH_USD=OFF -DWITH_HYDRA=OFF \
  -DWITH_INSTALL_PORTABLE=ON -DCMAKE_INSTALL_PREFIX="$OUT/install" 2>&1 | tail -5
# USD and Hydra are off: Cycles needs neither, and their headers break where a system TBB sits in /usr/include.
cmake --build "$B" -j "${BUILD_JOBS:-$(nproc)}" > "$OUT/build.log" 2>&1 || { grep -m20 -E "error:|Error [0-9]" "$OUT/build.log"; exit 1; }
cmake --install "$B" > /dev/null
[ -z "${GCC_RUNTIME:-}" ] || cp -L "$GCC_RUNTIME/libstdc++.so.6" "$OUT/install/lib/libstdc++.so.6"
"$OUT/install/blender" --version | head -1
echo "install: $OUT/install"
