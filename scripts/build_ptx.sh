#!/bin/bash
# build_ptx.sh <kernel tree>/intern/cycles <out dir> [scene ...]   per-scene OptiX PTX for the OptiX route
# The tree needs patches/optix/kernel.diff (and kernel-exact.diff for an exact tree). Flags: the scene's
# <FLAGS> file (default nrc_nvcc_flags.txt; CONFIGS picks another configs dir) without the CUDA-only ones: launch
# bounds, BVH2 dynamic fetch, BVH4, flat triangles, FMA slab test, shadow child order, compile-time KernelData,
# SSS dynamic fetch and per-material programs (OptiX traces with its own BVH; shading runs in the cubin).
# Writes <out>/<scene>/kernel_optix.ptx.zst for CYCLES_OPTIX_PTX. Needs the OptiX 8 SDK (OPTIX_ROOT) and CUDA 12.9.
set -euo pipefail
K=$(cd "${1:?kernel tree}" && pwd); O=${2:?out dir}; shift 2; SCENES=("${@:-monster junkshop classroom}"); SCENES=(${SCENES[@]})
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$(cd "${CONFIGS:-$HERE/configs}" && pwd)
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}; : "${OPTIX_ROOT:?OptiX SDK}"
mkdir -p "$O"; O=$(cd "$O" && pwd)
for s in "${SCENES[@]}"; do
  mapfile -t F < <(sed "s#@CFG@#$C/$s#g" "$C/$s/${FLAGS:-nrc_nvcc_flags.txt}" | grep -v -E \
    "ISECT_|SHADE_|SHADOW_MINBLOCKS|CLOSEST_MINBLOCKS|SSS_BLOCK|SSS_MINBLOCKS|BVH2DF|BVH_FMA|BVH4|FLATTRI|KDATA|SHADOW_ORDER|SSSDF|PERMAT2")
  mkdir -p "$O/$s"
  (cd "$K/kernel/device/optix" && "$NVCC" --ptx -arch=compute_75 -I "$OPTIX_ROOT/include" -I "$K" -std=c++17 --use_fast_math \
    -Wno-deprecated-gpu-targets -D WITH_NANOVDB "${F[@]}" kernel.cu -o "$O/$s/kernel_optix.ptx")
  zstd -q -f --rm "$O/$s/kernel_optix.ptx" -o "$O/$s/kernel_optix.ptx.zst"
  echo "$s: $O/$s/kernel_optix.ptx.zst"
done
