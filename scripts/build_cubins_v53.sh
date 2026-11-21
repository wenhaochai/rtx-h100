#!/bin/bash
# build_cubins_v53.sh <v53 tree>/intern/cycles <out dir> <exact|exact_nofold|strict> [scene ...]
# Cubins of the OptiX-route exact and strict configurations built on the rtxpro6000-blender Blender 5.3.0 tree:
# v5.3.0 + rtxpro6000-blender/blender-5.3.0/patches/v53-all.patch + patches/v53/x1-shading.diff + x2-kdata.diff.
# Writes <out>/<scene>/kernel_sm_120.cubin.zst for CYCLES_CUDA_CUBIN of that project's host (build-v53all).
#   exact        no SVM specialization; SVM node pruning, hot-function inlining, per-scene closure count, compile-time
#                KernelData (configs/pro6000-v53/<scene>/kdata_const_exact.h); classroom: CUDA 13.4, shade 256 x 2
#   exact_nofold the same without compile-time KernelData
#   strict       + full SVM specialization (svm_specialized.h; masks 7 on junkshop/classroom, MONSTER_MASKS on monster,
#                default unset), kdata_const_strict.h; CUDA 13.4 front end (PTX .version rewritten to 8.8) + 12.9 ptxas,
#                as that project's B7 (not a combination NVIDIA supports)
set -euo pipefail
K=$(cd "${1:?tree}" && pwd); O=${2:?out}; CFG=${3:?config}; shift 3
SC=("$@"); [ ${#SC[@]} = 0 ] && SC=(monster junkshop classroom)
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/configs/pro6000-v53
declare -A F=([monster]=133147 [junkshop]=4859907 [classroom]=6817803) MC=([monster]=24 [junkshop]=14 [classroom]=12)
one() {
  local s=$1 tc=129 d=$O/$1 fl
  fl=(-D CCL_NAMESPACE_BEGIN= -D CCL_NAMESPACE_END= -D NVCC -D _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -I "$K" -std=c++17
      --use_fast_math -Wno-deprecated-gpu-targets -D WITH_NANOVDB --jump-table-density 80
      -D__KERNEL_FEATURES__=${F[$s]} -D__MAX_CLOSURE__=${MC[$s]} -DCYCLES_SVM_NODES_USED -I"$C/$s" -DCYCLES_INLINE_HOT=2)
  case $CFG in
    exact|exact_nofold)
      [ $s = classroom ] && { tc=134; fl+=(-DCYCLES_KB_SHADE_SURFACE_BLOCK=256 -DCYCLES_KB_SHADE_SURFACE_MINB=2); }
      [ $CFG = exact ] && fl+=(-DCYCLES_KDATA_CONST "-DCYCLES_KDATA_CONST_HEADER=\"$C/$s/kdata_const_exact.h\"") ;;
    strict)
      tc=mix; fl+=(-DCYCLES_SVM_SPECIALIZED -DCYCLES_KDATA_CONST "-DCYCLES_KDATA_CONST_HEADER=\"$C/$s/kdata_const_strict.h\"")
      if [ $s = monster ]; then [ -n "${MONSTER_MASKS:-}" ] && fl+=(-DCYCLES_SVM_SPECIALIZE_MASKS=$MONSTER_MASKS)
      else fl+=(-DCYCLES_SVM_SPECIALIZE_MASKS=7); fi ;;
    *) echo "unknown configuration $CFG"; exit 1 ;;
  esac
  mkdir -p "$d"; local cu=$K/kernel/device/cuda/kernel.cu
  case $tc in
    129|134) /usr/local/cuda-${tc:0:2}.${tc:2}/bin/nvcc -arch=sm_120 --cubin "$cu" "${fl[@]}" --ptxas-options=-v -o "$d/kernel_sm_120.cubin" > "$d/ptxas.log" 2>&1 ;;
    mix) /usr/local/cuda-13.4/bin/nvcc -arch=sm_120 --ptx "$cu" "${fl[@]}" -o "$d/k.ptx" > "$d/ptxas.log" 2>&1
         sed -i '0,/^\.version 9\.[0-9]$/s//.version 8.8/' "$d/k.ptx"
         /usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -m64 -v "$d/k.ptx" -o "$d/kernel_sm_120.cubin" >> "$d/ptxas.log" 2>&1; rm -f "$d/k.ptx" ;;
  esac || { echo "$s failed: $d/ptxas.log"; return 1; }
  zstd -q -f "$d/kernel_sm_120.cubin" -o "$d/kernel_sm_120.cubin.zst"
  echo "$(sha256sum "$d/kernel_sm_120.cubin" | cut -d' ' -f1)  $s/kernel_sm_120.cubin"
}
export -f one; export K O CFG C MONSTER_MASKS; export F_monster=1
for s in "${SC[@]}"; do one "$s" & done; wait
