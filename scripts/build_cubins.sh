#!/bin/bash
# build_cubins.sh <kernel tree>/intern/cycles <out dir> [scene ...]
# One sm_90 cubin per scene from configs/<scene>/nrc_nvcc_flags.txt (CUDA 12.9 nvcc; 5-15 min per cubin, three in parallel).
# Prints the sha256 of each cubin; the nrc cubins are listed in configs/nrc_cubins.sha256.
# FLAGS=exact_nvcc_flags.txt builds the exact configuration (tree from kernel_tree.sh ... 03; configs/exact_cubins.sha256).
# FLAGS=exact_nvcc_flags.txt EXTRA=-DCYCLES_NRC builds the bake cubins (tree from kernel_tree.sh ... bake;
# configs/bake_cubins.sha256): the exact configuration plus the cache's training path.
# SCATTER=1 builds the nrc-scatter cubins (configs/scatter_cubins.sha256): the nrc flags with -DNRC2_FLAGS_C=1, the
# scattered form of the cache query, and the seed read at run time (the seed line dropped from kdata_const.h).
set -euo pipefail
K=${1:?kernel tree}; O=${2:?out dir}; shift 2; SCENES=("${@:-monster junkshop classroom}"); SCENES=(${SCENES[@]})
HERE=$(cd "$(dirname "$0")/.." && pwd)
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
mkdir -p "$O"
one() {
  local s=$1 cfg=$HERE/configs/$1 flags sub=()
  if [ "${SCATTER:-0}" = 1 ]; then
    mkdir -p "$O/configs"; rm -rf "$O/configs/$s"; cp -r "$cfg" "$O/configs/$s"; cfg=$O/configs/$s
    sed -i '/^  static constexpr int seed = 0;$/d' "$cfg/kdata_const.h"; sub=(-e 's#^-DNRC2_FLAGS_C=0$#-DNRC2_FLAGS_C=1#')
  fi
  mapfile -t flags < <(sed -e "s#@CFG@#$cfg#g" "${sub[@]}" "$cfg/${FLAGS:-nrc_nvcc_flags.txt}")   # one flag per line; quotes stay literal
  "$NVCC" -arch=sm_90 --cubin "$K/kernel/device/cuda/kernel.cu" -D CCL_NAMESPACE_BEGIN= -D CCL_NAMESPACE_END= \
    -D NVCC -m 64 -I "$K" -I "$K/kernel/device/cuda" --use_fast_math -std=c++17 -D WITH_NANOVDB -Wno-deprecated-gpu-targets \
    --ptxas-options=-v "${flags[@]}" ${EXTRA:-} -o "$O/$s.cubin" > "$O/$s.ptxas.log" 2>&1 || { echo "$s failed:"; grep -m5 error "$O/$s.ptxas.log" || tail -5 "$O/$s.ptxas.log"; return 1; }
  echo "$(sha256sum "$O/$s.cubin" | cut -d' ' -f1)  $s.cubin"
}
export -f one; export K O HERE NVCC FLAGS EXTRA SCATTER
printf '%s\n' "${SCENES[@]}" | xargs -P 3 -I{} bash -c 'one {}'
