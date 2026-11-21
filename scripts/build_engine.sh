#!/bin/bash
# build_engine.sh <out dir>
# The neural radiance cache engine that Cycles loads with dlopen (CYCLES_NRC_ENGINE=<out>/libcycles_nrc.so).
# tiny-cuda-nn is pinned and patched (nrc/patches): only the 3D hash grid and the
# 64-neuron fully fused MLP are compiled, and the weights can be saved, loaded and frozen.
# The fused inference kernel (nrc/src/nrc_fused.cu) uses wgmma (sm_90a); run_wgmma.json selects it at run time.
# The frozen inference engine (nrc/src/nrc_infer.cu, sm_90a) runs NRCINF1 caches, hash or permuto; run_infer.json selects it.
# Other GPUs run the fused kernel's mma.sync core (run_mma.json); for the RTX PRO 6000: ARCHS=120 FUSED_ARCHS=120-real.
# Step 1 needs internet; step 2 runs anywhere with CUDA 12.9 (~5 GB per nvcc, ~30 min on 8 cores).
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")/.." && pwd); N=$HERE/nrc
CU=${CUDA_HOME:-/usr/local/cuda-12.9}
TCNN_PIN=3daa6e5b8b8e0fa17971a1a20acfbb3754494b38
T=$N/third_party/tiny-cuda-nn

# 1. tiny-cuda-nn at the pinned commit, with submodules, patched
if [ ! -e "$T/.patched" ]; then
  rm -rf "$T"; mkdir -p "$N/third_party"
  git clone -q https://github.com/NVlabs/tiny-cuda-nn "$T"
  git -C "$T" checkout -q "$TCNN_PIN"
  git -C "$T" submodule update -q --init --recursive
  git -C "$T" apply "$N/patches/tiny-cuda-nn.diff"
  git -C "$T/dependencies/cutlass" apply "$N/patches/cutlass.diff"
  touch "$T/.patched"
fi
[ "${FETCH_ONLY:-0}" = 1 ] && { echo "fetched $T"; exit 0; }

# 2. build and link the shared engine
export PATH=$CU/bin:$PATH
cmake -S "$N" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CU/bin/nvcc \
  -DTCNN_CUDA_ARCHITECTURES="${ARCHS:-86;90}" ${FUSED_ARCHS:+-DNRC_FUSED_ARCHITECTURES=$FUSED_ARCHS} ${F16ACC:+-DNRC_MMA_F16ACC=ON} > /dev/null
cmake --build "$OUT/build" --target cycles_nrc -j "${JOBS:-8}" > "$OUT/build.log" 2>&1 \
  || { grep -m20 -i "error" "$OUT/build.log"; exit 1; }
B=$OUT/build; TB=$B/third_party/tiny-cuda-nn
printf '{ global: nrc_create; nrc_infer; nrc_train; nrc_destroy; nrc_save; nrc_set_learning_rate; nrc_train_steps; local: *; };\n' > "$OUT/exports.map"
g++ -shared -o "$OUT/libcycles_nrc.so" -Wl,--version-script="$OUT/exports.map" \
  -Wl,--whole-archive "$B/libcycles_nrc.a" -Wl,--no-whole-archive \
  "$TB/libtiny-cuda-nn.a" "$TB/libtiny-cuda-nn-resources.a" "$TB/dependencies/fmt/libfmt.a" \
  -L$CU/lib64 -lcudart_static -L$CU/lib64/stubs -lcuda -ldl -lrt -lpthread
nm -D --defined-only "$OUT/libcycles_nrc.so" | grep -c " T " | xargs echo "exported symbols:"
echo "engine: $OUT/libcycles_nrc.so"
