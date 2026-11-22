#!/bin/bash
# make_reference.sh <stock blender dir> <out dir> [gpu ...]
# The exact images the gate compares against, rendered with stock Blender 5.3.0 and stock scene settings:
#   r64k_<k>/<scene>_s16384.exr, k = 0..3, seeds 100-103 (their mean is the 65,536-spp reference)
#   e32_<k>/<scene>_s32.exr,     k = 0..7, seeds 200-207 (the 32-spp bar)
# The official binary ships no sm_90 cubin; the driver compiles its PTX once (CUDA_CACHE_PATH keeps it).
# About 2 GPU-hours on one H100 at stock speed; the GPUs given share the renders.
# The project's reference was rendered with the exact bhotB build, which equals stock up to FP rounding.
# MOVE="tx,ty,tz,rx,ry,rz" renders it from a moved camera (scripts/move_camera.py).
set -euo pipefail
B=${1:?stock blender dir}; OUT=${2:?out dir}; shift 2; GPUS=("${@:-0}")
HERE=$(cd "$(dirname "$0")/.." && pwd); : "${BENCH:?BENCH = the launcher data dir}"
mkdir -p "$OUT"; export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/.nv}; export CUDA_CACHE_MAXSIZE=4294967296
jobs=()
for s in monster junkshop classroom; do
  for k in 0 1 2 3; do jobs+=("$s r64k_$k 16384 $((100 + k))"); done
  for k in 0 1 2 3 4 5 6 7; do jobs+=("$s e32_$k 32 $((200 + k))"); done
done
run() {  # run <gpu> <scene> <dir> <spp> <seed>
  local blend; blend=$(find "$BENCH/scenes" -path "*/$2/main.blend" | head -1); mkdir -p "$OUT/$3"
  [ -s "$OUT/$3/$2_s$4.exr" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 "$B/blender" -b --factory-startup -noaudio "$blend" --python "$HERE/scripts/move_camera.py" --python "$HERE/scripts/render_check.py" \
    -- "$OUT/$3/$2_s$4.exr" "$4" "$5" 2>&1 | grep render_check
}
i=0; pids=(); fail=0
reap() { for p in "${pids[@]}"; do wait "$p" || fail=1; done; pids=(); }
for j in "${jobs[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  run "$g" $j & pids+=($!)
  [ $((i % ${#GPUS[@]})) = 0 ] && reap
done
reap
n=$(ls "$OUT"/r64k_*/*_s16384.exr "$OUT"/e32_*/*_s32.exr 2>/dev/null | wc -l)
[ "$fail" = 0 ] && [ "$n" = 36 ] || { echo "reference incomplete: $n of 36 images"; exit 1; }
