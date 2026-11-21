#!/bin/bash
# run.sh <nrc|exact|stock> <bench|check> <out dir> [scene ...]
# One measurement of a configuration on one GPU (CUDA_VISIBLE_DEVICES picks it).
#   nrc:   the baked neural radiance cache configuration (48,641 samples/min; approximate, passes scripts/gate.py)
#   exact: the best configuration whose image equals stock up to floating-point rounding (14,106 samples/min; no cache)
#   stock: the official Blender binary, for the exactness reference
#   bench: the Open Data protocol per scene (a warm-up process, then a timed process, 30 s), as the launcher runs it;
#          writes <out>/bench.json and prints samples/min per scene and their sum.
#   check: a fixed-seed image per scene: nrc 4096 spp for scripts/gate.py; exact and stock 32 spp for
#          scripts/identity.py. SEED (default 0) sets the seed, CHECK_SPP the sample count. Writes <out>/<scene>_s<spp>.exr
#          and <scene>_s<spp>.log, whose "Rendered N samples in T seconds" line is the render time without scene load.
#   BENCH_LOG=<file> keeps the full Blender log of a bench run.
#   RTX PRO 6000: CONFIGS=configs/pro6000 (CUDA route) or configs/pro6000-optix with PTX=<build_ptx.sh dir> (OptiX
#   route); NRC_CONFIG=nrc/configs/run_mma_pro6000.json; BENCH_DEVICE=OPTIX with stock for stock OptiX.
# Environment (paths):
#   BENCH    the launcher's data dir, with scenes/ and benchmark-scripts/ (fetch.sh)
#   BLENDER  install dir of build_host.sh (nrc, exact) or the official binary's dir (stock)
#   CUBINS   dir with <scene>.cubin from build_cubins.sh, built for the same configuration (nrc, exact)
#   ENGINE   libcycles_nrc.so (build_engine.sh)     BAKES   dir with <scene>/bake_c0.nrc (fetch.sh)     (nrc only)
#   NRC_CONFIG  engine config (default nrc/configs/run_wgmma.json); nrc-permuto: nrc/configs/run_infer.json, BAKES=$W/bakes_permuto;
#               nrc-scatter: nrc/configs/run_infer.json, BAKES=$W/bakes_scatter, CUBINS=$W/scatter_cubins (SCATTER=1 build_cubins.sh)
set -euo pipefail
CFG=${1:?nrc|exact|stock}; MODE=${2:?bench|check}; OUT=${3:?out dir}; shift 3
SCENES=("$@"); [ ${#SCENES[@]} = 0 ] && SCENES=(monster junkshop classroom)
HERE=$(cd "$(dirname "$0")/.." && pwd); C=${CONFIGS:-$HERE/configs}   # CONFIGS: another dir of the same layout
: "${BLENDER:?}" "${BENCH:?}" "${CUDA_VISIBLE_DEVICES:?set it to one GPU index}"
case $CFG in
  nrc)   : "${CUBINS:?}" "${ENGINE:?}" "${BAKES:?}"; G=nrc_env.txt; S=nrc_env.txt; SPP=4096 ;;
  exact) : "${CUBINS:?}"; G=exact_env.txt; S=exact_env.txt; SPP=32 ;;
  stock) G=; S=; SPP=32; export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache} CUDA_CACHE_MAXSIZE=4294967296 ;;  # no sm_90 cubin ships: the driver compiles the PTX once
  *) echo "unknown configuration $CFG"; exit 1 ;;
esac
SPP=${CHECK_SPP:-$SPP}
# PTX=<dir of build_ptx.sh>: the OptiX route (RT cores; needs a host with patches/optix/host.diff): Blender's OptiX
# device, each scene's OptiX module from <dir>/<scene>/kernel_optix.ptx.zst, the shading kernels from CUBINS.
[ -n "${PTX:-}" ] && export BENCH_DEVICE=OPTIX CHECK_DEVICE=OPTIX
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export TMPDIR=${TMPDIR:-$OUT/tmp}; mkdir -p "$TMPDIR"; export XDG_CACHE_HOME=$TMPDIR/xdg

# global settings
[ -z "$G" ] || while read -r kv; do [ -n "$kv" ] && export "$kv"; done < "$C/$G"
# GLOBAL_EXTRA="K=V K=V": settings on top of the configuration's (tuning runs)
for kv in ${GLOBAL_EXTRA:-}; do export "$kv"; done
if [ "$CFG" = nrc ]; then
  export CYCLES_NRC_ENGINE=$ENGINE CYCLES_NRC_CONFIG=${NRC_CONFIG:-$HERE/nrc/configs/run_wgmma.json}
  export PRE_PY="exec(open('$C/scene_settings.py').read())"
fi
# per-scene settings: SCENE_ENV="<scene>:K=V,K=V;..." (read by direct_bench.py; set here for the check render)
senv() {
  local s=$1
  if [ "$CFG" = stock ]; then
    local part; local IFS=';'
    for part in ${SCENE_EXTRA:-}; do [ "${part%%:*}" = "$s" ] && printf '%s' "${part#*:}"; done
    return 0
  fi
  printf '%s' "CYCLES_CUDA_CUBIN=$CUBINS/$s.cubin,CYCLES_SVM_NODES_CHECK=$C/$s/svm_nodes_used.h"
  [ "$CFG" = nrc ] && printf ',%s' "CYCLES_NRC_LOAD=$BAKES/$s/bake_c0.nrc"
  [ -n "${PTX:-}" ] && printf ',%s' "CYCLES_OPTIX_PTX=$PTX/$s/kernel_optix.ptx.zst"
  while read -r kv; do [ -n "$kv" ] && printf ',%s' "$kv"; done < "$C/$s/$S"
  # SCENE_EXTRA="<scene>:K=V,K=V;<scene>:...": per-scene settings on top (later values win)
  local part; local IFS=';'
  for part in ${SCENE_EXTRA:-}; do [ "${part%%:*}" = "$s" ] && printf ',%s' "${part#*:}"; done
  return 0
}
if [ "$CFG" != stock ] || [ -n "${SCENE_EXTRA:-}" ]; then
  SE=""; for s in "${SCENES[@]}"; do SE="$SE${SE:+;}$s:$(senv "$s")"; done
  export SCENE_ENV=$SE
fi
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" || true

if [ "$MODE" = bench ]; then
  python3 "$HERE/scripts/direct_bench.py" "$BLENDER/blender" "$BENCH/scenes" \
    "$(find "$BENCH/benchmark-scripts" -name main.py | head -1)" "${SCENES[@]}" > "$OUT/bench.json.tmp"
  mv "$OUT/bench.json.tmp" "$OUT/bench.json"   # only a finished measurement leaves bench.json
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1])); p = {e["scene"]["label"]: e["stats"]["samples_per_minute"] for e in d}
print(" ".join(f"{k}={v:.1f}" for k, v in p.items()), f"score={sum(p.values()):.1f}")' "$OUT/bench.json"
else
  for s in "${SCENES[@]}"; do
    blend=$(find "$BENCH/scenes" -path "*/$s/main.blend" | head -1)
    env $(senv "$s" | tr ',' ' ') "$BLENDER/blender" -b --factory-startup -noaudio --debug-cycles "$blend" ${PRE_PY:+--python-expr "$PRE_PY"} \
      --python "$HERE/scripts/render_check.py" -- "$OUT/${s}_s$SPP.exr" "$SPP" "${SEED:-0}" > "$OUT/${s}_s$SPP.log" 2>&1 || true
    grep -E "render_check|scene_settings|Error|error" "$OUT/${s}_s$SPP.log" | tail -4 || true
    [ -s "$OUT/${s}_s$SPP.exr" ] || { echo "no image for $s; log: $OUT/${s}_s$SPP.log"; tail -20 "$OUT/${s}_s$SPP.log"; exit 1; }
  done
fi
