#!/bin/bash
#SBATCH -J rtx-h100-repro
#SBATCH --gres=gpu:8
#SBATCH --time=00:40:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=96
# One configuration on one 8-GPU H100 node: GPUs 0-5 run one official three-scene measurement each (n = 6),
# GPUs 6-7 render the check images. Score = sum over scenes of the per-scene median.
# The project ran on 96-core nodes; with fewer cores the eight Blender processes share them and the timing changes.
#   sbatch slurm/repro.sbatch <repo> <work dir> [nrc|nrc-permuto|exact]
# <work dir> has the layout of RECIPE.md section 3: host/install, bench/, and for nrc nrc_cubins/ engine/ bakes/
# (nrc-permuto: bakes_permuto/),
# for exact exact_cubins/ and blender-5.3.0-linux-x64/ (the official binary renders the stock seed-0 and seed-1 images).
# Results: <work dir>/runs/<job id>/. Then, on CPU:
#   nrc, nrc-permuto: python3 scripts/gate.py <ref dir> <run>/check
#   exact: python3 scripts/identity.py <run>/stock_s0 <run>/stock_s1 <run>/check
set -uo pipefail
REPO=${1:?repo}; W=${2:?work dir}; CFG=${3:-nrc}
W=$(cd "$W" && pwd); STOCK=$W/blender-5.3.0-linux-x64
JID=${RUN_ID:-${SLURM_JOB_ID:-$$}}   # RUN_ID names the run when several share one Slurm job
export TMPDIR=$W/tmp/$JID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # node /tmp is often small
export BLENDER=$W/host/install BENCH=$W/bench/blender-benchmark-launcher
case $CFG in
  nrc)   export CUBINS=$W/nrc_cubins ENGINE=$W/engine/libcycles_nrc.so BAKES=$W/bakes; NIMG=3 ;;
  nrc-permuto) export CUBINS=$W/nrc_cubins ENGINE=$W/engine/libcycles_nrc.so BAKES=$W/bakes_permuto \
         NRC_CONFIG=$REPO/nrc/configs/run_infer.json; NIMG=3; CFG=nrc ;;
  exact) export CUBINS=$W/exact_cubins; NIMG=9 ;;
  *) echo "unknown configuration $CFG"; exit 1 ;;
esac
O=$W/runs/$JID; mkdir -p "$O"
hostname; date -u; nvidia-smi --query-gpu=index,name,clocks.max.sm --format=csv,noheader
(cd "$O" && sha256sum "$CUBINS"/*.cubin "$BLENDER/blender" ${ENGINE:+"$ENGINE"} > inputs.sha256)
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
run() { local g=$1 t=$2; shift 2; CUDA_VISIBLE_DEVICES=${G[$g]} TMPDIR=$TMPDIR/$t bash "$REPO/scripts/run.sh" "$@" > "$O/$t.log" 2>&1; }
for k in 1 2 3 4 5 6; do run $((k - 1)) bench_$k $CFG bench "$O/bench_$k" & done
if [ $CFG = nrc ]; then
  run 6 check_1 nrc check "$O/check" monster junkshop &
  run 7 check_2 nrc check "$O/check" classroom &
else
  run 6 check exact check "$O/check" &
  ( BLENDER=$STOCK; export CUDA_CACHE_PATH=$W/tmp/nv; run 7 stock_s0 stock check "$O/stock_s0"; SEED=1 run 7 stock_s1 stock check "$O/stock_s1" ) &
fi
wait
python3 - "$O" <<'PY'
import glob, json, statistics, sys
o = sys.argv[1]; per = {}
for f in sorted(glob.glob(f"{o}/bench_*/bench.json")):
    for e in json.load(open(f)):
        per.setdefault(e["scene"]["label"], []).append(e["stats"]["samples_per_minute"])
med = {s: statistics.median(v) for s, v in per.items()}
assert len(per) == 3 and all(len(v) == 6 for v in per.values()), {s: len(v) for s, v in per.items()}
print("n =", {s: len(v) for s, v in per.items()})
print(" ".join(f"{s}={m:.1f}" for s, m in med.items()), f"score={sum(med.values()):.1f}")
PY
rc=$?; date -u
# fail unless every scene has six measurements and every check image exists
[ $rc = 0 ] && [ "$(find "$O" -name "*_s*.exr" -size +0 | wc -l)" = $NIMG ]
