#!/bin/bash
# Training records of the nrc-scatter caches, on one GPU: dump_scatter.sh <scene> <out dir>  -> <out>/s<seed>.rec, .json
# Env: as render.sh, with BAKE_CUBINS = the grouped dump cubins (kernel_tree.sh ... bake-scatter, then
# FLAGS=exact_nvcc_flags.txt EXTRA="-DCYCLES_NRC -DNRC2_FLAGS_C=1 -DNRC2_PARAM_C=0 -DNRC_GROUP_N=8" build_cubins.sh;
# configs/scatter_dump_cubins.sha256), plus ENGINE (libcycles_nrc.so) and PYTHON (with torch, for group_avg.py).
# Each render tags groups of 8 consecutive samples of a pixel for training; a group's 8 paths share one camera sample,
# so they reach the same first vertex, and take independent BSDF samples there. A tagged path stashes one record, at
# the first vertex where a normal path would end (CYCLES_NRC_TRAIN_MATCH=1, CYCLES_NRC_C=0): the features and the
# radiance it scatters towards the camera, from the full path suffix, without the vertex's own emission and light
# sampling. group_avg.py then replaces each group's records by one record with their mean target, and the raw dump
# is deleted. Seeds >= 63 also shift the camera by up to +-0.002 of the frame.
#   monster    seeds 61-79,  2048 spp, train fraction 0.2: 1.01e9 records, 45 GB
#   junkshop   seeds 61-90,  1024 spp, train fraction 0.2: 1.53e9 records, 67 GB
#   classroom  seeds 61-117, 1024 spp, train fraction 0.1: 1.49e9 records, 65 GB
# A raw dump takes 9 to 19 GB; one render and its averaging take 55 to 95 s on an H100. A seed whose .rec exists is
# skipped. Then bake (4 passes on junkshop and classroom, 1 on monster) and calibrate:
#   python3 nrc/bake/stream/bake_stream.py --dump <out> --files 's*.rec' --epochs 4 --nrcinf --out <bake dir>/bake.nrc
#   nrc/bake/calibrate.sh <scene> <bake dir>      (BAKE_CUBINS as here: zero cache and cache in the scattered form)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SC=${1:?scene}; S=${2:?out dir}; mkdir -p "$S"; S=$(cd "$S" && pwd)
: "${ENGINE:?}"; export OUT=$S/renders; mkdir -p "$OUT"; PY=${PYTHON:-python3}
case $SC in
  monster)   seeds=$(seq 61 79);  spp=2048; frac=0.2 ;;
  junkshop)  seeds=$(seq 61 90);  spp=1024; frac=0.2 ;;
  classroom) seeds=$(seq 61 117); spp=1024; frac=0.1 ;;
  *) echo "unknown scene $SC"; exit 1 ;;
esac
BASE=(CYCLES_NRC=1 CYCLES_NRC_STATS=1 CYCLES_NRC_ENGINE=$ENGINE CYCLES_NRC_TERM=0 CYCLES_NRC_TRAIN_FRAC=$frac
      CYCLES_NRC_SLOTS=33554432 CYCLES_NRC_RECORDS=16777216 CYCLES_NRC_TRAIN_SLOTS=1048576 CYCLES_NRC_TRAIN_WEIGHT=0
      CYCLES_NRC_TRAIN_MATCH=1 CYCLES_NRC_C=0 CYCLES_NRC_TRAIN_STEPS=1)
cfg=$S/opt_3e-3_0.999_0.99.json   # as dump.sh; the engine's weights are discarded
python3 -c "
import json
m = json.load(open('$HERE/../configs/t24.json'))
m['optimizer'] = {'otype': 'EMA', 'decay': 0.99, 'nested': {'otype': 'Adam', 'learning_rate': 3e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-15, 'l2_reg': 1e-6}}
m['loss'] = {'otype': 'L2'}
m['nrc'] = {'warmup_infer_batch': 262144, 'warmup_train_batch': 262144}
json.dump(m, open('$cfg', 'w'))" || exit 1
t0=$(date +%s)
for sd in $seeds; do
  rec=$S/s$sd.rec; [ -e "$rec" ] && continue
  j=0; [ "$sd" -ge 63 ] && j=$sd
  rm -f "$S/raw.rec"   # the engine appends
  JITTER=$j bash "$HERE/render.sh" "$SC" "dump-s$sd" "$sd" "$spp" "${BASE[@]}" CYCLES_NRC_CONFIG=$cfg \
    CYCLES_NRC_DUMP=$S/raw.rec || exit 1
  rm -f "$OUT/$SC-dump-s${sd}0001.exr"
  grep -q 'cycles_nrc: dumped' "$OUT/$SC-dump-s$sd.log" || { echo "no dump for seed $sd"; exit 1; }
  "$PY" "$HERE/stream/group_avg.py" "$S/raw.rec" "$rec" --n 8 --seed "$sd" --stats "$S/s$sd.json" > /dev/null || exit 1
  rm -f "$S/raw.rec"
done
echo "dump $(( $(date +%s) - t0 )) s: $S"
