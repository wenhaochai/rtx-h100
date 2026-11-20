#!/bin/bash
# Dump one scene's training records for the offline bake, on one GPU: dump.sh <scene> <out dir>  -> <out>/s61.rec .. s69.rec
# Env: as render.sh, plus ENGINE (libcycles_nrc.so from scripts/build_engine.sh); MODEL (default nrc/configs/t24.json).
# Renders the scene 9 times at 2048 spp, seeds 61-69, with the records of bake.sh (CYCLES_NRC_TERM=0, one record per
# vertex); seeds >= 63 also shift the camera by up to +-0.002 of the frame. Each render appends every engine call's
# records to <out>/s<seed>.rec: [u32 n][n x 16 fp16 inputs][n x 3 fp32 targets] per engine call. About 240, 455 and
# 725 GB in all for monster, junkshop and classroom. The engine also trains on the records (stage-1 optimizer of
# bake.sh, 1 step per call); its weights are discarded. A seed whose .rec exists is skipped.
# Then bake and calibrate:
#   python3 nrc/bake/stream/bake_stream.py --dump <out> --out <bake dir>/bake.nrc
#   nrc/bake/calibrate.sh <scene> <bake dir>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SC=${1:?scene}; S=${2:?out dir}; mkdir -p "$S"; S=$(cd "$S" && pwd)
: "${ENGINE:?}"; export OUT=$S/renders; mkdir -p "$OUT"
MODEL=${MODEL:-$HERE/../configs/t24.json}
BASE=(CYCLES_NRC=1 CYCLES_NRC_STATS=1 CYCLES_NRC_ENGINE=$ENGINE CYCLES_NRC_TERM=0 CYCLES_NRC_TRAIN_FRAC=0.25
      CYCLES_NRC_SLOTS=8388608 CYCLES_NRC_RECORDS=16777216 CYCLES_NRC_TRAIN_SLOTS=1048576 CYCLES_NRC_TRAIN_WEIGHT=0)
cfg=$S/opt_3e-3_0.999_0.99.json   # as bake.sh's opt 3e-3 0.999 0.99
python3 -c "
import json
m = json.load(open('$MODEL'))
m['optimizer'] = {'otype': 'EMA', 'decay': 0.99, 'nested': {'otype': 'Adam', 'learning_rate': 3e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-15, 'l2_reg': 1e-6}}
m['loss'] = {'otype': 'L2'}
m['nrc'] = {'warmup_infer_batch': 262144, 'warmup_train_batch': 262144}
json.dump(m, open('$cfg', 'w'))" || exit 1
[ -s "$cfg" ] || { echo "no optimizer config"; exit 1; }
t0=$(date +%s)
for sd in 61 62 63 64 65 66 67 68 69; do
  rec=$S/s$sd.rec; [ -e "$rec" ] && continue
  j=0; [ "$sd" -ge 63 ] && j=$sd
  rm -f "$rec.part"   # the engine appends
  JITTER=$j bash "$HERE/render.sh" "$SC" "dump-s$sd" "$sd" 2048 "${BASE[@]}" CYCLES_NRC_TRAIN_STEPS=1 \
    CYCLES_NRC_CONFIG=$cfg CYCLES_NRC_DUMP=$rec.part || exit 1
  rm -f "$OUT/$SC-dump-s${sd}0001.exr"
  grep -q 'cycles_nrc: dumped' "$OUT/$SC-dump-s$sd.log" || { echo "no dump for seed $sd"; exit 1; }
  mv "$rec.part" "$rec"
done
echo "dump $(( $(date +%s) - t0 )) s: $S"
