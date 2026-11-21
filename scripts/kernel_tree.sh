#!/bin/bash
# kernel_tree.sh <blender v5.3.0 checkout> <out dir> [last patch number | bake | bake-scatter]
# Copies intern/cycles at the pinned commit to <out dir> and applies patches/kernel/NN-*.diff in order
# (all ten by default; give a number to stop at that milestone, e.g. 03 = the exact track's bhotB).
# "bake" applies patches/bake/kernel.diff instead: the tree of the bake cubins. "bake-scatter" applies it and then
# patches/bake/scatter.diff: the tree of the nrc-scatter dump cubins.
set -euo pipefail
SRC=${1:?blender checkout}; OUT=${2:?out dir}; LAST=${3:-99}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PIN=02f0899d78e374b63e7181d9061e1f7ab03419cc
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); [ -z "$(ls -A "$OUT")" ] || { echo "$OUT is not empty"; exit 1; }
git -C "$SRC" archive "$PIN" intern/cycles | tar -x -C "$OUT"
cd "$OUT"
# Inside another git work tree, `git apply` skips every file and still exits 0: stop repository discovery here,
# and prove each patch landed by checking that its reverse applies.
export GIT_CEILING_DIRECTORIES=$(dirname "$PWD")
ap() { git apply --whitespace=nowarn "$1" && git apply --check -R "$1" || { echo "patch did not apply: $1"; exit 1; }; }
if [ "$LAST" = bake ] || [ "$LAST" = bake-scatter ]; then
  ap "$HERE/patches/bake/kernel.diff"; echo "applied bake/kernel.diff"
  [ "$LAST" = bake ] || { ap "$HERE/patches/bake/scatter.diff"; echo "applied bake/scatter.diff"; }
  exit 0
fi
for p in "$HERE"/patches/kernel/[0-9][0-9]-*.diff; do
  n=$(basename "$p" | cut -c1-2); [ "$((10#$n))" -le "$((10#$LAST))" ] || break
  ap "$p"; echo "applied $(basename "$p")"
done
echo "kernel tree: $OUT/intern/cycles"
