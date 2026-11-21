# RTX PRO 6000: the rtx-h100 recipe on Blackwell

The `exact` and `nrc` configurations of this repository, ported from one H100 (sm_90) to one NVIDIA RTX PRO 6000 Blackwell
Server Edition (sm_120, 188 SMs, 96 GB, driver 595.91.07) and tuned there. Blender 5.3.0 Cycles, Open Data benchmark
(launcher 3.3.0, monster + junkshop + classroom), one GPU per measurement, official protocol (`scripts/direct_bench.py`:
a warm-up process, then a fresh timed process of 30 s per scene). The card has RT cores, so every configuration exists
twice: the CUDA route, which is comparable with the H100 numbers, and an OptiX route that traces rays on the RT cores and
shades with the same cubins.

## Results

Final confirmation, 2026-11-20 15:07–15:58 UTC: six configurations alternating on the same GPU, three rounds on each of
GPU0, GPU1 and GPU2 (9 runs per configuration). Per-scene values are medians over the 9 runs; the total is their sum,
as in the H100 project. Raw data: `results/final/g{0,1,2}` of the work directory.

| samples per minute | monster | junkshop | classroom | total | vs stock, same device | image |
|---|---|---|---|---|---|---|
| stock Blender 5.3.0, CUDA | 4,052 | 2,671 | 2,369 | 9,092 | | reference |
| stock Blender 5.3.0, OptiX | 8,092 | 6,172 | 5,271 | 19,535 | | reference |
| `exact`, CUDA | 9,790 | 5,502 | 4,650 | **19,942** | 2.19x stock CUDA | equal to stock CUDA up to FP rounding |
| `exact`, OptiX | 10,778 | 8,080 | 7,006 | **25,864** | 1.32x stock OptiX | equal to stock OptiX up to FP rounding |
| `nrc`, CUDA | 33,890 | 19,571 | 21,385 | **74,847** | 8.23x stock CUDA, 3.83x stock OptiX | passes the gate |
| **`nrc`, OptiX** | **35,634** | **22,040** | **20,623** | **78,297** | **4.01x stock OptiX** | passes the gate |

Spread of the 9 totals (mean ± sd): stock CUDA 9,048 ± 105, stock OptiX 19,379 ± 429 (one run on GPU1 overran the
30-s window on junkshop and classroom, 18,266), exact CUDA 19,944 ± 13, exact OptiX 25,851 ± 36, nrc CUDA
74,775 ± 246 (one monster run 1.6% low on GPU2), nrc OptiX 78,289 ± 64. Per-GPU means of nrc OptiX: 78,332 / 78,304 /
78,230.

For comparison, the H100 SXM: stock 7,041, `exact` 14,106, `nrc` 48,641. The unchanged H100 configurations ported to
sm_120 (only `-arch` changed) score `exact` 14,009 and `nrc` 49,173 on this card (6 runs each, 3 GPUs), and pass the
same checks.

Per-scene choice of device is not part of these numbers: an Open Data result has one device type. With monster and
junkshop on OptiX and classroom on CUDA the sum of medians would be 79,059.

## Image checks

**`nrc` (both routes)**: `scripts/gate.py` with a reference rendered on this card (stock 5.3.0, CUDA: four
16,384-spp renders at seeds 100–103, eight 32-spp bar renders at seeds 200–207). The bar values equal the H100
project's to the second decimal (24.25 / 18.84 / 23.84 dB).

| | monster | junkshop | classroom |
|---|---|---|---|
| 32-spp bar | 24.25 dB | 18.84 dB | 23.84 dB |
| `nrc` H100 configuration on this card | 33.05 dB, -0.11% | 23.80 dB, +0.24% | 32.31 dB, -0.10% |
| `nrc` final, CUDA | 33.03 dB, -0.05% | 23.81 dB, +0.22% | 32.32 dB, -0.04% |
| `nrc` final, OptiX | 33.03 dB, -0.05% | 23.81 dB, +0.22% | 32.32 dB, -0.04% |

The final configurations change one image setting against the H100 one, scrambling distance 0 (below). Its gate
results at seeds 1 and 2 (cubins without compile-time KernelData, which fold the seed): monster -0.28 / -0.31%,
junkshop -0.01 / +0.23%, classroom +0.04 / +0.28%, all pass; the H100 settings give -0.19 / -0.45%, +0.02 / +0.02%,
+0.06 / +0.21% there.

**`exact`**: `scripts/identity.py`, MAE against the stock image at seed 0 over the seed-to-seed noise floor, 32 spp.

| | monster | junkshop | classroom |
|---|---|---|---|
| `exact` CUDA vs stock CUDA | 6.78e-5 | 1.38e-4 | 4.15e-5 |
| `exact` OptiX vs stock OptiX | 3.12e-5 | 5.24e-5 | 2.67e-5 |
| (stock OptiX vs stock CUDA) | 5.48e-3 | 7.53e-3 | 8.54e-3 |

OptiX intersection rounds differently from Cycles' CUDA BVH, so the OptiX route is compared with stock OptiX.

## Samples per minute and image quality

As on the H100, a cached sample does far less work than an exact one. Seed-0 renders at a range of sample counts
(three renders per point, median of Cycles' render time without scene load), PSNR against the 65,536-spp reference
as in the gate (`results/quality/*` of the work directory):

| scene | `nrc` samples | PSNR | luminance | `nrc` CUDA | `nrc` OptiX | `exact` to the same PSNR, CUDA / OptiX |
|---|---|---|---|---|---|---|
| monster | 16 | 24.72 dB | +1.76% | 0.055 s | 0.052 s | 0.33 s / 0.29 s |
| monster | 256 | 31.43 dB | -1.43% | 0.50 s | 0.46 s | 1.16 s / 1.06 s |
| monster | 1,024 | 32.87 dB | +0.59% | 1.87 s | 1.76 s | 1.55 s / 1.41 s |
| junkshop | 20 | 18.89 dB | -1.26% | 0.112 s | 0.102 s | 0.55 s / 0.42 s |
| junkshop | 1,024 | 23.76 dB | -0.10% | 3.25 s | 2.84 s | 1.11 s / 0.77 s |
| classroom | 20 | 23.93 dB | +5.02% | 0.100 s | 0.092 s | 0.87 s / 0.78 s |
| classroom | 1,024 | 32.24 dB | +0.34% | 2.98 s | 3.06 s | 3.19 s / 2.39 s |

`exact` times interpolate its measured points (32, 64, 128, 256, 1,024 spp) in log time; it gains about 3.0 to 4.5 dB
per doubling there. The CUDA and OptiX routes render the same `nrc` images (PSNR equal to 0.01 dB at every sample count). The fewest
samples that clear the 32-spp bar are 16 / 20 / 20; images per second at the bar, summed over the three scenes:

| images per second at the 32-spp bar | monster | junkshop | classroom | total | vs stock, same device |
|---|---|---|---|---|---|
| stock CUDA, 32 spp | 1.65 | 1.04 | 0.77 | 3.46 | |
| stock OptiX, 32 spp | 3.19 | 2.12 | 1.20 | 6.52 | |
| `exact` CUDA, 32 spp | 3.26 | 1.82 | 1.17 | 6.25 | 1.8x |
| `exact` OptiX, 32 spp | 3.80 | 2.38 | 1.30 | 7.48 | 1.15x |
| `nrc` CUDA, 16 / 20 / 20 spp | 18.2 | 8.9 | 10.0 | 37.1 | 10.7x |
| **`nrc` OptiX, 16 / 20 / 20 spp** | **19.2** | **9.8** | **10.9** | **39.9** | **6.1x** |

The H100 project measured 28.6 images per second for `nrc` and 2.95 for stock on one H100. The same limits hold here:
at the bar `nrc`'s mean luminance is off by +1.8%, -1.3% and +5.0%; its PSNR levels off near 33, 24 and 32 dB (the
gate images at 4,096 spp: 33.03, 23.81, 32.32 dB), while `exact` keeps gaining 3 to 4.5 dB per doubling and reaches
39.3, 36.5 and 39.2 dB at 1,024 spp. Near that plateau the exact build gives the same image sooner: on monster from
about 32.5 dB, on junkshop from about 23.6 dB, on classroom from about 32 dB (OptiX; on CUDA the two are level at
32.2 dB). Below it, `nrc` is 2-9x sooner. As on the H100.

## What changed against the H100 configuration

Each step was measured against the previous one on the same GPU, alternating, 3 rounds (`results/b*` of the work
directory); gains are on the total unless a scene is named.

**CUDA route, `nrc`** (H100 configuration on sm_120: 49,173)

| step | total | gain | image |
|---|---|---|---|
| shade kernels 384 threads x 1 block, 168 registers (H100: 512x2 / 1024x1, 64 registers) | 69,265 | +40.8% | unchanged |
| hash grid gathered inside the fused MLP kernel (`fused_grid_levels` 0; H100: split grid pass) | 71,282 | +2.8% | unchanged |
| monster SSS walk kernel 64 x 16 (64 registers; H100: 64 x 12) | 71,894 | monster +2.0% | unchanged |
| scrambling distance 0 on all scenes, KernelData re-dumped | 74,344 | +3.6% | gate passes at seeds 0-2 |
| classroom cubin with the CUDA 13.4 front end, junkshop with `--jump-table-density 80`; fp16 accumulation in the MLP | 74,763 | +0.6% | gate unchanged |

**OptiX route, `nrc`** (same cubins for shading, OptiX module for every intersection)

| step | total | gain |
|---|---|---|
| OptiX module from the nrc tree, H100 shade bounds | 47,512 | -3.2% against CUDA |
| 168-register shade kernels, grid gathered in the MLP | 70,473 | -1.0% against CUDA |
| OptiX module register limit 56 / 72 / 64 | 71,694 | +1.7% |
| scrambling distance 0 | 76,539 | +6.8% (monster +13%) |
| walk roulette of the CUDA SSS walk in the OptiX SSS walk; fp16 MLP accumulation | 77,912 | +1.8% (monster +3.7%) |
| 24x concurrent states on monster and junkshop | 78,297 (final) | +0.5% |

**`exact`**: CUDA 14,009 → 19,589 with the 168-register shade kernels (+39.8%) → 19,874 with monster SSS 64 x 16
(+1.4%) → 19,942 with the classroom cubin from CUDA 13.4 (+0.4%). OptiX: 24,764 (16x states) → 25,512 with the
module register limit (+3.2%) → 25,864 with the exact CUDA cubins above (classroom +4.9%).

The biggest step is the H100's 64-register shade bounds, which are the worst point measured on sm_120: 64, 80, 128,
168 and 176-200 registers (255 cap) give 49,183, 52,698, 59,299, 69,265 and 57,676. Blender 5.3's own sm_120 default
is the same 384 x 1 / 168 point. ptxas 12.9 has no point between 128 and 168 for these kernels (every cap in between
compiles to 128). The cause was not profiled further (no ncu on this machine); sm_120 has the H100's 65,536 registers
per SM but at most 1,536 resident threads (H100: 2,048).

Time split of the final `nrc` configurations (nsys, seed-0 1,024-spp render, share of kernel time): CUDA route
shade_surface 51%, intersection 18-27%, MLP 13-17%, camera ray setup 5-6%; OptiX route shade_surface 54-60%, OptiX
intersection 10-16%, MLP 15-24%.

## What did not help

| idea | result |
|---|---|
| CUDA 13.4 or 13.4 front end + 12.9 ptxas for all scenes at 168 registers | -5.6% / -8.8% (junkshop -17%); at 64 registers they were +6% |
| `--jump-table-density 80` alone | -0.5% / -0.2% |
| intersect launch bounds 384x1, 128x4, 64x12, 64x16 without the per-kernel minimum blocks | -0.9% to -2.2% |
| concurrent states 8 / 12 / 24 / 32 | -1.4% / -1.3% / +0.1% / out of memory |
| sort partition size 65,536 / 262,144 / 1M, shadow compaction 2 / 8 | within ±0.3%; 1M -3.7% |
| MLP warpgroups 2 / 1, grid levels per pass 2 / 4 / 16 | -2.7% / 0; +0.7% / 0 / -2.2% (all below gathering in the kernel) |
| light threshold 0.5 / 1.0 | +1.4% / +2.8%, fails the luminance limit on junkshop at seed 0 and on monster and classroom at seeds 1 and 2 |
| monster SSS walk kernels 64x20, 64x24, 128x4, 384x1 | -0.7% / -4.1% / -1.1% / -4.2% |
| splitting the OptiX SSS walk over launches (24 / 48 / 96 steps; +10.9% on stock in earlier Pro 6000 work) | -3.4% / -2.1% / -1.3% once the walk roulette shortens the walks |
| OptiX register limit 48-96 around 56/72/64, sort partitions on OptiX | within ±0.1% |
| resetting the persisting-L2 set-aside for the MLP on the OptiX route | no effect; the MLP's per-query cost there stays 1.6x the CUDA route's after the first batch, cause not found (no ncu on this machine) |

## Reproduce

Same tools as the H100 build plus the OptiX 8.0 SDK; CUDA 12.9 and 13.4 (the classroom cubins use 13.4's nvcc).

```bash
W=<work dir>                                   # scripts/fetch.sh $W as for the H100
scripts/build_host.sh $W/blender $W/host       # CUDA route host (GCC 14: GCC_TOOLSET_DIR, or CC/CXX/GCC_RUNTIME)
PATCH2=patches/optix/host.diff OPTIX_ROOT=<optix-8.0.0> scripts/build_host.sh $W/blender $W/host_optix
ARCHS=120 FUSED_ARCHS=120-real F16ACC=1 scripts/build_engine.sh $W/engine

R=$PWD; export GIT_CEILING_DIRECTORIES=$W     # git apply inside another work tree would skip every file
scripts/kernel_tree.sh $W/blender $W/tree_nrc && (cd $W/tree_nrc && git apply $R/patches/optix/kernel.diff)
scripts/kernel_tree.sh $W/blender $W/tree_exact 03 && (cd $W/tree_exact && git apply $R/patches/optix/kernel.diff $R/patches/optix/kernel-exact.diff)
ARCH=sm_120 CONFIGS=configs/pro6000 scripts/build_cubins.sh $W/tree_nrc/intern/cycles $W/nrc_cubins
ARCH=sm_120 CONFIGS=configs/pro6000 FLAGS=exact_nvcc_flags.txt scripts/build_cubins.sh $W/tree_exact/intern/cycles $W/exact_cubins
ARCH=sm_120 CONFIGS=configs/pro6000-optix scripts/build_cubins.sh $W/tree_nrc/intern/cycles $W/nrc_cubins_optix
ARCH=sm_120 CONFIGS=configs/pro6000-optix FLAGS=exact_nvcc_flags.txt scripts/build_cubins.sh $W/tree_exact/intern/cycles $W/exact_cubins_optix
OPTIX_ROOT=<optix-8.0.0> CONFIGS=configs/pro6000-optix scripts/build_ptx.sh $W/tree_nrc/intern/cycles $W/nrc_ptx
OPTIX_ROOT=<optix-8.0.0> CONFIGS=configs/pro6000-optix FLAGS=exact_nvcc_flags.txt scripts/build_ptx.sh $W/tree_exact/intern/cycles $W/exact_ptx

export BENCH=$W/bench/blender-benchmark-launcher ENGINE=$W/engine/libcycles_nrc.so BAKES=$W/bakes
export NRC_CONFIG=nrc/configs/run_mma_pro6000.json CUDA_VISIBLE_DEVICES=0
BLENDER=$W/host/install CUBINS=$W/nrc_cubins CONFIGS=configs/pro6000 scripts/run.sh nrc bench out/nrc_cuda
BLENDER=$W/host_optix/install CUBINS=$W/nrc_cubins_optix CONFIGS=configs/pro6000-optix PTX=$W/nrc_ptx scripts/run.sh nrc bench out/nrc_optix
BLENDER=$W/host/install CUBINS=$W/exact_cubins CONFIGS=configs/pro6000 scripts/run.sh exact bench out/exact_cuda
BLENDER=$W/host_optix/install CUBINS=$W/exact_cubins_optix CONFIGS=configs/pro6000-optix PTX=$W/exact_ptx scripts/run.sh exact bench out/exact_optix
BLENDER=$W/blender-5.3.0-linux-x64 BENCH_DEVICE=OPTIX scripts/run.sh stock bench out/stock_optix
```

`check` in place of `bench` renders the gate (nrc, 4,096 spp) and identity (exact, 32 spp) images; `CHECK_DEVICE=OPTIX`
renders stock OptiX. The cubins and PTX of the four configurations rebuild byte-identical to the measured ones:
`configs/pro6000*/{nrc,exact}_cubins.sha256`, `configs/pro6000-optix/{nrc,exact}_ptx.sha256` (uncompressed PTX).
The OptiX host was measured with the OptiX device switches of `patches/optix/host.diff` on top of `patches/host.diff`.
A host rebuilt from scratch with `PATCH2=patches/optix/host.diff scripts/build_host.sh` has the same Cycles source, the
same gate and identity results, and scores within noise of it (one round on each of 3 GPUs: nrc -0.02 / +0.10 /
+0.26%, exact +0.10 / -0.02 / +0.09%).

Notes:

- The bakes are the H100 project's (`configs/bakes.sha256`); nothing was re-baked. The compile-time KernelData was
  re-dumped on this card with the new scene settings (`scripts/gen_kconst.py`); with the H100 settings the re-dump equals
  the shipped `kdata_const.h` field for field.
- `configs/pro6000*/scene_settings.py` sets scrambling distance 0 instead of 0.04 / 0.025 / 0.035; everything else in
  it is the H100 file.
- fp16 accumulation (`F16ACC=1`) makes the fused MLP compute exactly what tiny-cuda-nn's own inference computes (self
  check max relative difference 0; the fp32-accumulating kernel differs from it by up to 0.8-4.6%), +0.27%.
- Hardware: four RTX PRO 6000 Blackwell Server Edition in one node; GPU0-2 were used, one GPU per measurement.

## Second stage: the OptiX-side work of the earlier Pro 6000 project, merged into `exact` and strict

The earlier RTX PRO 6000 project (rtxpro6000-blender, Blender 5.3.0, OptiX route) had a strict configuration B3s
(27,316) and a best configuration B7 (30,386, with full SVM specialization). This stage starts from its tree
(v5.3.0 + `blender-5.3.0/patches/v53-all.patch`, which already has the SSS walk split, warp-aggregated queue atomics,
per-kernel launch bounds, sort partitions and SVM specialization) and its host, and adds the shading changes of this
repository that do not change the image: SVM node pruning per scene and forced inlining of hot functions
(`patches/v53/x1-shading.diff`, from patches/kernel/03-best), the per-scene closure count, and the compile-time
KernelData of the nrc track (`patches/v53/x2-kdata.diff`, headers re-dumped for this tree).

Image tiers, both against stock 5.3.0 OptiX at seed 0 over the seed-1 noise floor, at 32 and at 256 spp:
**exact** at most 2e-4 at both (the level of a toolchain change; this repository's OptiX `exact` is 3e-5 to 1.5e-4);
**strict** below 2e-3 at both (that project's rule).

Final confirmation, 2026-11-20 18:04-18:45 UTC: six configurations alternating on the same GPU, 3 rounds on each of
GPU0-2; per-scene medians of 9 runs.

| samples per minute | monster | junkshop | classroom | total | vs | image (32 / 256 spp, max over scenes) |
|---|---|---|---|---|---|---|
| stock 5.3.0 OptiX | 8,086 | 6,197 | 5,272 | 19,554 | | |
| B3s (that project's strict) | 12,047 | 7,952 | 7,281 | 27,279 | | 5.2e-5 / 1.5e-4: exact |
| B7 (that project's best) | 13,504 | 8,796 | 8,069 | 30,369 | | 2.0e-4 / 5.8e-4: strict |
| `exact`, no folded scene data | 12,157 | 8,127 | 7,569 | **27,853** | B3s +2.1% | 5.2e-5 / 1.5e-4: exact |
| **`exact`** | 12,295 | 8,361 | 7,531 | **28,187** | B3s +3.3%, stock 1.44x | 5.2e-5 / 1.65e-4: exact |
| **strict** | 13,686 | 9,332 | 8,366 | **31,384** | B7 +3.3%, stock 1.61x | 2.1e-4 / 6.1e-4: strict |

Per scene, 32 / 256 spp: `exact` 5.22e-5 / 1.65e-4 (monster), 5.22e-5 / 1.52e-4 (junkshop), 3.63e-5 / 1.04e-4
(classroom); strict 2.13e-4 / 6.11e-4, 1.23e-4 / 3.53e-4, 5.24e-5 / 1.55e-4. The two stock-OptiX floors at 256 spp,
1.037e-2 / 6.0e-3 / 1.787e-2, equal that project's.

What each step gave (same GPU, alternating, 3 rounds):

| step | exact (base B3s) | strict (base B7) |
|---|---|---|
| per-scene closure count alone (24 / 14 / 12 instead of 64) | -0.15% | -0.15% |
| + SVM node pruning and hot-function inlining | +1.51% (junkshop +2.3%, classroom +2.2%) | +0.44% (junkshop +1.6%) |
| classroom cubin from CUDA 13.4 (one-step nvcc), shade 256 x 2 kept | classroom +1.7% | (specialized classroom stays 13.4 front end + 12.9 ptxas) |
| monster without an OptiX register limit (B3s: 56) | monster +0.9% | (B7 already has none) |
| + compile-time KernelData | +1.26% (monster +1.2%, junkshop +2.6%) | +3.16% (monster +1.8%, junkshop +4.5%, classroom +3.9%) |

Not kept: monster states 12, SSS steps 32 / 64, 1,128 sort partitions (noise or -0.5%); monster or junkshop cubins
from CUDA 13.4 (-1.4% / -0.2%); classroom shade at 168 registers (-4.2%, -1.7% with 13.4); OptiX register limits off
or raised on junkshop and classroom (-0.8% to -2.6%, or noise); specialization masks 15 / 31 on classroom (-0.1% /
-0.4%) and 31 on junkshop (+0.36%, within two standard deviations); the 13.4 front end + 12.9 ptxas build for the
exact cubins (+0.49%; not used, NVIDIA does not support it). Not confirmed: specialization mask 7 on monster in the
strict configuration, +1.58% on monster in one 3-round screen (+0.5% total); the confirmation was cancelled when the
work ended.

Notes: the strict cubins use that project's B7 build, CUDA 13.4's front end with the PTX header rewritten for CUDA
12.9's ptxas, which NVIDIA does not support; `exact` uses only one-step nvcc builds (12.9; 13.4 for classroom).
Compile-time KernelData folds the scene's constants into the shading code; its images stay within the exact tier,
and the variant without it (27,853) is the one that changes no arithmetic at all.

Reproduce: Blender v5.3.0 + `v53-all.patch` (that project's host build) for the host and the base PTX; the kernel
tree for the cubins is the same plus `patches/v53/x1-shading.diff` and `x2-kdata.diff`; `scripts/build_cubins_v53.sh
<tree>/intern/cycles <out> exact|exact_nofold|strict` rebuilds all nine cubins byte-identical to the measured ones
(`configs/pro6000-v53/*_cubins.sha256`); run settings in `configs/pro6000-v53/runs.txt`.
