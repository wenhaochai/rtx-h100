# RTX H100

[X post](https://x.com/wenhaocha1/status/2123597363335876818) · [Recipe](RECIPE.md) · [Baked caches](https://huggingface.co/wchai/cycles-h100-nrc-bake) · [Skills](skills/)

![The three benchmark scenes rendered for the same 30 seconds by stock Blender and by this build on one H100](assets/race.webp)

<sub>Stock Blender and this build render the three benchmark scenes for the same 30 seconds on one H100.</sub>

**A compute card can ray-trace like a graphics card.** On one NVIDIA H100, Blender 5.3.0 Cycles renders the Open Data benchmark at 48,641 samples per minute: 6.9 times stock Blender on the same card and 2.5 times an RTX 5090 with OptiX.

Exact per-scene kernel specialization gives the first 2.0 times. A per-scene baked neural radiance cache that ends every path at its first hit gives most of the rest. The result passes a 32-spp image bar at seed 0, the only seed its monster and junkshop kernels can render; the same cache on an earlier build misses the luminance limit on monster at seed 2. Wide BVHs, kernel concurrency and online cache training gave nothing. At the quality of an exact 32-spp render, `nrc` delivers 9.7 times as many images per second as stock Blender; its quality has a ceiling, above which the exact build is faster, and at low sample counts its mean brightness is off by 2 to 8%.

## Results

| samples per minute | monster | junkshop | classroom | total | image |
|---|---|---|---|---|---|
| stock Blender 5.3.0 | 2,768 | 2,450 | 1,824 | 7,041 | reference |
| RTX 5090, OptiX, Open Data median | | | | 19,202 | reference |
| `exact`: specialized kernels, no approximation | 6,706 | 4,050 | 3,356 | 14,106 | equal to stock up to floating-point rounding |
| **`nrc`: + baked neural radiance cache** | **20,275** | **13,319** | **15,047** | **48,641** | approximate, passes the image gate |
| `nrc-hq`: `nrc` with the offline-baked caches | 20,377 | 13,434 | 15,187 | 48,997 | approximate, passes the image gate |
| `nrc-permuto`: caches on a lattice over position and direction | 19,087 | 13,318 | 13,979 | 46,385 | approximate, passes the image gate |

One H100 SXM 80 GB, official benchmark protocol. Per-scene values are medians over 6 to 12 three-scene runs; totals are as the project reported them, so they can differ from the column sum by a few units. Rebuilt and re-measured from this repository, `nrc` scores 48,721 and passes the same gate, and `exact` scores 14,116 with the same image difference. `nrc-hq` runs the same model, engine and cubins with caches baked offline from the same exact renders; it was measured on another node, so its difference from `nrc` is node spread. Its gate PSNR is 36.17, 32.00 and 35.63 dB, against 33.05, 23.80 and 32.31 dB for `nrc`. `nrc-permuto` puts the outgoing direction into the cache's grid, a permutohedral lattice, and runs it on a second engine in the same library; measured on a third node, its gate PSNR is 38.30, 34.24 and 38.09 dB.

| images per second at the 32-spp bar | monster | junkshop | classroom | total | vs stock |
|---|---|---|---|---|---|
| stock Blender 5.3.0, 32 spp | 1.27 | 1.01 | 0.67 | 2.95 | 1.0x |
| `exact`, 32 spp | 2.68 | 1.57 | 1.07 | 5.32 | 1.8x |
| **`nrc`, 20 / 20 / 24 spp** | **13.5** | **7.52** | **7.58** | **28.6** | **9.7x** |
| `nrc-hq`, 20 / 32 / 24 spp | 12.7 | 5.32 | 6.80 | 24.8 | 8.4x |
| `nrc-permuto`, 24 / 40 / 24 spp | 8.75 | 3.61 | 6.48 | 18.8 | 6.4x |

A sample of `nrc` does less work than a sample of stock Blender, so this table counts images instead: how many images per second reach the PSNR of an exact 32-spp render against the 65,536-spp reference, the gate's bar. Each value is one over the median render time of three seed-0 renders at the fewest samples that clear the bar, without scene load; `nrc`'s mean luminance at those counts is off by 2 to 8%.

## How it got there

![The score of one H100 on Blender Open Data climbing from 7,041 to 48,641, each key step named](assets/evolution.jpg)

<sub>Every measured configuration of the project in order, with each new best named.</sub>

1. **Specialize the kernel to the scene, exactly.** Per-kernel launch bounds, the scene's own feature mask, eight times more paths in flight, a BVH traversal that refills idle lanes, and a shader interpreter with every unused node compiled out. The image stays equal to stock. 7,041 to 14,106.
2. **Approximate the sampling.** Stronger Russian roulette, a light threshold, cheaper indirect bounces, then more exact work on top: compile-time scene constants, flat leaf triangles, per-material shading programs, and a per-cell radiance cache. 14,106 to 27,041.
3. **Bake the lighting into a neural cache.** A hash grid and a small MLP per scene, trained offline on exact renders, frozen, and queried at the first hit of every camera path, where the path then ends. 27,041 to 48,641.

[RECIPE.md](RECIPE.md) has every step with its gain and image cost, the bake, the reproduction, and the fifteen ideas that failed.

## Samples per minute and image quality

![PSNR against render time for three nrc caches, exact and stock Blender on the three scenes](assets/quality_time.png)

<sub>Seed-0 renders on one H100, PSNR against the 65,536-spp exact reference. The cache lines have points at 16, 64, 256 and 1,024 samples; `exact` at 32 to 512. `nrc` is the production cache, `nrc-hq` the offline-baked one, `nrc-permuto` the lattice cache. Render time is Cycles' render time without scene load, the median of three renders for every cache point and one render for `exact`. Stock Blender's time is the exact build's, scaled by the ratio of their samples per minute.</sub>

The benchmark counts samples, and a cached sample does far less work than an exact one, so samples per minute overstate how fast a usable image appears. Measured as image quality against render time, `nrc` clears the PSNR bar at 20 to 24 samples, in 0.07 to 0.13 s: 4.8 to 7.1 times sooner than the exact build and 7.4 to 11.4 times sooner than stock Blender, which reach the bar at 32 samples. Its bias caps the quality near 33, 24 and 32 dB, so beyond about 1,000 samples the exact build gets there first and keeps improving; the benchmark score and the gate images both sit in that region. At those sample counts its mean luminance is off by 2 to 8%, past the gate's 0.5% limit, and the 9 to 18 minutes of baking per scene are not counted.

`nrc-hq` lifts that ceiling. At 1,024 samples it reaches 35.3, 31.2 and 35.1 dB, against 32.8, 23.7 and 32.1 dB for `nrc`, and it still gets there 1.3, 1.1 and 1.8 times sooner than the exact build. At 256 samples it is 3.3, 2.5 and 4.9 times sooner. At the 32-sample bar it is slower than `nrc`, mainly on junkshop, which needs 32 samples instead of 20; the three scenes give 24.8 images per second against 28.6.

`nrc-permuto` lifts it further with the same 64-wide network. At 1,024 samples it reaches 37.0, 32.6 and 37.1 dB, 1.7, 1.5 and 2.1 dB above `nrc-hq`. At the bar it is the slowest of the three published caches: at 32 samples its noise on junkshop stays at the exact build's level, and junkshop needs 40 samples. The three scenes give 18.8 images per second. RECIPE.md section 5 has the bias and variance split.

## Reproduce

Needs an H100, the CUDA 12.9 toolkit, GCC 14, CMake, git-lfs, and Python 3 with numpy and OpenEXR. `$W` is one work directory.

```bash
scripts/fetch.sh $W                            # Blender source, benchmark data, baked caches
scripts/build_host.sh $W/blender $W/host       # headless Blender + patches/host.diff
scripts/build_engine.sh $W/engine              # the cache engine
scripts/kernel_tree.sh $W/blender $W/tree      # intern/cycles + patches/kernel/01..10
scripts/build_cubins.sh $W/tree/intern/cycles $W/nrc_cubins

export BENCH=$W/bench/blender-benchmark-launcher BLENDER=$W/host/install
export CUBINS=$W/nrc_cubins ENGINE=$W/engine/libcycles_nrc.so BAKES=$W/bakes
CUDA_VISIBLE_DEVICES=0 scripts/run.sh nrc bench out/nrc_1
```

For `nrc-hq`, run the same commands with `BAKES=$W/bakes_hq`. For `nrc-permuto`, add `BAKES=$W/bakes_permuto NRC_CONFIG=nrc/configs/run_infer.json`.

The `exact` configuration, the image gate, the exactness check and the Slurm job are in [RECIPE.md](RECIPE.md), sections 3 and 6.

## Skills

Four [Claude Code skills](skills/) distilled from how the work was done. Each is general, not tied to Cycles, and lists the traps this project actually hit.

- **[gpu-profile](skills/gpu-profile/SKILL.md)**: find where a CUDA workload's time goes before changing it. Split time per kernel, name the bound of each hot kernel, read registers and spills, and write an Amdahl table of what each lever can buy. Here that table showed early that no exact change could reach the RTX 5090, which moved the work to approximation.
- **[kernel-specialize](skills/kernel-specialize/SKILL.md)**: build per-input kernels whose output does not change: launch bounds, feature masks, dead-code pruning, compile-time constants, every change behind a macro. This gave the 2.0x of the exact track. Its main trap: a binary with folded constants ignores later settings, and 22 sweeps on such binaries measured nothing.
- **[quality-gate](skills/quality-gate/SKILL.md)**: accept an approximation only against a reference far better than both sides, a bar set by a cheap exact baseline, and a bias limit, and check the gain per unit of quality, not per sample. Here the cache more than tripled samples per minute, but its quality levelled off, and above that level the exact build produced the same image sooner.
- **[cache-bake](skills/cache-bake/SKILL.md)**: train a per-scene neural cache offline on unbiased targets, calibrate it on held-out seeds, freeze it, and choose where paths query it. Online training left a smooth bias, and more network capacity did not remove it.

## Future work

- **More scenes.** Every gain here is fit to the three benchmark scenes. The next step is scenes the method has not seen: derive the per-scene kernel choices from scene content, which already matched hand tuning on the exact track, and make the bake part of loading a new scene.
- **More hardware.** The kernels, launch bounds and register budgets are tuned for one H100. The same recipe should run on other compute and graphics GPUs, starting with the H200, B200 and RTX Pro 6000, where the balance between memory, registers and RT cores differs.

## Citation

```bibtex
@misc{chai2026rtxh100,
  title  = {{RTX H100}: Blender Cycles on One {H100}},
  author = {Chai, Wenhao and Chen, Liyan and Li, Zhifei and Guo, Wentao and
            You, Jiacheng and Li, Kunjun and Wei, Qingfu and Peng, Bo},
  year   = {2026},
  howpublished = {\url{https://github.com/wenhaochai/rtx-h100}}
}
```

## License

Apache-2.0, as Blender Cycles, which the patches modify. `nrc/patches` modify tiny-cuda-nn and CUTLASS, both BSD-3-Clause. The benchmark scenes in the animation are the Blender Open Data benchmark scenes, by their respective authors.
