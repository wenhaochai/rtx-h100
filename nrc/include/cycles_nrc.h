/* cycles_nrc.h: neural radiance cache (NRC) engine for Blender Cycles (plain C ABI).
 *
 * Owned by the engine task; the Cycles integration only reads it.
 *
 * Context: every call must run with Cycles' CUDA context current (inside a CUDAContextScope). Cycles 5.3.0
 * retains the device's primary context (device/cuda/device_impl.cpp: cuDevicePrimaryCtxRetain), so the engine's
 * CUDA runtime calls land in that same context. nrc_create records the current context; a later call made with
 * another context current pushes the recorded one for its duration.
 *
 * Streams: all work is enqueued on the stream passed to each call (a CUstream, e.g. Cycles' CU_STREAM_NON_BLOCKING
 * queue stream). Calls return without a host sync. Device pointers must stay valid until the stream passes the call.
 * nrc_create synchronises once (parameter init and workspace warm-up). One engine per device; not thread-safe.
 *
 * Inputs: NRC_IN fp16 features per query, row-major (query i at d_inputs + i * NRC_IN), all in [0,1]:
 *   [0..2]   position, normalised to the scene bounding box
 *   [3..4]   scattered (outgoing) direction, spherical: theta/pi, phi/(2 pi)
 *   [5..6]   shading normal, spherical, same mapping
 *   [7]      roughness: 1 - exp(-r)
 *   [8..10]  diffuse albedo, [11..13] specular albedo
 *   [14..15] padding (ignored)
 * Default encoding: hash grid on [0..2] (16 levels x 2 features, 2^19 entries per level, fp16), one-blob
 * (4 bins) on [3..7], identity on [8..13]. Output: n x 3 fp32 radiance, clamped at 0, from the EMA weights.
 *
 * Config (json_config may be NULL or ""): a JSON object whose "encoding", "network", "optimizer" and "loss" keys
 * replace the engine defaults key by key (tiny-cuda-nn syntax; the encoding sees the 14 used features).
 * "nrc": {"warmup_infer_batch": 262144, "warmup_train_batch": 16384, "seed": 1337} sizes the workspace at create
 * time so later calls never grow it (growth calls cudaDeviceSynchronize).
 * "engine": "tcnn" (default) | "wgmma" | "mma" picks the inference kernel; training is tcnn in every case. "wgmma" is the
 * fused sm_90a kernel (nrc_fused.cu; mma.sync on other GPUs), "mma" the same kernel with mma.sync. Fused engines need
 * the default model shape (any hidden-layer count 1-8, grid size, scale) and fall back if their create-time self check
 * against tcnn fails. "nrc": {"fused_grid_levels": -1 (by L2 size) | 0 (gathers in the kernel) | k, "fused_wg": 0|2|4}.
 * "engine": "infer" (nrc_infer.cu): frozen weights from the NRCINF1 file named by CYCLES_NRC_LOAD (or "nrc": {"weights":
 * path}), no tcnn model; hash grid or permutohedral lattice grid ("encoding": "permuto" in the file), MLP width
 * 64/128/256, depth 1-16, ReLU or SiLU, fp16 or fp8 (e4m3). sm_90 only (wgmma). nrc_train is a no-op returning 0.
 * "nrc": {"infer_nc": 0 (auto) | 2 | 3, "infer_silu": "ex2" (default) | "h2", "infer_grid_levels": -1 (auto) | 0 | k,
 * "infer_rs": 0 | 1, "infer_cluster": 1 (default) | 2}.
 *
 * Return values: nrc_create returns NULL on error; nrc_infer and nrc_train return 0 on success, -1 on error
 * (a message goes to stderr).
 */
#ifndef CYCLES_NRC_H
#define CYCLES_NRC_H

#ifdef __cplusplus
extern "C" {
#endif

#define NRC_IN 16 /* features per query, fp16 (layout above) */

typedef struct NrcEngine NrcEngine;

/* cu_stream: CUstream of Cycles' CUDA queue (driver API context = current). Returns NULL on error. */
NrcEngine *nrc_create(const char *json_config, void *cu_stream);

/* inputs: n x NRC_IN fp16 (row-major); out: n x 3 fp32 radiance. Uses the EMA weights. n == 0 is a no-op. */
int nrc_infer(NrcEngine *, int n, const void *d_inputs, float *d_rgb, void *cu_stream);

/* targets: n x 3 fp32. Runs `steps` Adam steps (relative L2 loss); step s trains on records s, s+steps, s+2*steps,
 * ... (ceil(n/steps) records, wrapped to a multiple of 256). The EMA weights are updated after every step. */
int nrc_train(NrcEngine *, int n, const void *d_inputs, const float *d_target_rgb, int steps, void *cu_stream);

void nrc_destroy(NrcEngine *);

#ifdef __cplusplus
}
#endif

#endif /* CYCLES_NRC_H */
