// nrc_infer.cu: frozen-weight NRC inference for wide, deep MLPs (see nrc_infer.h and analysis/nrc_infer.md).
//
// Two passes, as the fused engine's split mode: grid_pass writes the 16 hash-grid levels (half2 each, level-major),
// then one persistent kernel runs the MLP.
// MLP kernel: one CTA per SM = NC consumer warpgroups (64-query tiles each) + 1 producer warpgroup.
//   * Weights are cut into chunks of W rows x 128 bytes (a K-slab of 64 fp16 or 128 e4m3 columns, 128-byte swizzled,
//     K-major), pre-packed on the host in the order the layers consume them. The producer streams them with
//     cp.async.bulk (TMA, one instruction per chunk) into a ring of S stages guarded by full/empty mbarriers. When all
//     C chunks fit (C <= S), they are loaded once and stay resident.
//   * The producer warpgroup gives its registers to the consumers (setmaxnreg: 24 vs 240 / 160).
//   * A consumer warpgroup keeps its 64 x W activations in shared memory (the A operand) and the layer output in
//     registers: per chunk 4 x wgmma.m64nWk16 (fp16) or m64nWk32 (e4m3), SS, N = W (256 = Hopper's largest tile).
//     The epilogue (scale, SiLU via ex2.approx + rcp.approx or ReLU, cvt to fp16 or e4m3) writes the next layer's A
//     operand in place. The output layer (16 x W) stays resident in its own shared-memory area.
//   * All consumer warpgroups of a CTA read the same weight stage, so each chunk crosses L2 -> SM once per
//     NC x 64 queries; the warp scheduler overlaps one warpgroup's epilogue with another's MMAs.
#include "nrc_infer.h"

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cub/device/device_radix_sort.cuh>
#include <json/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>

namespace nrcw {
struct SortWs {
	uint32_t cap = 0;
	uint32_t *k0 = nullptr, *k1 = nullptr, *i0 = nullptr, *i1 = nullptr;
	__half* in = nullptr;
	float* out = nullptr;
	void* tmp = nullptr;
	size_t tmp_bytes = 0;
};

namespace {

constexpr int TILE = 64;
constexpr int N_LEVELS = 16;
constexpr int SMEM_MAX = 232448;  // H100: 227 KB dynamic shared memory per block

// ---------------------------------------------------------------- encoding (as nrc_fused.cu, tcnn's arithmetic)
__device__ inline float ex2_approx(float x) {
	float y;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}
__device__ inline float rcp_approx(float x) {
	float y;
	asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}
__device__ inline float quartic_cdf(float x, float inv_radius) {
	const float u = x * inv_radius, u2 = u * u, u4 = u2 * u2;
	return fmaxf(0.0f, fminf(1.0f, ((float)15 / 16) * u * (1 - ((float)2 / 3) * u2 + ((float)1 / 5) * u4) + 0.5f));
}
__device__ inline float blob_cdf(float b, float x) {
	return quartic_cdf(b - x, 4.0f) + quartic_cdf(b - x - 1.0f, 4.0f) + quartic_cdf(b - x + 1.0f, 4.0f);
}

struct GridArgs {
	const __half* in;
	const __half2* grid;
	uint32_t n;
	uint32_t hash_mask, dense_mask;
	uint32_t offsets[N_LEVELS + 1], res[N_LEVELS];
	float scale[N_LEVELS];
};

// Level parameters come from the weight file (the grid as trained): scale, resolution, offset, dense or hashed.
__device__ __forceinline__ __half2 grid_level(float x0, float x1, float x2, uint32_t l, const uint32_t* offsets,
                                              const uint32_t* resv, const float* scalev, uint32_t dense_mask,
                                              const __half2* __restrict__ grid, uint32_t hash_mask) {
	const float scale = scalev[l];
	const uint32_t res = resv[l];
	const uint32_t base = offsets[l], hsize = offsets[l + 1] - base;
	const bool hashed = !((dense_mask >> l) & 1);
	const float x[3] = {x0, x1, x2};
	float pos[3];
	uint32_t pg[3];
#pragma unroll
	for (int d = 0; d < 3; ++d) {
		const float p = fmaf(scale, x[d], 0.5f);
		const float fl = floorf(p);
		pg[d] = (uint32_t)(int)fl;
		pos[d] = p - fl;
	}
	__half2 val[8];
#pragma unroll
	for (int c = 0; c < 8; ++c) {
		const uint32_t q0 = pg[0] + (c & 1), q1 = pg[1] + ((c >> 1) & 1), q2 = pg[2] + ((c >> 2) & 1);
		uint32_t idx;
		if (hashed) {
			idx = (q0 ^ q1 * 2654435761u ^ q2 * 805459861u) & hash_mask;
		} else {
			idx = q0 + q1 * res + q2 * res * res;
			if (idx >= hsize) idx -= hsize;
		}
		val[c] = __ldg(grid + base + idx);
	}
	__half2 acc = __float2half2_rn(0.0f);
#pragma unroll
	for (int c = 0; c < 8; ++c) {
		float w = 1.0f;
#pragma unroll
		for (int d = 0; d < 3; ++d) w *= (c >> d) & 1 ? pos[d] : 1 - pos[d];
		acc = __hfma2(__half2half2(__float2half(w)), val[c], acc);
	}
	return acc;
}

__global__ void __launch_bounds__(256) grid_kernel(const GridArgs a, uint32_t levels_per_pass, __half2* __restrict__ enc) {
	__shared__ uint32_t offsets[N_LEVELS + 1], resv[N_LEVELS];
	__shared__ float scalev[N_LEVELS];
	if (threadIdx.x == 0) {
#pragma unroll
		for (int i = 0; i <= N_LEVELS; ++i) offsets[i] = a.offsets[i];
#pragma unroll
		for (int i = 0; i < N_LEVELS; ++i) { resv[i] = a.res[i]; scalev[i] = a.scale[i]; }
	}
	__syncthreads();
	const uint32_t row = blockIdx.x * 256 + threadIdx.x;
	if (row >= a.n) return;
	const uint2 v = __ldg(reinterpret_cast<const uint2*>(a.in + (size_t)row * 16));
	const float x0 = __half2float(__ushort_as_half((unsigned short)(v.x & 0xFFFF)));
	const float x1 = __half2float(__ushort_as_half((unsigned short)(v.x >> 16)));
	const float x2 = __half2float(__ushort_as_half((unsigned short)(v.y & 0xFFFF)));
	const uint32_t l0 = blockIdx.y * levels_per_pass, l1 = min(l0 + levels_per_pass, (uint32_t)N_LEVELS);
#pragma unroll 4
	for (uint32_t l = l0; l < l1; ++l)
		enc[(size_t)l * a.n + row] = grid_level(x0, x1, x2, l, offsets, resv, scalev, a.dense_mask, a.grid, a.hash_mask);
}

// 128-byte swizzle, K-major rows of 128 bytes: 16-byte chunk c of row r sits at chunk c ^ (r % 8).
__host__ __device__ inline uint32_t swz_byte(uint32_t r, uint32_t kbyte) {
	return r * 128 + ((((kbyte >> 4) ^ (r & 7)) << 4) | (kbyte & 15));
}

// ---------------------------------------------------------------- permutohedral lattice grid (encoding "permuto")
// Forward of nrc/bake/stream/permuto_triton.py (tcnn's PermutoEncoding) with the same fp32 operation order: per level l,
// c_j = (x_j + shift_lj) * scale_lj, elevate to the d+1 hyperplane, round to the nearest multiple of d+1, rank the
// remainders, wrap the ranks with rem0's sum, barycentric weights, the d+1 vertex keys hashed with tcnn's
// base_convert_hash (h = (h + key_j) * 2531011, uint32) into the level's 2^T rows of F fp16 features; fp32 blend.
// The pass writes the whole 64-wide MLP input of a query as 8 groups of 8 halves (16 B): group g of query i at
// enc[g * n + i]. Groups 0 .. L F / 8 - 1 hold the levels (level l at halves l F .. l F + F - 1), the rest the one-blob
// on x[d:8] (4 bins per dim, the hash path's arithmetic), the identity on x[8:14] and ones.
struct PermutoArgs {
	const __half* in;
	const __half* table;
	uint32_t n, hmask, log2_T, groups;  // groups = L F / 8
	uint32_t gps;                       // level groups per sweep (blockIdx.y)
	float scale[PERMUTO_MAX_LEVELS * PERMUTO_MAX_D], shift[PERMUTO_MAX_LEVELS * PERMUTO_MAX_D];
};

template <int F> struct PRow;
template <> struct PRow<2> { using T = uint32_t; };
template <> struct PRow<4> { using T = uint2; };
template <> struct PRow<8> { using T = uint4; };

__device__ __forceinline__ void unpack_h(uint32_t v, float* o) {
	o[0] = __half2float(__ushort_as_half((unsigned short)(v & 0xFFFF)));
	o[1] = __half2float(__ushort_as_half((unsigned short)(v >> 16)));
}
template <int F>
__device__ __forceinline__ void load_row(const __half* __restrict__ table, uint64_t row, float* v) {
	const typename PRow<F>::T r = __ldg(reinterpret_cast<const typename PRow<F>::T*>(table) + row);
	if constexpr (F == 2) unpack_h(r, v);
	else if constexpr (F == 4) { unpack_h(r.x, v); unpack_h(r.y, v + 2); }
	else { unpack_h(r.x, v); unpack_h(r.y, v + 2); unpack_h(r.z, v + 4); unpack_h(r.w, v + 6); }
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
	const __half2 h = __floats2half2_rn(a, b);
	return *reinterpret_cast<const uint32_t*>(&h);
}

// One level of one query: out[0 .. F-1] = sum over the d+1 vertices of weight x feature (fp32).
template <int D, int F>
__device__ __forceinline__ void permuto_level(const float* x, const float* __restrict__ sc, const float* __restrict__ sh,
                                              const __half* __restrict__ table, uint64_t lvo, uint32_t hmask, float* out) {
	constexpr int D1 = D + 1;
	constexpr float INV = 1.0f / D1;
	float c[D];
#pragma unroll
	for (int j = 0; j < D; ++j) c[j] = (x[j] + sh[j]) * sc[j];
	float e[D1];
	e[D] = 0.0f - (float)D * c[D - 1];
	float sm = c[D - 1];
#pragma unroll
	for (int k = D - 1; k > 0; --k) {
		e[k] = sm - (float)k * c[k - 1];
		sm = sm + c[k - 1];
	}
	e[0] = sm;
	int q[D1], rk[D1];
	float di[D1];
	int qsum = 0;
#pragma unroll
	for (int i = 0; i < D1; ++i) {
		const float v = e[i] * INV;
		const float up = ceilf(v) * (float)D1, dn = floorf(v) * (float)D1;
		const float rf = up - e[i] < e[i] - dn ? up : dn;
		q[i] = (int)rf;
		di[i] = e[i] - rf;
		rk[i] = 0;
		qsum += q[i];
	}
	const int qs = qsum / D1;  // exact: every q is a multiple of d+1
#pragma unroll
	for (int i = 0; i < D; ++i)
#pragma unroll
		for (int j = i + 1; j < D1; ++j) {
			const int t = di[i] < di[j];
			rk[i] += t;
			rk[j] += 1 - t;
		}
	float pr[D1];  // pr[r] = delta of the coordinate of rank r
#pragma unroll
	for (int r = 0; r < D1; ++r) pr[r] = 0.0f;
#pragma unroll
	for (int i = 0; i < D1; ++i) {
		rk[i] += qs;
		if (rk[i] < 0) { rk[i] += D1; q[i] += D1; }
		else if (rk[i] > D) { rk[i] -= D1; q[i] -= D1; }
		const float dl = (e[i] - (float)q[i]) * INV;
#pragma unroll
		for (int r = 0; r < D1; ++r) pr[r] = rk[i] == r ? dl : pr[r];
	}
	float acc[F];
#pragma unroll
	for (int f = 0; f < F; ++f) acc[f] = 0.0f;
#pragma unroll
	for (int kk = 0; kk <= D; ++kk) {
		// barycentric: bary[d - rank] += delta, bary[d + 1 - rank] -= delta; w0 = bary[0] + (1 + bary[d + 1])
		const float w = kk == 0 ? pr[D] + (1.0f + (0.0f - pr[0])) : pr[D - kk] - pr[D + 1 - kk];
		uint32_t h = 0;
#pragma unroll
		for (int j = 0; j < D; ++j) {
			const int key = q[j] + kk - (rk[j] > D - kk ? D1 : 0);
			h = (h + (uint32_t)key) * 2531011u;
		}
		float v[F];
		load_row<F>(table, (uint64_t)(h & hmask) + lvo, v);
#pragma unroll
		for (int f = 0; f < F; ++f) acc[f] += w * v[f];
	}
#pragma unroll
	for (int f = 0; f < F; ++f) out[f] = acc[f];
}

template <int D, int F>
__global__ void __launch_bounds__(256) permuto_kernel(const __grid_constant__ PermutoArgs a, uint4* __restrict__ enc) {
	constexpr int LPG = 8 / F;              // levels per 16-byte group
	constexpr int NB = 8 - D;               // one-blob dims
	constexpr int NT = 4 * NB + 6;          // one-blob bins + identity
	const uint32_t row = blockIdx.x * 256 + threadIdx.x;
	if (row >= a.n) return;
	const uint4* src = reinterpret_cast<const uint4*>(a.in + (size_t)row * 16);
	const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
	const uint32_t wv[7] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z};
	float x[14];
#pragma unroll
	for (int i = 0; i < 7; ++i) unpack_h(wv[i], x + 2 * i);
	const uint32_t g0 = blockIdx.y * a.gps, g1 = min(g0 + a.gps, a.groups);
#pragma unroll 1
	for (uint32_t g = g0; g < g1; ++g) {
		float o[8];
#pragma unroll
		for (int i = 0; i < LPG; ++i) {
			const uint32_t l = g * LPG + i;
			permuto_level<D, F>(x, a.scale + l * D, a.shift + l * D, a.table, (uint64_t)l << a.log2_T, a.hmask, o + i * F);
		}
		enc[(size_t)g * a.n + row] = make_uint4(pack_h2(o[0], o[1]), pack_h2(o[2], o[3]), pack_h2(o[4], o[5]), pack_h2(o[6], o[7]));
	}
	if (blockIdx.y + 1 != gridDim.y) return;
	float tv[NT];  // one-blob, identity
#pragma unroll
	for (int j = 0; j < NB; ++j) {
		float cdf[5];
#pragma unroll
		for (int b = 0; b < 5; ++b) cdf[b] = blob_cdf(b * 0.25f, x[D + j]);
#pragma unroll
		for (int b = 0; b < 4; ++b) tv[4 * j + b] = cdf[b + 1] - cdf[b];
	}
#pragma unroll
	for (int i = 0; i < 6; ++i) tv[4 * NB + i] = x[8 + i];
#pragma unroll
	for (int k = 0; k < 8; ++k) {
		if (a.groups + k >= 8) break;
		uint32_t w[4];
#pragma unroll
		for (int i = 0; i < 4; ++i) {
			const int s0 = 8 * k + 2 * i;
			w[i] = pack_h2(s0 < NT ? tv[s0] : 1.0f, s0 + 1 < NT ? tv[s0 + 1] : 1.0f);
		}
		enc[(size_t)(a.groups + k) * a.n + row] = make_uint4(w[0], w[1], w[2], w[3]);
	}
}

// The 64 x 64 fp16 tile of a pre-built encoding (permuto_kernel) -> act: thread t copies groups 4 (t / 64) .. + 3 of
// query t % 64.
__device__ __forceinline__ void copy_tile(const uint4* __restrict__ enc, uint32_t n, uint32_t tile, uint32_t t, uint8_t* act) {
	const uint32_t q = t & 63, part = t >> 6, row = tile * TILE + q;
#pragma unroll
	for (uint32_t i = 0; i < 4; ++i) {
		const uint32_t g = part * 4 + i;
		*reinterpret_cast<uint4*>(act + swz_byte(q, 16 * g)) = row < n ? __ldg(enc + (size_t)g * n + row) : make_uint4(0, 0, 0, 0);
	}
}

// The 64 x 64 fp16 encoding of one tile (grid features from grid_kernel) -> act (8 KB, swizzled). Thread t: query
// t % 64, levels 8 * (t / 64) .. + 7 and half of the one-blob dims.
struct GridSmem {  // level parameters in shared memory (in-kernel gathers)
	uint32_t offsets[N_LEVELS + 1], res[N_LEVELS];
	float scale[N_LEVELS];
};

__device__ __forceinline__ void encode_tile(const __half* __restrict__ in, uint32_t n, const __half2* __restrict__ enc,
                                            const __half2* __restrict__ grid, const GridSmem* gs, uint32_t dense_mask,
                                            uint32_t hash_mask, uint32_t tile, uint32_t t, uint8_t* act) {
	const uint32_t q = t & 63, part = t >> 6;
	const uint32_t row = tile * TILE + q;
	float x[14];
	__half2 v[8];
	if (row < n) {
		const uint4* src = reinterpret_cast<const uint4*>(in + (size_t)row * 16);
		const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
		const uint32_t w[7] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z};
#pragma unroll
		for (int i = 0; i < 7; ++i) {
			x[2 * i] = __half2float(__ushort_as_half((unsigned short)(w[i] & 0xFFFF)));
			x[2 * i + 1] = __half2float(__ushort_as_half((unsigned short)(w[i] >> 16)));
		}
		if (enc) {
#pragma unroll
			for (uint32_t li = 0; li < 8; ++li) v[li] = __ldg(enc + (size_t)(part * 8 + li) * n + row);
		} else {
#pragma unroll 4
			for (uint32_t li = 0; li < 8; ++li)  // gathers: store each level at once (no runtime-indexed array)
				*reinterpret_cast<__half2*>(act + swz_byte(q, 4 * (part * 8 + li))) =
					grid_level(x[0], x[1], x[2], part * 8 + li, gs->offsets, gs->res, gs->scale, dense_mask, grid, hash_mask);
		}
	} else {
#pragma unroll
		for (int i = 0; i < 14; ++i) x[i] = 0.0f;
#pragma unroll
		for (uint32_t li = 0; li < 8; ++li) v[li] = __float2half2_rn(0.0f);
	}
	auto st = [&](uint32_t k, __half2 val) { *reinterpret_cast<__half2*>(act + swz_byte(q, 2 * k)) = val; };
	if (enc || row >= n) {
#pragma unroll
		for (uint32_t li = 0; li < 8; ++li) st(2 * (part * 8 + li), v[li]);
	}
#pragma unroll
	for (uint32_t d = 0; d < 5; ++d) {
		if ((d >= 3) != (part != 0)) continue;
		const float xv = x[3 + d];
		float cdf[5];
#pragma unroll
		for (int b = 0; b < 5; ++b) cdf[b] = blob_cdf(b * 0.25f, xv);
		st(32 + 4 * d, __floats2half2_rn(cdf[1] - cdf[0], cdf[2] - cdf[1]));
		st(34 + 4 * d, __floats2half2_rn(cdf[3] - cdf[2], cdf[4] - cdf[3]));
	}
	if (part) {
#pragma unroll
		for (int i = 0; i < 3; ++i) st(52 + 2 * i, __floats2half2_rn(x[8 + 2 * i], x[9 + 2 * i]));
#pragma unroll
		for (int i = 0; i < 3; ++i) st(58 + 2 * i, __float2half2_rn(1.0f));
	}
}

// ---------------------------------------------------------------- sm_90 primitives
__device__ inline uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// K-major, 128-byte swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused (1), 1024-byte aligned atoms. A k-slice of
// 32 bytes inside the atom starts at +32 B (+2 in 16-byte units), as CUTLASS advances the descriptor.
__device__ inline uint64_t desc_sw128(uint32_t addr) {
	return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

__device__ inline void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" ::"r"(wg + 1) : "memory"); }
__device__ inline void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ inline void mbar_init(uint32_t bar, uint32_t count) {
	asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ inline void mbar_wait(uint32_t bar, uint32_t parity) {
	asm volatile(
		"{\n.reg .pred p;\nWAIT_%=:\n"
		"mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
		"@!p bra WAIT_%=;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ inline void mbar_arrive(uint32_t bar) {
	asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ inline void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
	asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// Multicast to the CTAs in mask (cluster): the bytes land at the same offset in each CTA, and each CTA's mbarrier at
// the same offset receives the complete_tx.
__device__ inline void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t mask) {
	asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
	             ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
}
__device__ inline void mbar_arrive_cluster(uint32_t bar, uint32_t cta) {
	asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}"
	             ::"r"(bar), "r"(cta) : "memory");
}
__device__ inline void cluster_sync() {
	asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ inline uint32_t cluster_rank() {
	uint32_t r;
	asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
	return r;
}
__device__ inline void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
	asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
	             ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

template <int N>
__device__ inline void fence_regs(float* d) {
#pragma unroll
	for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

using GK = cute::GMMA::Major;
template <int N> struct F16Op;
template <> struct F16Op<16> { using T = cute::SM90_64x16x16_F32F16F16_SS<GK::K, GK::K>; };
template <> struct F16Op<64> { using T = cute::SM90_64x64x16_F32F16F16_SS<GK::K, GK::K>; };
template <> struct F16Op<128> { using T = cute::SM90_64x128x16_F32F16F16_SS<GK::K, GK::K>; };
template <> struct F16Op<256> { using T = cute::SM90_64x256x16_F32F16F16_SS<GK::K, GK::K>; };
template <int N> struct F16RSOp;
template <> struct F16RSOp<16> { using T = cute::SM90_64x16x16_F32F16F16_RS<GK::K, GK::K>; };
template <> struct F16RSOp<64> { using T = cute::SM90_64x64x16_F32F16F16_RS<GK::K, GK::K>; };
template <> struct F16RSOp<128> { using T = cute::SM90_64x128x16_F32F16F16_RS<GK::K, GK::K>; };
template <> struct F16RSOp<256> { using T = cute::SM90_64x256x16_F32F16F16_RS<GK::K, GK::K>; };
template <int N> struct F8Op;
template <> struct F8Op<16> { using T = cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>; };
template <> struct F8Op<64> { using T = cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>; };
template <> struct F8Op<128> { using T = cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>; };
template <> struct F8Op<256> { using T = cute::SM90_64x256x32_F32E4M3E4M3_SS_TN<>; };

template <class Op, size_t... I>
__device__ __forceinline__ void mma_call(uint64_t a, uint64_t b, float* d, cute::GMMA::ScaleOut s, std::index_sequence<I...>) {
	Op::fma(a, b, d[I]..., s);
}

// 4 k-slices (one 128-byte chunk of K) of A (64 rows at a_addr) x B (N rows at b_addr) into d.
template <int N, bool F8>
__device__ __forceinline__ void mma_chunk(uint32_t a_addr, uint32_t b_addr, float* d, bool zero_first) {
	using Op = typename std::conditional<F8, typename F8Op<N>::T, typename F16Op<N>::T>::type;
	const uint64_t da = desc_sw128(a_addr), db = desc_sw128(b_addr);
#pragma unroll
	for (int kb = 0; kb < 4; ++kb)
		mma_call<Op>(da + 2 * kb, db + 2 * kb, d,
		             (zero_first && kb == 0) ? cute::GMMA::ScaleOut::Zero : cute::GMMA::ScaleOut::One,
		             std::make_index_sequence<N / 2>{});
}

// RS: A (64 x 16 fp16 per k-slice) from registers, 4 per thread: a[0] = row 16w + g, k 2tq + {0, 1}; a[1] = row
// 16w + g + 8, same k; a[2], a[3] = the same rows, k 8 + 2tq + {0, 1} (the mma.sync m16n8k16 A layout per warp).
template <class Op, size_t... I>
__device__ __forceinline__ void mma_call_rs(const uint32_t* a, uint64_t b, float* d, cute::GMMA::ScaleOut s, std::index_sequence<I...>) {
	Op::fma(a[0], a[1], a[2], a[3], b, d[I]..., s);
}
// 4 k-slices (one 128-byte chunk of K = 64) of A (16 registers: 4 per k-slice) x B (N rows at b_addr) into d.
template <int N>
__device__ __forceinline__ void mma_chunk_rs(const uint32_t* a16, uint32_t b_addr, float* d, bool zero_first) {
	using Op = typename F16RSOp<N>::T;
	const uint64_t db = desc_sw128(b_addr);
#pragma unroll
	for (int kb = 0; kb < 4; ++kb)
		mma_call_rs<Op>(a16 + 4 * kb, db + 2 * kb, d,
		                (zero_first && kb == 0) ? cute::GMMA::ScaleOut::Zero : cute::GMMA::ScaleOut::One,
		                std::make_index_sequence<N / 2>{});
}
template <int N>
__device__ inline void fence_u32(uint32_t* a) {
#pragma unroll
	for (int i = 0; i < N; ++i) asm volatile("" : "+r"(a[i])::"memory");
}

__device__ inline uint16_t cvt_e4m3x2(float lo, float hi) {
	uint16_t r;
	asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
	return r;
}

constexpr int ACT_SILU_T16 = 2;  // kernel template value for SILU_H2

template <int ACT>
__device__ __forceinline__ float act_fn(float x) {
	if constexpr (ACT == ACT_SILU) return x * rcp_approx(1.0f + ex2_approx(-1.4426950408889634f * x));
	else return fmaxf(x, 0.0f);
}

// SiLU on a pair in f16x2 without cancellation: e = 2^(-|x| log2 e) (one ex2.approx.f16x2 for two elements),
// s = sigmoid(-|x|) = e / (1 + e) with the reciprocal of 1 + e in [1, 2] by a linear guess and two Newton steps on the
// FMA pipe, sigmoid(x) = s (x < 0) or 1 - s (x >= 0), silu = x * sigmoid(x). (h + h tanh(h) loses the small negative
// outputs: tanh(h) rounds to -1 in fp16, which gave 4-14% error at depth >= 9.)
__device__ __forceinline__ uint32_t silu_h2(float x0, float x1) {
	const __half2 x = __floats2half2_rn(x0, x1);
	const uint32_t xu = *reinterpret_cast<const uint32_t*>(&x);
	const uint32_t au = xu & 0x7FFF7FFFu;
	const __half2 n = __hmul2(*reinterpret_cast<const __half2*>(&au), __float2half2_rn(-1.4426950408889634f));
	uint32_t eu;
	asm("ex2.approx.f16x2 %0, %1;" : "=r"(eu) : "r"(*reinterpret_cast<const uint32_t*>(&n)));
	const __half2 e = *reinterpret_cast<const __half2*>(&eu);
	const __half2 one = __float2half2_rn(1.0f), two = __float2half2_rn(2.0f);
	const __half2 y = __hadd2(one, e);
	__half2 r = __hfma2(y, __float2half2_rn(-8.0f / 17.0f), __float2half2_rn(24.0f / 17.0f));  // 1/y within 1/17
	r = __hmul2(r, __hfma2(__hneg2(y), r, two));
	r = __hmul2(r, __hfma2(__hneg2(y), r, two));
	const __half2 sg = __hmul2(e, r);           // sigmoid(-|x|)
	const __half2 sp = __hsub2(one, sg);        // sigmoid(|x|)
	const uint32_t neg = (xu >> 15) & 0x00010001u, m = neg * 0xFFFFu;
	const uint32_t su = (*reinterpret_cast<const uint32_t*>(&sg) & m) | (*reinterpret_cast<const uint32_t*>(&sp) & ~m);
	const __half2 yv = __hmul2(x, *reinterpret_cast<const __half2*>(&su));
	return *reinterpret_cast<const uint32_t*>(&yv);
}
__device__ inline uint16_t cvt_e4m3x2_h2(uint32_t h2) {
	uint16_t r;
	asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
	return r;
}

// Accumulator (m64nN, fp32) of thread (warp w, g = lane / 4, tq = lane % 4): d[4j + 0, 1] = row 16w + g, columns
// 8j + 2tq + {0, 1}; d[4j + 2, 3] = row 16w + g + 8. The epilogue writes act(d * cs[col]) as the next layer's A operand:
// fp16 (K-chunk of 64 columns = 8 KB) or e4m3 (K-chunk of 128 columns = 8 KB), both 128-byte swizzled.
template <int W, int ACT, bool OUT8>
__device__ __forceinline__ void epilogue(const float* d, uint8_t* act, uint32_t r0, uint32_t tq, const float* __restrict__ cs,
                                         float inv_a) {
	const uint32_t r1 = r0 + 8;
#pragma unroll
	for (int j = 0; j < W / 8; ++j) {
		float2 c = make_float2(1.0f, 1.0f);
		if (cs) c = __ldg(reinterpret_cast<const float2*>(cs + 8 * j + 2 * tq));
		if constexpr (ACT == ACT_SILU_T16) {
			uint32_t y01 = silu_h2(d[4 * j] * c.x, d[4 * j + 1] * c.y), y23 = silu_h2(d[4 * j + 2] * c.x, d[4 * j + 3] * c.y);
			if constexpr (OUT8) {
				const __half2 s = __float2half2_rn(inv_a);
				__half2 a01 = __hmul2(*reinterpret_cast<__half2*>(&y01), s), a23 = __hmul2(*reinterpret_cast<__half2*>(&y23), s);
				const uint32_t chunk = (j >> 4) * 8192, kb = ((j & 15) >> 1) * 16 + (j & 1) * 8 + 2 * tq;
				*reinterpret_cast<uint16_t*>(act + chunk + swz_byte(r0, kb)) = cvt_e4m3x2_h2(*reinterpret_cast<uint32_t*>(&a01));
				*reinterpret_cast<uint16_t*>(act + chunk + swz_byte(r1, kb)) = cvt_e4m3x2_h2(*reinterpret_cast<uint32_t*>(&a23));
			} else {
				const uint32_t chunk = (j >> 3) * 8192, kb = (j & 7) * 16 + 4 * tq;
				*reinterpret_cast<uint32_t*>(act + chunk + swz_byte(r0, kb)) = y01;
				*reinterpret_cast<uint32_t*>(act + chunk + swz_byte(r1, kb)) = y23;
			}
			continue;
		}
		const float v0 = act_fn<ACT>(d[4 * j] * c.x), v1 = act_fn<ACT>(d[4 * j + 1] * c.y);
		const float v2 = act_fn<ACT>(d[4 * j + 2] * c.x), v3 = act_fn<ACT>(d[4 * j + 3] * c.y);
		if constexpr (OUT8) {
			const uint32_t chunk = (j >> 4) * 8192, kb = ((j & 15) >> 1) * 16 + (j & 1) * 8 + 2 * tq;
			*reinterpret_cast<uint16_t*>(act + chunk + swz_byte(r0, kb)) = cvt_e4m3x2(v0 * inv_a, v1 * inv_a);
			*reinterpret_cast<uint16_t*>(act + chunk + swz_byte(r1, kb)) = cvt_e4m3x2(v2 * inv_a, v3 * inv_a);
		} else {
			const uint32_t chunk = (j >> 3) * 8192, kb = (j & 7) * 16 + 4 * tq;
			*reinterpret_cast<__half2*>(act + chunk + swz_byte(r0, kb)) = __floats2half2_rn(v0, v1);
			*reinterpret_cast<__half2*>(act + chunk + swz_byte(r1, kb)) = __floats2half2_rn(v2, v3);
		}
	}
}

// RS epilogue: act(d) packed to fp16 in place of the next layer's A operand. Accumulator column block j (8 columns,
// d[4j .. 4j+3]) is half of k-slice j / 2: rows g / g+8, columns 2tq, 2tq+1 of that half, which is exactly the A
// fragment, so no data moves between threads (as FlashAttention-3's P x V).
template <int W, int ACT>
__device__ __forceinline__ void epilogue_rs(const float* d, uint32_t* A) {
#pragma unroll
	for (int j = 0; j < W / 8; ++j) {
		uint32_t y01, y23;
		if constexpr (ACT == ACT_SILU_T16) {
			y01 = silu_h2(d[4 * j], d[4 * j + 1]);
			y23 = silu_h2(d[4 * j + 2], d[4 * j + 3]);
		} else {
			__half2 h01 = __floats2half2_rn(act_fn<ACT>(d[4 * j]), act_fn<ACT>(d[4 * j + 1]));
			__half2 h23 = __floats2half2_rn(act_fn<ACT>(d[4 * j + 2]), act_fn<ACT>(d[4 * j + 3]));
			y01 = *reinterpret_cast<uint32_t*>(&h01);
			y23 = *reinterpret_cast<uint32_t*>(&h23);
		}
		A[4 * (j >> 1) + 2 * (j & 1)] = y01;
		A[4 * (j >> 1) + 2 * (j & 1) + 1] = y23;
	}
}

struct KArgs {
	const __half* in;
	float* out;
	const __half2* enc;
	const uint8_t* chunks;  // C chunk images
	const uint8_t* outw;    // output-layer image
	const float* cs;        // fp8 epilogue scales
	const float* inv_a;     // fp8 activation scales (inverse)
	uint32_t n, n_tiles, depth, n_chunks, stages, out_bytes;
	uint32_t out_exp;
	// in-kernel gathers (enc == nullptr)
	const __half2* grid;
	uint32_t hash_mask, dense_mask;
	uint32_t offsets[N_LEVELS + 1], res[N_LEVELS];
	float scale[N_LEVELS];
};

template <int W, bool F8, bool RS = false>
struct Cfg {
	static constexpr uint32_t CHUNK = W * 128;                       // one K-slab of a W-row weight matrix
	static constexpr uint32_t CPL = F8 ? W / 128 : W / 64;           // chunks per hidden layer (and output layer)
	// RS keeps the activations in registers: shared memory holds only the 64 x 64 encoding tile of layer 0
	static constexpr uint32_t ACTB = RS ? 8192 : (F8 ? 64 * W : 128 * W) > 8192 ? (F8 ? 64 * W : 128 * W) : 8192;
	static constexpr uint32_t OUTB = CPL * 2048;
};

// CL = 2: a cluster of two CTAs (two SMs) shares the weight stream; each CTA's producer loads half of every chunk and
// multicasts it to both, so a chunk crosses L2 -> SM once per 2 x NC x 64 queries. A stage is refilled only after the
// consumers of both CTAs released it (empty barriers count NC x CL arrivals).
// PE (permuto): enc is the whole 64-wide input (permuto_kernel), copied by copy_tile; a template parameter, so the hash
// kernels compile exactly as before.
template <int W, bool F8, int ACT, int NC, int CL, bool RS, bool PE = false>
__global__ void __launch_bounds__(NC * 128 + 128, 1) mlp_kernel(const KArgs a) {
#if defined(__CUDA_ARCH__) && defined(CUTE_ARCH_MMA_SM90A_ENABLED)
	static_assert(!(RS && F8), "RS is fp16 only");
	using C = Cfg<W, F8, RS>;
	extern __shared__ uint8_t smem_raw[];
	uint8_t* base = reinterpret_cast<uint8_t*>(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
	uint8_t* stages = base;
	uint8_t* outw = stages + a.stages * C::CHUNK;
	uint8_t* acts = outw + C::OUTB;
	uint64_t* bars = reinterpret_cast<uint64_t*>(acts + NC * C::ACTB);  // full[S], empty[S]
	GridSmem* gs = reinterpret_cast<GridSmem*>(bars + 2 * a.stages);
	const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * a.stages;
	const bool resident = a.n_chunks <= a.stages;

	for (uint32_t c = threadIdx.x; c < a.out_bytes / 16; c += blockDim.x)
		reinterpret_cast<uint4*>(outw)[c] = __ldg(reinterpret_cast<const uint4*>(a.outw) + c);
	if (threadIdx.x < N_LEVELS) {  // static indices into a (a runtime index would copy the parameters to local memory)
#pragma unroll
		for (int l = 0; l < N_LEVELS; ++l)
			if (threadIdx.x == l) { gs->offsets[l] = a.offsets[l]; gs->res[l] = a.res[l]; gs->scale[l] = a.scale[l]; }
		if (threadIdx.x == 0) gs->offsets[N_LEVELS] = a.offsets[N_LEVELS];
	}
	if (threadIdx.x == 0) {
		for (uint32_t s = 0; s < a.stages; ++s) {
			mbar_init(full0 + 8 * s, 1);
			mbar_init(empty0 + 8 * s, NC * CL);
		}
		asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
	}
	fence_async_smem();
	__syncthreads();
	if constexpr (CL > 1) cluster_sync();  // the peer's barriers are initialised before any multicast or remote arrive

	const uint32_t wg = threadIdx.x >> 7;
	const uint32_t step = gridDim.x * NC;
	const uint32_t lead = blockIdx.x & ~(uint32_t)(CL - 1);  // both CTAs of a cluster run the lead CTA's rounds
	const uint32_t rounds = a.n_tiles > lead * NC ? (a.n_tiles - lead * NC + step - 1) / step : 0;

	if (wg == NC) {  // producer warpgroup: one thread issues the copies; registers go to the consumers
		asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory");
		if (threadIdx.x == NC * 128) {
			const uint32_t rank = CL > 1 ? cluster_rank() : 0, part = C::CHUNK / CL;
			uint32_t g = 0;
			for (uint32_t r = 0; r < (resident ? 1u : rounds); ++r) {
				for (uint32_t i = 0; i < a.n_chunks; ++i, ++g) {
					const uint32_t s = g % a.stages;
					if (g >= a.stages) mbar_wait(empty0 + 8 * s, ((g / a.stages) - 1) & 1);
					mbar_expect_tx(full0 + 8 * s, C::CHUNK);
					if constexpr (CL > 1)
						bulk_g2s_mc(smem_u32(stages + s * C::CHUNK + rank * part), a.chunks + (size_t)i * C::CHUNK + rank * part,
						            part, full0 + 8 * s, (uint16_t)((1u << CL) - 1));
					else
						bulk_g2s(smem_u32(stages + s * C::CHUNK), a.chunks + (size_t)i * C::CHUNK, C::CHUNK, full0 + 8 * s);
				}
			}
		}
	} else {

	if constexpr (NC == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n" ::: "memory");
	else asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n" ::: "memory");
	const uint32_t t = threadIdx.x & 127, warp = t >> 5, lane = t & 31, g8 = lane >> 2, tq = lane & 3;
	uint8_t* act = acts + wg * C::ACTB;
	const uint32_t act_a = smem_u32(act), stage_a = smem_u32(stages), out_a = smem_u32(outw);
	const uint32_t r0 = warp * 16 + g8;
	uint32_t gc = 0;  // chunks consumed (streaming)
	auto release = [&](uint32_t bar) {  // the stage's empty barrier in every CTA of the cluster
		if constexpr (CL > 1) {
#pragma unroll
			for (uint32_t c = 0; c < CL; ++c) mbar_arrive_cluster(bar, c);
		} else {
			mbar_arrive(bar);
		}
	};

	// One chunk from the ring: wait full, 4 k-slices, commit; release the previous chunk once its MMAs are done.
	auto consume = [&](uint32_t a_off, float* d, bool first, bool f8) {
		const uint32_t i = resident ? (gc % a.n_chunks) : gc;
		const uint32_t s = i % a.stages;
		mbar_wait(full0 + 8 * s, resident ? 0u : (i / a.stages) & 1);
		fence_regs<W / 2>(d);
		cute::warpgroup_arrive();
		if (f8) {
			if constexpr (F8) mma_chunk<W, true>(act_a + a_off, stage_a + s * C::CHUNK, d, first);
		} else {
			mma_chunk<W, false>(act_a + a_off, stage_a + s * C::CHUNK, d, first);
		}
		cute::warpgroup_commit_batch();
		if (!resident && !first) {
			cute::warpgroup_wait<1>();
			if (t == 0) release(empty0 + 8 * ((i - 1) % a.stages));
		}
		++gc;
	};
	auto finish_layer = [&](float* d) {
		cute::warpgroup_wait<0>();
		fence_regs<W / 2>(d);
		if (!resident && t == 0) release(empty0 + 8 * ((gc - 1) % a.stages));
	};

	for (uint32_t r = 0; r < rounds; ++r) {
		const uint32_t tile = (r * gridDim.x + blockIdx.x) * NC + wg;
		if constexpr (PE) copy_tile(reinterpret_cast<const uint4*>(a.enc), a.n, tile, t, act);
		else encode_tile(a.in, a.n, a.enc, a.grid, gs, a.dense_mask, a.hash_mask, tile, t, act);
		fence_async_smem();
		wg_bar(wg);

		float d[W / 2];
		consume(0, d, true, false);  // layer 0: fp16, K = 64
		finish_layer(d);
		if constexpr (RS) {
			uint32_t A[W / 4];
			wg_bar(wg);  // layer 0 has read the encoding tile
			epilogue_rs<W, ACT>(d, A);
			for (uint32_t l = 1; l < a.depth; ++l) {
				fence_u32<W / 4>(A);
#pragma unroll
				for (uint32_t c = 0; c < C::CPL; ++c) {
					const uint32_t i = resident ? (gc % a.n_chunks) : gc;
					const uint32_t s = i % a.stages;
					mbar_wait(full0 + 8 * s, resident ? 0u : (i / a.stages) & 1);
					fence_regs<W / 2>(d);
					cute::warpgroup_arrive();
					mma_chunk_rs<W>(A + 16 * c, stage_a + s * C::CHUNK, d, c == 0);
					cute::warpgroup_commit_batch();
					if (!resident && c > 0) {
						cute::warpgroup_wait<1>();
						if (t == 0) release(empty0 + 8 * ((i - 1) % a.stages));
					}
					++gc;
				}
				finish_layer(d);
				fence_u32<W / 4>(A);
				epilogue_rs<W, ACT>(d, A);
			}
			float o[8];
			fence_regs<8>(o);
			fence_u32<W / 4>(A);
			cute::warpgroup_arrive();
#pragma unroll
			for (uint32_t c = 0; c < C::CPL; ++c) mma_chunk_rs<16>(A + 16 * c, out_a + c * 2048, o, c == 0);
			cute::warpgroup_commit_batch();
			cute::warpgroup_wait<0>();
			fence_regs<8>(o);
			if (tq < 2) {
				const uint32_t q0 = tile * TILE + r0, q1 = q0 + 8;
				auto fin = [&](float v) { return a.out_exp ? ex2_approx(1.4426950408889634f * v) : fmaxf(v, 0.0f); };
				if (q0 < a.n) {
					a.out[(size_t)q0 * 3 + 2 * tq] = fin(o[0]);
					if (tq == 0) a.out[(size_t)q0 * 3 + 1] = fin(o[1]);
				}
				if (q1 < a.n) {
					a.out[(size_t)q1 * 3 + 2 * tq] = fin(o[2]);
					if (tq == 0) a.out[(size_t)q1 * 3 + 1] = fin(o[3]);
				}
			}
			continue;
		}
		if (F8) epilogue<W, ACT, F8>(d, act, r0, tq, nullptr, __ldg(a.inv_a));
		else epilogue<W, ACT, false>(d, act, r0, tq, nullptr, 1.0f);
		fence_async_smem();
		wg_bar(wg);
		for (uint32_t l = 1; l < a.depth; ++l) {
#pragma unroll 1
			for (uint32_t c = 0; c < C::CPL; ++c) consume(c * 8192, d, c == 0, F8);
			finish_layer(d);
			if constexpr (F8) epilogue<W, ACT, true>(d, act, r0, tq, a.cs + (l - 1) * W, __ldg(a.inv_a + l));
			else epilogue<W, ACT, false>(d, act, r0, tq, nullptr, 1.0f);
			fence_async_smem();
			wg_bar(wg);
		}
		// output layer (resident): 16 x W
		float o[8];
		fence_regs<8>(o);
		cute::warpgroup_arrive();
#pragma unroll
		for (uint32_t c = 0; c < C::CPL; ++c) mma_chunk<16, F8>(act_a + c * 8192, out_a + c * 2048, o, c == 0);
		cute::warpgroup_commit_batch();
		cute::warpgroup_wait<0>();
		fence_regs<8>(o);
		wg_bar(wg);  // every warp has finished reading act before the next tile's encoding overwrites it
		if (tq < 2) {
			float s0 = 1.0f, s1 = 1.0f;
			if constexpr (F8) {
				const float* cso = a.cs + (a.depth - 1) * W;
				s0 = __ldg(cso + 2 * tq);
				s1 = __ldg(cso + 2 * tq + 1);
			}
			const uint32_t q0 = tile * TILE + r0, q1 = q0 + 8;
			auto fin = [&](float v) { return a.out_exp ? ex2_approx(1.4426950408889634f * v) : fmaxf(v, 0.0f); };
			if (q0 < a.n) {
				a.out[(size_t)q0 * 3 + 2 * tq] = fin(o[0] * s0);
				if (tq == 0) a.out[(size_t)q0 * 3 + 1] = fin(o[1] * s1);
			}
			if (q1 < a.n) {
				a.out[(size_t)q1 * 3 + 2 * tq] = fin(o[2] * s0);
				if (tq == 0) a.out[(size_t)q1 * 3 + 1] = fin(o[3] * s1);
			}
		}
	}
	}  // consumers
	if constexpr (CL > 1) cluster_sync();  // no CTA exits while its peer may still multicast into it or arrive on it
#else
	__trap();
#endif
}

// ---------------------------------------------------------------- host
template <int W, bool F8, int ACT, int CL, bool RS = false, bool PE = false>
const void* pick_nc(int nc) {
	if constexpr (W <= 128) {  // W = 256 keeps 128 accumulator registers: 3 warpgroups + producer exceed 64K registers
		if (nc == 3) return (const void*)mlp_kernel<W, F8, ACT, 3, CL, RS, PE>;
	}
	return nc == 2 ? (const void*)mlp_kernel<W, F8, ACT, 2, CL, RS, PE> : nullptr;
}
template <int W, bool F8, int CL>
const void* pick_act(int act, int nc) {
	if (act == ACT_SILU_T16) return pick_nc<W, F8, ACT_SILU_T16, CL>(nc);
	return act == ACT_SILU ? pick_nc<W, F8, ACT_SILU, CL>(nc) : pick_nc<W, F8, ACT_RELU, CL>(nc);
}
template <int CL>
const void* pick_cl(int W, bool f8, int act, int nc) {
	if (W == 64) return f8 ? nullptr : pick_act<64, false, CL>(act, nc);
	if (W == 128) return f8 ? pick_act<128, true, CL>(act, nc) : pick_act<128, false, CL>(act, nc);
	if (W == 256) return f8 ? pick_act<256, true, CL>(act, nc) : pick_act<256, false, CL>(act, nc);
	return nullptr;
}
// RS: fp16, no cluster
const void* pick_rs(int W, int act, int nc) {
	auto by_act = [&](auto w) -> const void* {
		constexpr int WW = decltype(w)::value;
		if (act == ACT_SILU_T16) return pick_nc<WW, false, ACT_SILU_T16, 1, true>(nc);
		return act == ACT_SILU ? pick_nc<WW, false, ACT_SILU, 1, true>(nc) : pick_nc<WW, false, ACT_RELU, 1, true>(nc);
	};
	if (W == 128) return by_act(std::integral_constant<int, 128>{});
	if (W == 256) return by_act(std::integral_constant<int, 256>{});
	return nullptr;
}
// permuto: fp16, no cluster, SS (W = 64, 128, 256) or RS (W = 128, 256)
const void* pick_pe(int W, int act, int nc, bool rs) {
	auto by_act = [&](auto w, auto r) -> const void* {
		constexpr int WW = decltype(w)::value;
		constexpr bool R = decltype(r)::value;
		if (act == ACT_SILU_T16) return pick_nc<WW, false, ACT_SILU_T16, 1, R, true>(nc);
		return act == ACT_SILU ? pick_nc<WW, false, ACT_SILU, 1, R, true>(nc) : pick_nc<WW, false, ACT_RELU, 1, R, true>(nc);
	};
	using T = std::true_type; using F = std::false_type;
	if (W == 64) return rs ? nullptr : by_act(std::integral_constant<int, 64>{}, F{});
	if (W == 128) return rs ? by_act(std::integral_constant<int, 128>{}, T{}) : by_act(std::integral_constant<int, 128>{}, F{});
	if (W == 256) return rs ? by_act(std::integral_constant<int, 256>{}, T{}) : by_act(std::integral_constant<int, 256>{}, F{});
	return nullptr;
}
const void* pick_kernel(int W, bool f8, int act, int nc, int cl, bool rs = false, bool pe = false) {
	if (pe) return (f8 || cl != 1) ? nullptr : pick_pe(W, act, nc, rs);
	if (rs) return (f8 || cl != 1) ? nullptr : pick_rs(W, act, nc);
	return cl == 2 ? pick_cl<2>(W, f8, act, nc) : pick_cl<1>(W, f8, act, nc);
}

// Row-major (rows x k_total, elem bytes) matrix, K-slab ks of 128 bytes -> swizzled image (rows x 128 B).
void pack_slab(const uint8_t* m, int rows, int k_total, int elem, int ks, uint8_t* dst) {
	const int kper = 128 / elem;
	for (int r = 0; r < rows; ++r)
		for (int kk = 0; kk < kper; ++kk) {
			const int k = ks * kper + kk;
			for (int b = 0; b < elem; ++b) dst[swz_byte(r, kk * elem + b)] = k < k_total ? m[((size_t)r * k_total + k) * elem + b] : 0;
		}
}

uint32_t grid_offsets(uint32_t log2_hashmap, uint32_t base_res, float pls, uint32_t* offsets) {
	uint32_t off = 0;
	for (int l = 0; l < N_LEVELS; ++l) {
		const float scale = exp2f(l * std::log2(pls)) * base_res - 1.0f;
		const uint32_t res = (uint32_t)ceilf(scale) + 1;
		const uint32_t max_params = 0xFFFFFFFFu / 2;
		uint32_t in_level = std::pow((float)res, 3.0f) > (float)max_params ? max_params : res * res * res;
		in_level = (in_level + 7) / 8 * 8;
		in_level = std::min(in_level, 1u << log2_hashmap);
		offsets[l] = off;
		off += in_level;
	}
	offsets[N_LEVELS] = off;
	return off;
}

template <class T>
bool upload(T** dst, const void* src, size_t bytes) {
	if (cudaMalloc((void**)dst, std::max<size_t>(bytes, 16)) != cudaSuccess) return false;
	return cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

bool load_nrcinf(const char* path, HostModel& m, std::string* err) {
	std::ifstream f(path, std::ios::binary);
	if (!f) { *err = std::string{"cannot read "} + path; return false; }
	char magic[8];
	uint64_t hl = 0;
	f.read(magic, 8);
	if (!f || memcmp(magic, "NRCINF1\0", 8) != 0) { *err = std::string{path} + " is not an NRCINF1 file"; return false; }
	f.read((char*)&hl, 8);
	std::string hs(hl, '\0');
	f.read(hs.data(), hl);
	std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	nlohmann::json h;
	try { h = nlohmann::json::parse(hs); } catch (const std::exception& e) { *err = e.what(); return false; }
	m.header = h.dump();
	m.width = h.value("width", 0);
	m.depth = h.value("depth", 0);
	const std::string act = h.value("act", "relu"), oact = h.value("out_act", "linear"), prec = h.value("precision", "fp16");
	if (act != "relu" && act != "silu") { *err = "act must be relu or silu"; return false; }
	if (oact != "linear" && oact != "exp") { *err = "out_act must be linear or exp"; return false; }
	if (prec != "fp16" && prec != "fp8") { *err = "precision must be fp16 or fp8"; return false; }
	m.act = act == "silu" ? ACT_SILU : ACT_RELU;
	m.out_act = oact == "exp" ? OUT_EXP : OUT_LINEAR;
	m.fp8 = prec == "fp8";
	const std::string enc = h.value("encoding", "hash");
	if (enc != "hash" && enc != "permuto") { *err = "encoding must be hash or permuto"; return false; }
	m.encoding = enc == "permuto" ? ENC_PERMUTO : ENC_HASH;
	if (h.value("mlp_in", 64) != 64) { *err = "only a 64-wide MLP input is supported (mlp_in " + std::to_string(h.value("mlp_in", 0)) + ")"; return false; }
	const int W = m.width, D = m.depth, el = m.fp8 ? 1 : 2;
	if (!(W == 64 || W == 128 || W == 256) || D < 1 || D > 16 || (m.fp8 && W < 128)) { *err = "unsupported width/depth/precision"; return false; }
	auto sec = [&](const char* name, size_t bytes, void* dst) {
		const auto& s = h["sections"][name];
		const size_t off = s[0], len = s[1];
		if (len != bytes || off + len > blob.size()) throw std::runtime_error{std::string{"section "} + name + " has the wrong size"};
		memcpy(dst, blob.data() + off, len);
	};
	if (m.encoding == ENC_PERMUTO) {
		try {
			const auto& g = h.at("permuto");
			m.p_d = g.at("d"); m.p_levels = g.at("n_levels"); m.p_features = g.at("n_features"); m.p_log2_T = g.at("log2_hashmap");
			const int d = m.p_d, L = m.p_levels, F = m.p_features;
			if (d != 3 && d != 5 && d != 8) throw std::runtime_error{"permuto: d must be 3, 5 or 8"};
			if (F != 2 && F != 4 && F != 8) throw std::runtime_error{"permuto: n_features must be 2, 4 or 8"};
			if (L < 1 || L > PERMUTO_MAX_LEVELS || (L * F) % 8) throw std::runtime_error{"permuto: n_levels x n_features must be a multiple of 8, at most 16 levels"};
			if (L * F + 4 * (8 - d) + 6 > 64) throw std::runtime_error{"permuto: the encoding does not fit a 64-wide MLP input"};
			if (m.p_log2_T < 1 || m.p_log2_T > 26) throw std::runtime_error{"permuto: log2_hashmap out of range"};
			const auto& in = h.at("inputs");
			if (in.at("lattice") != nlohmann::json::array({0, d}) || in.at("one_blob") != nlohmann::json::array({d, 8}) ||
			    in.value("one_blob_bins", 4) != 4 || in.at("identity") != nlohmann::json::array({8, 14}))
				throw std::runtime_error{"permuto: inputs must be lattice x[0:d], one-blob (4 bins) x[d:8], identity x[8:14]"};
			const auto& hs = g.at("hash");
			if (hs.value("multiplier", 0u) != 2531011u || hs.value("init", 1u) != 0u) throw std::runtime_error{"permuto: unsupported hash"};
			m.n_entries = g.at("n_entries");
			if ((uint64_t)m.n_entries != ((uint64_t)L << m.p_log2_T)) throw std::runtime_error{"permuto: n_entries must be n_levels << log2_hashmap"};
			m.p_scale.clear(); m.p_shift.clear();
			for (const char* k : {"scale", "shift"}) {
				const auto& M = g.at(k);
				if (M.size() != (size_t)L) throw std::runtime_error{std::string{"permuto: "} + k + " must have n_levels rows"};
				for (int l = 0; l < L; ++l) {
					if (M[l].size() != (size_t)d) throw std::runtime_error{std::string{"permuto: "} + k + " rows must have d values"};
					for (int j = 0; j < d; ++j) (k[1] == 'c' ? m.p_scale : m.p_shift).push_back((float)M[l][j].get<double>());
				}
			}
			m.grid.resize((size_t)m.n_entries * F);
			sec("grid", m.grid.size() * 2, m.grid.data());
		} catch (const std::exception& e) { *err = e.what(); return false; }
	} else {
		const auto& g = h["grid"];
		if (g.value("n_levels", 0) != 16 || g.value("n_features", 0) != 2) { *err = "grid must have 16 levels x 2 features"; return false; }
		m.log2_hashmap = g.value("log2_hashmap", 19u);
		m.base_resolution = g.value("base_resolution", 16u);
		m.per_level_scale = g.value("per_level_scale", 1.5f);
		m.n_entries = g.value("n_entries", 0u);
		try {
			// per-level grid: the file's "levels" (as trained), else tcnn's float arithmetic
			grid_offsets(m.log2_hashmap, m.base_resolution, m.per_level_scale, m.level_off);
			for (int l = 0; l < N_LEVELS; ++l) {
				m.level_scale[l] = exp2f(l * std::log2(m.per_level_scale)) * m.base_resolution - 1.0f;
				m.level_res[l] = (uint32_t)ceilf(m.level_scale[l]) + 1;
				m.level_size[l] = m.level_off[l + 1] - m.level_off[l];
			}
			if (g.contains("levels")) {
				const auto& L = g["levels"];
				if (L.size() != N_LEVELS) throw std::runtime_error{"grid.levels must have 16 entries"};
				for (int l = 0; l < N_LEVELS; ++l) {
					m.level_scale[l] = L[l][0]; m.level_res[l] = L[l][1]; m.level_size[l] = L[l][2]; m.level_off[l] = L[l][3];
				}
				m.level_off[N_LEVELS] = m.level_off[N_LEVELS - 1] + m.level_size[N_LEVELS - 1];
			}
			for (int l = 0; l < N_LEVELS; ++l) {
				const uint64_t r = m.level_res[l];
				m.level_dense[l] = r <= 0x659 && r * r * r <= m.level_size[l];
				if (!m.level_dense[l] && m.level_size[l] != (1u << m.log2_hashmap)) throw std::runtime_error{"hashed level size must be 2^log2_hashmap"};
			}
			if (m.level_off[N_LEVELS] != m.n_entries) throw std::runtime_error{"grid size mismatch"};
			m.grid.resize((size_t)m.n_entries * 2);
			sec("grid", m.grid.size() * 2, m.grid.data());
		} catch (const std::exception& e) { *err = e.what(); return false; }
	}
	try {
		m.w0.resize((size_t)W * 64);
		sec("w0", m.w0.size() * 2, m.w0.data());
		m.hidden.resize((size_t)(D - 1) * W * W * el);
		if (D > 1) sec("hidden", m.hidden.size(), m.hidden.data());
		m.out.resize((size_t)16 * W * el);
		sec("out", m.out.size(), m.out.data());
		if (m.fp8) {
			m.w_scale.resize((size_t)(D - 1) * W + 16);
			sec("w_scale", m.w_scale.size() * 4, m.w_scale.data());
			m.a_scale.resize(D);
			sec("a_scale", m.a_scale.size() * 4, m.a_scale.data());
		}
	} catch (const std::exception& e) { *err = e.what(); return false; }
	return true;
}

bool create(Plan& p, const HostModel& m, int nc, std::string* err, SiluImpl silu, int cluster, bool rs) {
	int dev, major, minor, sms;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
	cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	if (!(major == 9 && minor == 0)) { *err = "nrc_infer needs sm_90 (H100)"; return false; }
	const int W = m.width, D = m.depth, el = m.fp8 ? 1 : 2;
	const uint32_t CHUNK = W * 128, CPL = m.fp8 ? W / 128 : W / 64;
	if (m.encoding == ENC_PERMUTO) {
		if (m.fp8) { *err = "nrc_infer: permuto encoding needs fp16 weights"; return false; }
		cluster = 1;
	}
	if (rs && (m.fp8 || cluster != 1 || W < 128)) rs = false;  // RS covers fp16 at width 128 and 256 without a cluster; else SS
	const uint32_t ACTB = rs ? 8192u : std::max<uint32_t>(8192, m.fp8 ? 64 * W : 128 * W), OUTB = CPL * 2048;
	const uint32_t C = 1 + (D - 1) * CPL;

	// chunk images in consumption order: layer 0 (one chunk, fp16 K = 64), then CPL K-slabs per hidden layer
	std::vector<uint8_t> img((size_t)C * CHUNK, 0);
	pack_slab(reinterpret_cast<const uint8_t*>(m.w0.data()), W, 64, 2, 0, img.data());
	for (int l = 1; l < D; ++l)
		for (uint32_t c = 0; c < CPL; ++c)
			pack_slab(m.hidden.data() + (size_t)(l - 1) * W * W * el, W, W, el, c, img.data() + (size_t)(1 + (l - 1) * CPL + c) * CHUNK);
	std::vector<uint8_t> oimg(OUTB, 0);
	for (uint32_t c = 0; c < CPL; ++c) pack_slab(m.out.data(), 16, W, el, c, oimg.data() + c * 2048);
	std::vector<float> cs, inv_a;
	if (m.fp8) {
		cs.resize((size_t)(D - 1) * W + 16);
		for (int l = 1; l <= D; ++l) {
			const int nout = l < D ? W : 16;
			for (int n = 0; n < nout; ++n) cs[(size_t)(l - 1) * W + n] = m.w_scale[(size_t)(l - 1) * W + n] * m.a_scale[l - 1];
		}
		for (int i = 0; i < D; ++i) inv_a.push_back(1.0f / m.a_scale[i]);
	}

	auto try_nc = [&](int k, Plan& q) -> bool {
		const void* kern = pick_kernel(W, m.fp8, m.act == ACT_SILU ? (silu == SILU_H2 ? ACT_SILU_T16 : ACT_SILU) : ACT_RELU, k, cluster, rs,
		                               m.encoding == ENC_PERMUTO);
		if (!kern) return false;
		const uint32_t fixed = 1024 + OUTB + k * ACTB + (uint32_t)sizeof(GridSmem);
		int64_t room = (int64_t)SMEM_MAX - fixed;
		int S = (int)std::min<int64_t>(C, (room - 16 * 8) / (CHUNK + 16));
		if (S < 2 && S < (int)C) return false;
		const size_t smem = fixed + (size_t)S * CHUNK + 16 * (size_t)S;
		if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) { cudaGetLastError(); return false; }
		int bps = 0;
		if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, k * 128 + 128, smem) != cudaSuccess || bps < 1) { cudaGetLastError(); return false; }
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, kern);
		q.kernel = kern; q.nc = k; q.cluster = cluster; q.stages = S; q.smem = smem; q.grid = sms * bps; q.regs = attr.numRegs;
		q.rs = rs;
		q.resident = (int)C <= S;
		return true;
	};
	Plan best;
	bool ok = false;
	for (int k : {2, 3}) {
		if (nc && nc != k) continue;
		Plan q;
		if (!try_nc(k, q)) continue;
		// auto: more consumer warpgroups first (H100: 3 beat 2 at W <= 128 even when 2 keep the weights resident)
		if (!ok || q.nc > best.nc || (q.nc == best.nc && q.resident && !best.resident)) { best = q; ok = true; }
	}
	if (!ok) { *err = "nrc_infer: no launch configuration fits"; return false; }
	p = best;
	p.sort_ws = new SortWs;
	p.levels_per_pass = W <= 128 ? 0 : 4;  // H100: in-kernel gathers win at W <= 128, the split grid pass at W = 256
	p.width = W; p.depth = D; p.fp8 = m.fp8; p.act = m.act; p.silu = silu; p.out_act = m.out_act; p.chunks = (int)C;
	for (int l = 0; l <= N_LEVELS; ++l) p.offsets[l] = m.level_off[l];
	p.dense_mask = 0;
	for (int l = 0; l < N_LEVELS; ++l) { p.res[l] = m.level_res[l]; p.scale[l] = m.level_scale[l]; p.dense_mask |= (uint32_t)m.level_dense[l] << l; }
	p.hash_mask = (1u << m.log2_hashmap) - 1;
	p.encoding = m.encoding;
	if (m.encoding == ENC_PERMUTO) {
		p.p_d = m.p_d; p.p_levels = m.p_levels; p.p_features = m.p_features; p.p_log2_T = m.p_log2_T;
		for (size_t i = 0; i < m.p_scale.size(); ++i) { p.p_scale[i] = m.p_scale[i]; p.p_shift[i] = m.p_shift[i]; }
		// one 16-byte group per sweep (4 levels at F = 2, 1 at F = 8): on 2^20 ordered monster queries, grid pass
		// 0.746 vs 0.779 ns (permuto3 F2), 2.320 vs 2.438 (permuto5 F2), 0.645 vs 0.740 (permuto5 F8), 1.442 vs 1.700
		// (permuto8 F8) against all levels in one sweep
		p.levels_per_pass = 8 / m.p_features;
		p.dense_mask = 0;
		p.hash_mask = (1u << m.p_log2_T) - 1;
	}
	if (!upload(&p.d_chunks, img.data(), img.size()) || !upload(&p.d_out, oimg.data(), oimg.size()) ||
	    !upload(&p.d_grid, m.grid.data(), m.grid.size() * 2) ||
	    (m.fp8 && (!upload(&p.d_cs, cs.data(), cs.size() * 4) || !upload(&p.d_inv_a, inv_a.data(), inv_a.size() * 4)))) {
		*err = "nrc_infer: upload failed";
		destroy(p);
		return false;
	}
	return true;
}

void destroy(Plan& p) {
	if (p.sort_ws) {
		SortWs& w = *p.sort_ws;
		cudaFree(w.k0); cudaFree(w.k1); cudaFree(w.i0); cudaFree(w.i1); cudaFree(w.in); cudaFree(w.out); cudaFree(w.tmp);
		delete p.sort_ws; p.sort_ws = nullptr;
	}
	cudaFree(p.d_chunks); cudaFree(p.d_out); cudaFree(p.d_cs); cudaFree(p.d_inv_a); cudaFree(p.d_grid);
	p.d_chunks = p.d_out = nullptr; p.d_cs = p.d_inv_a = nullptr; p.d_grid = nullptr;
}

namespace {
template <int D>
void launch_permuto(int F, dim3 grid, cudaStream_t stream, const PermutoArgs& a, uint4* enc) {
	if (F == 2) permuto_kernel<D, 2><<<grid, 256, 0, stream>>>(a, enc);
	else if (F == 4) permuto_kernel<D, 4><<<grid, 256, 0, stream>>>(a, enc);
	else permuto_kernel<D, 8><<<grid, 256, 0, stream>>>(a, enc);
}
cudaError_t permuto_pass(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, __half2* enc) {
	if (!enc || ((uintptr_t)enc & 15) || ((uintptr_t)in & 15)) return cudaErrorInvalidValue;
	PermutoArgs a;
	a.in = in; a.table = p.d_grid; a.n = n; a.hmask = (1u << p.p_log2_T) - 1; a.log2_T = (uint32_t)p.p_log2_T;
	a.groups = (uint32_t)(p.p_levels * p.p_features / 8);
	a.gps = p.levels_per_pass > 0 ? std::max(1u, (uint32_t)(p.levels_per_pass * p.p_features / 8)) : a.groups;
	for (int i = 0; i < p.p_levels * p.p_d; ++i) { a.scale[i] = p.p_scale[i]; a.shift[i] = p.p_shift[i]; }
	const dim3 grid((n + 255) / 256, (a.groups + a.gps - 1) / a.gps);
	uint4* e4 = reinterpret_cast<uint4*>(enc);
	if (p.p_d == 3) launch_permuto<3>(p.p_features, grid, stream, a, e4);
	else if (p.p_d == 5) launch_permuto<5>(p.p_features, grid, stream, a, e4);
	else if (p.p_d == 8) launch_permuto<8>(p.p_features, grid, stream, a, e4);
	else return cudaErrorInvalidValue;
	return cudaGetLastError();
}
}  // namespace

cudaError_t grid_pass(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, __half2* enc) {
	if (n == 0) return cudaSuccess;
	if (p.encoding == ENC_PERMUTO) return permuto_pass(p, stream, n, in, enc);
	GridArgs g;
	g.in = in; g.grid = reinterpret_cast<const __half2*>(p.d_grid); g.n = n;
	g.hash_mask = p.hash_mask; g.dense_mask = p.dense_mask;
	for (int l = 0; l <= N_LEVELS; ++l) g.offsets[l] = p.offsets[l];
	for (int l = 0; l < N_LEVELS; ++l) { g.res[l] = p.res[l]; g.scale[l] = p.scale[l]; }
	const uint32_t lp = (uint32_t)p.levels_per_pass;
	grid_kernel<<<dim3((n + 255) / 256, (N_LEVELS + lp - 1) / lp), 256, 0, stream>>>(g, lp, enc);
	return cudaGetLastError();
}

cudaError_t mlp_pass(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, const __half2* enc) {
	if (n == 0) return cudaSuccess;
	if (((uintptr_t)in) & 15) return cudaErrorMisalignedAddress;
	KArgs a;
	a.in = in; a.out = out; a.enc = enc; a.chunks = p.d_chunks; a.outw = p.d_out; a.cs = p.d_cs; a.inv_a = p.d_inv_a;
	a.n = n; a.n_tiles = (n + TILE - 1) / TILE; a.depth = p.depth; a.n_chunks = p.chunks; a.stages = p.stages;
	a.out_bytes = (p.fp8 ? p.width / 128 : p.width / 64) * 2048; a.out_exp = p.out_act == OUT_EXP;
	if (p.encoding == ENC_PERMUTO && (!enc || ((uintptr_t)enc & 15))) return cudaErrorInvalidValue;
	a.grid = reinterpret_cast<const __half2*>(p.d_grid); a.hash_mask = p.hash_mask; a.dense_mask = p.dense_mask;
	for (int l = 0; l <= N_LEVELS; ++l) a.offsets[l] = p.offsets[l];
	for (int l = 0; l < N_LEVELS; ++l) { a.res[l] = p.res[l]; a.scale[l] = p.scale[l]; }
	const uint32_t want = (a.n_tiles + p.nc - 1) / p.nc;
	uint32_t grid = std::min<uint32_t>(want, (uint32_t)p.grid);
	void* argv[] = {&a};
	if (p.cluster <= 1) return cudaLaunchKernel(p.kernel, grid, p.nc * 128 + 128, argv, p.smem, stream);
	grid = std::max<uint32_t>(p.cluster, grid / p.cluster * p.cluster);
	cudaLaunchConfig_t cfg = {};
	cfg.gridDim = dim3(grid); cfg.blockDim = dim3(p.nc * 128 + 128); cfg.dynamicSmemBytes = p.smem; cfg.stream = stream;
	cudaLaunchAttribute attr[1];
	attr[0].id = cudaLaunchAttributeClusterDimension;
	attr[0].val.clusterDim.x = p.cluster; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
	cfg.attrs = attr; cfg.numAttrs = 1;
	return cudaLaunchKernelExC(&cfg, p.kernel, argv);
}

namespace {
__device__ __forceinline__ uint32_t spread3(uint32_t v) {  // 10 bits -> every third bit of 30
	v &= 0x3FF;
	v = (v | (v << 16)) & 0x030000FF;
	v = (v | (v << 8)) & 0x0300F00F;
	v = (v | (v << 4)) & 0x030C30C3;
	v = (v | (v << 2)) & 0x09249249;
	return v;
}
__global__ void morton_keys(uint32_t n, uint32_t bits_per_axis, const __half* __restrict__ in, uint32_t* key, uint32_t* idx) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const uint2 v = __ldg(reinterpret_cast<const uint2*>(in + (size_t)i * 16));
	const float x[3] = {__half2float(__ushort_as_half((unsigned short)(v.x & 0xFFFF))),
	                    __half2float(__ushort_as_half((unsigned short)(v.x >> 16))),
	                    __half2float(__ushort_as_half((unsigned short)(v.y & 0xFFFF)))};
	const float r = (float)(1u << bits_per_axis);
	uint32_t k = 0;
#pragma unroll
	for (int d = 0; d < 3; ++d) {
		const uint32_t q = (uint32_t)fminf(fmaxf(x[d] * r, 0.0f), r - 1.0f);
		k |= spread3(q << (10 - bits_per_axis)) << d;
	}
	key[i] = k >> (30 - 3 * bits_per_axis);
	idx[i] = i;
}
__global__ void permute_rows(uint32_t n, const uint32_t* __restrict__ idx, const __half* __restrict__ in, __half* __restrict__ dst) {
	const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n) return;
	const uint4* s = reinterpret_cast<const uint4*>(in + (size_t)__ldg(idx + j) * 16);
	uint4* d = reinterpret_cast<uint4*>(dst + (size_t)j * 16);
	d[0] = __ldg(s); d[1] = __ldg(s + 1);
}
__global__ void unpermute_out(uint32_t n, const uint32_t* __restrict__ idx, const float* __restrict__ src, float* __restrict__ out) {
	const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n) return;
	const size_t o = (size_t)__ldg(idx + j) * 3;
	out[o] = src[(size_t)j * 3]; out[o + 1] = src[(size_t)j * 3 + 1]; out[o + 2] = src[(size_t)j * 3 + 2];
}
}  // namespace

static cudaError_t infer_unsorted(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, __half2* enc) {
	if (p.encoding != ENC_PERMUTO && p.levels_per_pass == 0) return mlp_pass(p, stream, n, in, out, nullptr);
	cudaError_t e = grid_pass(p, stream, n, in, enc);
	if (e != cudaSuccess) return e;
	return mlp_pass(p, stream, n, in, out, enc);
}

static cudaError_t sort_reserve(SortWs& w, uint32_t n, int bits) {
	if (n <= w.cap) return cudaSuccess;
	cudaFree(w.k0); cudaFree(w.k1); cudaFree(w.i0); cudaFree(w.i1); cudaFree(w.in); cudaFree(w.out); cudaFree(w.tmp);
	w = SortWs{};
	const uint32_t c = n + n / 4;
	cub::DoubleBuffer<uint32_t> k(nullptr, nullptr), v(nullptr, nullptr);
	cudaError_t e = cub::DeviceRadixSort::SortPairs(nullptr, w.tmp_bytes, k, v, (int)c, 0, bits);
	if (e == cudaSuccess) e = cudaMalloc(&w.k0, c * 4ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.k1, c * 4ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.i0, c * 4ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.i1, c * 4ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.in, c * 32ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.out, c * 12ull);
	if (e == cudaSuccess) e = cudaMalloc(&w.tmp, w.tmp_bytes);
	if (e == cudaSuccess) w.cap = c;
	return e;
}

cudaError_t infer(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, __half2* enc) {
	if (p.sort_bits <= 0 || n < 2) return infer_unsorted(p, stream, n, in, out, enc);
	if (!p.sort_ws) return cudaErrorInvalidValue;
	SortWs& w = *p.sort_ws;
	const int bpa = std::min(10, std::max(1, p.sort_bits / 3)), bits = 3 * bpa;
	cudaError_t e = sort_reserve(w, n, bits);
	if (e != cudaSuccess) return e;
	const uint32_t b = (n + 255) / 256;
	morton_keys<<<b, 256, 0, stream>>>(n, (uint32_t)bpa, in, w.k0, w.i0);
	cub::DoubleBuffer<uint32_t> k(w.k0, w.k1), v(w.i0, w.i1);
	e = cub::DeviceRadixSort::SortPairs(w.tmp, w.tmp_bytes, k, v, (int)n, 0, bits, stream);
	if (e != cudaSuccess) return e;
	permute_rows<<<b, 256, 0, stream>>>(n, v.Current(), in, w.in);
	e = infer_unsorted(p, stream, n, w.in, w.out, enc);
	if (e != cudaSuccess) return e;
	unpermute_out<<<b, 256, 0, stream>>>(n, v.Current(), w.out, out);
	return cudaGetLastError();
}

}  // namespace nrcw
