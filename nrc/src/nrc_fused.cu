// nrc_fused.cu: fused NRC inference kernel (see nrc_fused.h and analysis/nrc_wgmma.md).
//
// One persistent kernel. A CTA holds all weights in shared memory (loaded once, 128-byte swizzled, K-major) and runs
// WG independent warpgroups. A warpgroup loops over 64-query tiles:
//   1. encode: 128 threads write the 64 x 64 fp16 encoding (hash grid with the gathers, one-blob, identity, ones) into
//      its own swizzled tile; thread t takes query t%64 and grid levels 8*(t/64)..+7, so a warp's load instruction
//      reads 32 queries at one level and corner;
//   2. layer 0: A (encoding) and B (weights) from shared memory;
//   3. the epilogue (ReLU, fp32 -> f16x2) turns the accumulator into the A registers of the next layer (as P in
//      FlashAttention); layers 1..H-1 and the n16 output layer take A from registers;
//   4. channels 0..2 of the output layer go to global memory as fp32, clamped at 0.
// CORE_WGMMA (sm_90a): wgmma.m64n64k16 SS for layer 0, RS for the rest (m64n16k16 last). CORE_MMA (sm_80+):
// mma.sync.m16n8k16, warp w takes rows 16w..16w+15 of the tile. Both cores use the same fragment convention (the
// wgmma m64nN accumulator and RS A fragments equal the mma.sync ones of warp w), so encoder, epilogue and store are
// shared. Warpgroups overlap each other (one gathers while another multiplies) through the warp scheduler; there is
// no producer/consumer hand-off.
// Split mode (Plan::levels_per_pass > 0, the default in the engine): grid_pass gathers the hash-grid levels first,
// a few levels per sweep over all queries, so the table working set fits in L2 (all 16 levels at once overflowed the
// A40's L2); step 1 then reads 16 x half2 per query instead of gathering. See analysis/nrc_wgmma.md.
#include "nrc_fused.h"

#include <cute/arch/mma_sm90_gmma.hpp>

#include <cmath>
#include <cstdio>

namespace nrc_fused {
namespace {

constexpr int TILE = 64;           // queries per tile (wgmma M)
constexpr int TILE_HALVES = TILE * 64;

struct Args {
	const __half* in;
	float* out;
	const __half* params;
	uint32_t n, n_tiles, hidden, n_net;
	float log2_scale;
	float base_resolution;
	uint32_t hash_mask;
	const __half2* enc;  // split mode: grid features from grid_pass (16 x n half2, level-major); nullptr = gather here
	uint32_t offsets[N_LEVELS + 1];
};

// 128-byte swizzle of a K-major tile with 64 fp16 per row (CuTe GMMA::Layout_K_SW128_Atom<half_t>, Swizzle<3,3,3>):
// 16-byte chunk c of row r goes to chunk c ^ (r % 8). Tiles start at 1024-byte boundaries.
__host__ __device__ inline uint32_t swz(uint32_t r, uint32_t k) { return r * 64 + ((((k >> 3) ^ (r & 7)) << 3) | (k & 7)); }

__device__ inline float ex2_approx(float x) {  // tcnn builds with --use_fast_math: exp2f is ex2.approx.ftz
	float y;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__device__ inline float quartic_cdf(float x, float inv_radius) {  // tcnn common_device.h
	const float u = x * inv_radius, u2 = u * u, u4 = u2 * u2;
	return fmaxf(0.0f, fminf(1.0f, ((float)15 / 16) * u * (1 - ((float)2 / 3) * u2 + ((float)1 / 5) * u4) + 0.5f));
}
__device__ inline float blob_cdf(float b, float x) {  // periodic: tcnn kernel_one_blob_soa
	return quartic_cdf(b - x, 4.0f) + quartic_cdf(b - x - 1.0f, 4.0f) + quartic_cdf(b - x + 1.0f, 4.0f);
}

__device__ inline uint32_t h2u(__half2 v) { return *reinterpret_cast<uint32_t*>(&v); }
__device__ inline uint32_t pack_relu(float x, float y) { return h2u(__floats2half2_rn(fmaxf(x, 0.0f), fmaxf(y, 0.0f))); }
__device__ inline void st_h2(__half* tile, uint32_t r, uint32_t k, __half2 v) { *reinterpret_cast<__half2*>(tile + swz(r, k)) = v; }

__device__ inline void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" ::"r"(wg + 1) : "memory"); }
__device__ inline void fence_async_smem() {  // generic-proxy smem writes -> visible to wgmma (async proxy)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

// ---- encoding ----
// One hash-grid level at position (x0, x1, x2) in [0,1]^3: tcnn's kernel_grid (linear interpolation, fp16 fma).
__device__ __forceinline__ __half2 grid_level(float x0, float x1, float x2, uint32_t l, const uint32_t* offsets,
                                              const __half2* __restrict__ grid, float log2_scale, float base_resolution,
                                              uint32_t hash_mask) {
	const float scale = ex2_approx((float)l * log2_scale) * base_resolution - 1.0f;  // tcnn grid_scale
	const uint32_t res = (uint32_t)ceilf(scale) + 1;                                    // tcnn grid_resolution
	const uint32_t base = offsets[l], hsize = offsets[l + 1] - base;
	const bool hashed = res > 0x659 || hsize < res * res * res;  // tcnn grid_index (3D)
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
			idx = (q0 ^ q1 * 2654435761u ^ q2 * 805459861u) & hash_mask;  // hsize == 2^log2 when hashed
		} else {
			idx = q0 + q1 * res + q2 * res * res;  // < 2 * hsize (a corner may be res at x = 1)
			if (idx >= hsize) idx -= hsize;
		}
		val[c] = __ldg(grid + base + idx);
	}
	__half2 acc = __float2half2_rn(0.0f);
#pragma unroll
	for (int c = 0; c < 8; ++c) {  // same order and fp16 fma as tcnn's kernel_grid
		float w = 1.0f;
#pragma unroll
		for (int d = 0; d < 3; ++d) w *= (c >> d) & 1 ? pos[d] : 1 - pos[d];
		acc = __hfma2(__half2half2(__float2half(w)), val[c], acc);
	}
	return acc;
}

// One tile: grid levels (gathered here, or read from the grid pass), one-blob, identity, ones.
// Scalars, not the Args reference: taking the address of a kernel parameter copies it to local memory.
__device__ __forceinline__ void encode_tile(const __half* __restrict__ in, uint32_t n, const __half2* __restrict__ grid,
                                            float log2_scale, float base_resolution, uint32_t hash_mask,
                                            const uint32_t* offsets, const __half2* __restrict__ enc, uint32_t tile,
                                            uint32_t t, __half* sA) {
	const uint32_t q = t & 63, part = t >> 6;
	const uint32_t row = tile * TILE + q;
	float x[14];
	if (row < n) {
		const uint4* src = reinterpret_cast<const uint4*>(in + (size_t)row * 16);
		const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
		const uint32_t w[7] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z};
#pragma unroll
		for (int i = 0; i < 7; ++i) {  // bit ops, not a pointer cast: x stays in registers
			x[2 * i] = __half2float(__ushort_as_half((unsigned short)(w[i] & 0xFFFF)));
			x[2 * i + 1] = __half2float(__ushort_as_half((unsigned short)(w[i] >> 16)));
		}
	} else {
#pragma unroll
		for (int i = 0; i < 14; ++i) x[i] = 0.0f;
	}

	if (enc) {  // split mode: the grid pass wrote level l of query row at enc[l * n + row]
		__half2 v[8];
#pragma unroll
		for (uint32_t li = 0; li < 8; ++li) v[li] = row < n ? __ldg(enc + (size_t)(part * 8 + li) * n + row) : __float2half2_rn(0.0f);
#pragma unroll
		for (uint32_t li = 0; li < 8; ++li) st_h2(sA, q, 2 * (part * 8 + li), v[li]);
	} else {
#pragma unroll 2
		for (uint32_t li = 0; li < 8; ++li) {
			const uint32_t l = part * 8 + li;
			st_h2(sA, q, 2 * l, grid_level(x[0], x[1], x[2], l, offsets, grid, log2_scale, base_resolution, hash_mask));
		}
	}

	// One-blob (features 3..7 -> columns 32..51), identity (8..13 -> 52..57), padding ones (58..63).
#pragma unroll
	for (uint32_t d = 0; d < 5; ++d) {  // part 0: dims 0..2, part 1: dims 3..4 (static indices keep x in registers)
		if ((d >= 3) != (part != 0)) continue;
		const float xv = x[3 + d];
		float cdf[5];
#pragma unroll
		for (int b = 0; b < 5; ++b) cdf[b] = blob_cdf(b * 0.25f, xv);
		st_h2(sA, q, 32 + 4 * d, __floats2half2_rn(cdf[1] - cdf[0], cdf[2] - cdf[1]));
		st_h2(sA, q, 34 + 4 * d, __floats2half2_rn(cdf[3] - cdf[2], cdf[4] - cdf[3]));
	}
	if (part) {
#pragma unroll
		for (int i = 0; i < 3; ++i) st_h2(sA, q, 52 + 2 * i, __floats2half2_rn(x[8 + 2 * i], x[9 + 2 * i]));
#pragma unroll
		for (int i = 0; i < 3; ++i) st_h2(sA, q, 58 + 2 * i, __float2half2_rn(1.0f));
	}
}

// ---- mma.sync core (sm_80+) ----
__device__ inline void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, bool zero) {
	const float z = 0.0f;
	asm volatile(
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
		: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(zero ? z : d[0]), "f"(zero ? z : d[1]),
		  "f"(zero ? z : d[2]), "f"(zero ? z : d[3]));
}

#ifdef NRC_MMA_F16ACC
// fp16-accumulate variant (RTX PRO 6000: GeForce-class Blackwell tensor cores may run fp16 accumulation at a higher
// rate than fp32). d[0] = row g, d[1] = row g+8, each a half2 of cols 2*tq + {0,1}: the f32 fragment, packed.
__device__ inline void mma16816h(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1, bool zero) {
	const uint32_t z = 0u;
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
	             : "=r"(d[0]), "=r"(d[1])
	             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(zero ? z : d[0]), "r"(zero ? z : d[1]));
}
template <int NB>
__device__ inline void mma_layer_h(const uint32_t (&a)[4][4], const __half* W, uint32_t g, uint32_t tq, uint32_t* d) {
	const __half* base = W + g * 64 + 2 * tq;
#pragma unroll
	for (int kb = 0; kb < 4; ++kb) {
		const uint32_t c0 = ((2 * kb) ^ g) << 3, c1 = ((2 * kb + 1) ^ g) << 3;
#pragma unroll
		for (int j = 0; j < NB; ++j)
			mma16816h(d + 2 * j, a[kb], *reinterpret_cast<const uint32_t*>(base + j * 512 + c0),
			          *reinterpret_cast<const uint32_t*>(base + j * 512 + c1), kb == 0);
	}
}
__device__ inline uint32_t relu_h2(uint32_t x) {
	__half2 h = *reinterpret_cast<__half2*>(&x);
	h = __hmax2(h, __float2half2_rn(0.0f));
	return *reinterpret_cast<uint32_t*>(&h);
}
__device__ inline float2 u2f2(uint32_t x) { return __half22float2(*reinterpret_cast<__half2*>(&x)); }
#endif

// d[4j..4j+3] = rows (g, g+8) x cols 8j + 2*tq + {0,1} of A(16 x 64) * W^T, W = NB*8 rows of swizzled K-major weights.
// Row 8j+g, k = 16kb + 2tq (+8) lies in 16-byte chunk 2kb (+1), stored at chunk (2kb (+1)) ^ g.
template <int NB>
__device__ inline void mma_layer(const uint32_t (&a)[4][4], const __half* W, uint32_t g, uint32_t tq, float* d) {
	const __half* base = W + g * 64 + 2 * tq;
#pragma unroll
	for (int kb = 0; kb < 4; ++kb) {
		const uint32_t c0 = ((2 * kb) ^ g) << 3, c1 = ((2 * kb + 1) ^ g) << 3;
#pragma unroll
		for (int j = 0; j < NB; ++j)
			mma16816(d + 4 * j, a[kb], *reinterpret_cast<const uint32_t*>(base + j * 512 + c0),
			         *reinterpret_cast<const uint32_t*>(base + j * 512 + c1), kb == 0);
	}
}

// ---- wgmma core (sm_90a) ----
// Shared-memory matrix descriptor, K-major, 128-byte swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused (1),
// base offset 0 (1024-byte aligned tiles). The k16 slice kb starts 32 B further (+2 in 16-byte units).
__device__ inline uint64_t desc_sw128(const void* smem) {
	const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem);
	return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

#define NRC_D32(d) d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], \
	d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]

template <int N>
__device__ inline void fence_regs(float* d) {
#pragma unroll
	for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
__device__ inline void fence_regs(uint32_t (&a)[4][4]) {
#pragma unroll
	for (int i = 0; i < 16; ++i) asm volatile("" : "+r"(a[i / 4][i % 4])::"memory");
}

using GK = cute::GMMA::Major;
using SS64 = cute::SM90_64x64x16_F32F16F16_SS<GK::K, GK::K>;
using RS64 = cute::SM90_64x64x16_F32F16F16_RS<GK::K, GK::K>;
using RS16 = cute::SM90_64x16x16_F32F16F16_RS<GK::K, GK::K>;

__device__ inline void wgmma_wait_all() {
	cute::warpgroup_commit_batch();
	cute::warpgroup_wait<0>();
}

__device__ inline cute::GMMA::ScaleOut scale_out(int kb) { return kb ? cute::GMMA::ScaleOut::One : cute::GMMA::ScaleOut::Zero; }

__device__ inline void wgmma_ss64(const __half* A, const __half* W, float* d) {
	const uint64_t da = desc_sw128(A), db = desc_sw128(W);
	fence_regs<32>(d);
	cute::warpgroup_arrive();
#pragma unroll
	for (int kb = 0; kb < 4; ++kb) SS64::fma(da + 2 * kb, db + 2 * kb, NRC_D32(d), scale_out(kb));
	wgmma_wait_all();
	fence_regs<32>(d);
}

__device__ inline void wgmma_rs64(uint32_t (&a)[4][4], const __half* W, float* d) {
	const uint64_t db = desc_sw128(W);
	fence_regs<32>(d);
	fence_regs(a);
	cute::warpgroup_arrive();
#pragma unroll
	for (int kb = 0; kb < 4; ++kb) RS64::fma(a[kb][0], a[kb][1], a[kb][2], a[kb][3], db + 2 * kb, NRC_D32(d), scale_out(kb));
	wgmma_wait_all();
	fence_regs<32>(d);
	fence_regs(a);
}

__device__ inline void wgmma_rs16(uint32_t (&a)[4][4], const __half* W, float* d) {
	const uint64_t db = desc_sw128(W);
	fence_regs<8>(d);
	fence_regs(a);
	cute::warpgroup_arrive();
#pragma unroll
	for (int kb = 0; kb < 4; ++kb)
		RS16::fma(a[kb][0], a[kb][1], a[kb][2], a[kb][3], db + 2 * kb, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], scale_out(kb));
	wgmma_wait_all();
	fence_regs<8>(d);
	fence_regs(a);
}

// ---- kernel ----
template <int WG, int CORE>
__global__ void __launch_bounds__(WG * 128, WG == 2 ? 3 : 1) fused_kernel(const Args a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	if constexpr (CORE == CORE_WGMMA) {
#if !defined(CUTE_ARCH_MMA_SM90A_ENABLED)
		__trap();  // the wgmma kernel exists only in the sm_90a cubin
		return;
#endif
	}
	extern __shared__ uint8_t smem_raw[];
	__half* sW = reinterpret_cast<__half*>(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
	const uint32_t w_rows = a.hidden * 64 + 16;
	__half* tiles = sW + w_rows * 64;  // w_rows * 128 B is a multiple of 1024 B
	uint32_t* offsets = reinterpret_cast<uint32_t*>(tiles + WG * TILE_HALVES);  // runtime-indexed: smem, not params
	if (threadIdx.x == 0) {  // static indices only (a per-thread index would copy Args to local memory)
#pragma unroll
		for (int i = 0; i <= N_LEVELS; ++i) offsets[i] = a.offsets[i];
	}

	{  // weights: row-major out x in (layer 0..H-1: 64 rows, last: 16 rows) -> one swizzled K-major matrix
		const uint4* src = reinterpret_cast<const uint4*>(a.params);
		for (uint32_t c = threadIdx.x; c < w_rows * 8; c += blockDim.x) {
			const uint32_t r = c >> 3, kc = c & 7;
			*reinterpret_cast<uint4*>(sW + r * 64 + ((kc ^ (r & 7)) << 3)) = __ldg(src + c);
		}
	}
	if constexpr (CORE == CORE_WGMMA) fence_async_smem();
	__syncthreads();

	const uint32_t wg = threadIdx.x >> 7, t = threadIdx.x & 127, warp = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3;
	__half* sA = tiles + wg * TILE_HALVES;
	const __half* Wout = sW + a.hidden * 64 * 64;
	const __half2* grid = reinterpret_cast<const __half2*>(a.params + a.n_net);

	for (uint32_t tile = blockIdx.x * WG + wg; tile < a.n_tiles; tile += gridDim.x * WG) {
		wg_bar(wg);  // the previous tile's layer 0 has finished reading sA
		encode_tile(a.in, a.n, grid, a.log2_scale, a.base_resolution, a.hash_mask, offsets, a.enc, tile, t, sA);
		if constexpr (CORE == CORE_WGMMA) fence_async_smem();
		wg_bar(wg);

#ifdef NRC_MMA_F16ACC
		if constexpr (CORE == CORE_MMA) {
			uint32_t A[4][4], dh[16];
			const __half* rows = sA + (warp * 16 + g) * 64 + 2 * tq;
#pragma unroll
			for (int kb = 0; kb < 4; ++kb) {
				const uint32_t c0 = ((2 * kb) ^ g) << 3, c1 = ((2 * kb + 1) ^ g) << 3;
				A[kb][0] = *reinterpret_cast<const uint32_t*>(rows + c0);
				A[kb][1] = *reinterpret_cast<const uint32_t*>(rows + 512 + c0);
				A[kb][2] = *reinterpret_cast<const uint32_t*>(rows + c1);
				A[kb][3] = *reinterpret_cast<const uint32_t*>(rows + 512 + c1);
			}
			mma_layer_h<8>(A, sW, g, tq, dh);
			for (uint32_t layer = 1; layer <= a.hidden; ++layer) {
#pragma unroll
				for (int kb = 0; kb < 4; ++kb) {
					A[kb][0] = relu_h2(dh[4 * kb + 0]);
					A[kb][1] = relu_h2(dh[4 * kb + 1]);
					A[kb][2] = relu_h2(dh[4 * kb + 2]);
					A[kb][3] = relu_h2(dh[4 * kb + 3]);
				}
				if (layer < a.hidden) mma_layer_h<8>(A, sW + layer * 64 * 64, g, tq, dh);
				else mma_layer_h<2>(A, Wout, g, tq, dh);
			}
			if (tq < 2) {
				const uint32_t r0 = tile * TILE + warp * 16 + g, r1 = r0 + 8;
				const float2 v0 = u2f2(dh[0]), v1 = u2f2(dh[1]);
				if (r0 < a.n) {
					a.out[(size_t)r0 * 3 + 2 * tq] = fmaxf(v0.x, 0.0f);
					if (tq == 0) a.out[(size_t)r0 * 3 + 1] = fmaxf(v0.y, 0.0f);
				}
				if (r1 < a.n) {
					a.out[(size_t)r1 * 3 + 2 * tq] = fmaxf(v1.x, 0.0f);
					if (tq == 0) a.out[(size_t)r1 * 3 + 1] = fmaxf(v1.y, 0.0f);
				}
			}
			continue;
		}
#endif
		float d[32];
		uint32_t A[4][4];
		if constexpr (CORE == CORE_WGMMA) {
			wgmma_ss64(sA, sW, d);
		} else {
			const __half* rows = sA + (warp * 16 + g) * 64 + 2 * tq;  // rows g, g+8 (+512) share the swizzle phase g
#pragma unroll
			for (int kb = 0; kb < 4; ++kb) {
				const uint32_t c0 = ((2 * kb) ^ g) << 3, c1 = ((2 * kb + 1) ^ g) << 3;
				A[kb][0] = *reinterpret_cast<const uint32_t*>(rows + c0);
				A[kb][1] = *reinterpret_cast<const uint32_t*>(rows + 512 + c0);
				A[kb][2] = *reinterpret_cast<const uint32_t*>(rows + c1);
				A[kb][3] = *reinterpret_cast<const uint32_t*>(rows + 512 + c1);
			}
			mma_layer<8>(A, sW, g, tq, d);
		}
		for (uint32_t layer = 1; layer <= a.hidden; ++layer) {
#pragma unroll
			for (int kb = 0; kb < 4; ++kb) {  // accumulator n-blocks 2kb, 2kb+1 = A fragment of k16 slice kb
				A[kb][0] = pack_relu(d[8 * kb + 0], d[8 * kb + 1]);
				A[kb][1] = pack_relu(d[8 * kb + 2], d[8 * kb + 3]);
				A[kb][2] = pack_relu(d[8 * kb + 4], d[8 * kb + 5]);
				A[kb][3] = pack_relu(d[8 * kb + 6], d[8 * kb + 7]);
			}
			const __half* W = sW + layer * 64 * 64;
			if (layer < a.hidden) {
				if constexpr (CORE == CORE_WGMMA) wgmma_rs64(A, W, d);
				else mma_layer<8>(A, W, g, tq, d);
			} else {
				if constexpr (CORE == CORE_WGMMA) wgmma_rs16(A, Wout, d);
				else mma_layer<2>(A, Wout, g, tq, d);
			}
		}
		// Output n-block 0: tq 0 holds channels 0, 1; tq 1 holds channel 2 (and the padding column 3).
		if (tq < 2) {
			const uint32_t r0 = tile * TILE + warp * 16 + g, r1 = r0 + 8;
			if (r0 < a.n) {
				a.out[(size_t)r0 * 3 + 2 * tq] = fmaxf(d[0], 0.0f);
				if (tq == 0) a.out[(size_t)r0 * 3 + 1] = fmaxf(d[1], 0.0f);
			}
			if (r1 < a.n) {
				a.out[(size_t)r1 * 3 + 2 * tq] = fmaxf(d[2], 0.0f);
				if (tq == 0) a.out[(size_t)r1 * 3 + 1] = fmaxf(d[3], 0.0f);
			}
		}
	}
#else
	__trap();
#endif
}

// Split mode, pass 1: grid levels [y*L, y*L + L) for 256 queries per block, written level-major (coalesced). Blocks
// run in launch order (x fastest), so the GPU sweeps the levels in groups of L and the table working set stays near
// L levels (L x 2 MB at 2^19 entries) instead of the whole table, as in tcnn's level-major grid kernel.
__global__ void __launch_bounds__(256) grid_pass(const Args a, uint32_t levels_per_pass, __half2* __restrict__ enc) {
	__shared__ uint32_t offsets[N_LEVELS + 1];
	if (threadIdx.x == 0) {
#pragma unroll
		for (int i = 0; i <= N_LEVELS; ++i) offsets[i] = a.offsets[i];
	}
	__syncthreads();
	const uint32_t row = blockIdx.x * 256 + threadIdx.x;
	if (row >= a.n) return;
	const uint2 v = __ldg(reinterpret_cast<const uint2*>(a.in + (size_t)row * 16));  // features 0..3
	const float x0 = __half2float(__ushort_as_half((unsigned short)(v.x & 0xFFFF)));
	const float x1 = __half2float(__ushort_as_half((unsigned short)(v.x >> 16)));
	const float x2 = __half2float(__ushort_as_half((unsigned short)(v.y & 0xFFFF)));
	const __half2* grid = reinterpret_cast<const __half2*>(a.params + a.n_net);
	const uint32_t l0 = blockIdx.y * levels_per_pass, l1 = min(l0 + levels_per_pass, (uint32_t)N_LEVELS);
#pragma unroll 4
	for (uint32_t l = l0; l < l1; ++l)
		enc[(size_t)l * a.n + row] = grid_level(x0, x1, x2, l, offsets, grid, a.log2_scale, a.base_resolution, a.hash_mask);
}

size_t smem_bytes(int hidden, int wg) { return 1024 + (size_t)(hidden * 64 + 16) * 128 + (size_t)wg * TILE_HALVES * 2 + 128; }

template <int WG, int CORE>
const void* kernel_ptr() { return (const void*)fused_kernel<WG, CORE>; }

const void* pick_kernel(Core core, int wg) {
	if (core == CORE_WGMMA) return wg == 4 ? kernel_ptr<4, CORE_WGMMA>() : kernel_ptr<2, CORE_WGMMA>();
	return wg == 4 ? kernel_ptr<4, CORE_MMA>() : kernel_ptr<2, CORE_MMA>();
}

bool configure(Plan& p, int wg, const char** err) {
	const void* k = pick_kernel(p.core, wg);
	const size_t smem = smem_bytes(p.model.hidden, wg);
	if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) {
		cudaGetLastError();
		*err = "fused kernel: cannot set the shared-memory size";
		return false;
	}
	int bps = 0;
	if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, wg * 128, smem) != cudaSuccess || bps == 0) {
		cudaGetLastError();
		*err = "fused kernel: zero occupancy";
		return false;
	}
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, k);
	int dev, sms;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	p.wg = wg;
	p.smem = smem;
	p.blocks_per_sm = bps;
	p.grid = sms * bps;
	p.regs = attr.numRegs;
	return true;
}

}  // namespace

uint32_t grid_offsets(const Model& m, uint32_t* offsets) {
	uint32_t off = 0;
	for (int l = 0; l < N_LEVELS; ++l) {  // tcnn GridEncodingTemplated constructor (3D, hash grid)
		const float scale = exp2f(l * std::log2(m.per_level_scale)) * m.base_resolution - 1.0f;
		const uint32_t res = (uint32_t)ceilf(scale) + 1;
		const uint32_t max_params = 0xFFFFFFFFu / 2;
		uint32_t in_level = std::pow((float)res, 3.0f) > (float)max_params ? max_params : res * res * res;
		in_level = (in_level + 7) / 8 * 8;
		in_level = std::min(in_level, 1u << m.log2_hashmap);
		offsets[l] = off;
		off += in_level;
	}
	offsets[N_LEVELS] = off;
	return off;
}

bool setup(Plan& p, const Model& m, Core core, int wg, const char** err) {
	if (m.hidden < 1 || m.hidden > MAX_HIDDEN) { *err = "fused kernel: n_hidden_layers must be 1..8"; return false; }
	if (m.log2_hashmap < 4 || m.log2_hashmap > 30) { *err = "fused kernel: log2_hashmap_size out of range"; return false; }
	int dev, major, minor;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
	cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
	if (core == CORE_WGMMA && !(major == 9 && minor == 0)) { *err = "fused kernel: wgmma needs sm_90 (H100)"; return false; }
	if (major < 8) { *err = "fused kernel: needs sm_80 or newer"; return false; }
	p.core = core;
	p.model = m;
	p.n_net = (uint32_t)(m.hidden * 64 * 64 + 16 * 64);
	grid_offsets(m, p.offsets);
	if (wg == 2 || wg == 4) return configure(p, wg, err);
	// Auto: more resident warpgroups per SM wins; ties go to 2 (smaller CTAs balance better).
	Plan p2 = p, p4 = p;
	const char* e2 = nullptr;
	const char* e4 = nullptr;
	const bool ok2 = configure(p2, 2, &e2), ok4 = configure(p4, 4, &e4);
	if (!ok2 && !ok4) { *err = e2; return false; }
	p = (ok4 && (!ok2 || p4.blocks_per_sm * 4 > p2.blocks_per_sm * 2)) ? p4 : p2;
	return true;
}

cudaError_t infer(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, const __half* params,
                  __half2* enc) {
	if (n == 0) return cudaSuccess;
	if (((uintptr_t)params | (uintptr_t)in) & 15) return cudaErrorMisalignedAddress;
	if (p.levels_per_pass && !enc) return cudaErrorInvalidValue;
	Args a;
	a.in = in;
	a.out = out;
	a.params = params;
	a.n = n;
	a.n_tiles = (n + TILE - 1) / TILE;
	a.hidden = (uint32_t)p.model.hidden;
	a.n_net = p.n_net;
	a.log2_scale = std::log2(p.model.per_level_scale);
	a.base_resolution = (float)p.model.base_resolution;
	a.hash_mask = (1u << p.model.log2_hashmap) - 1;
	for (int l = 0; l <= N_LEVELS; ++l) a.offsets[l] = p.offsets[l];
	a.enc = nullptr;
	if (p.levels_per_pass) {
		const uint32_t lp = (uint32_t)p.levels_per_pass;
		grid_pass<<<dim3((n + 255) / 256, (N_LEVELS + lp - 1) / lp), 256, 0, stream>>>(a, lp, enc);
		a.enc = enc;
	}
	const uint32_t want = (a.n_tiles + p.wg - 1) / p.wg;
	const int grid = (int)std::min<uint32_t>(want, (uint32_t)p.grid);
	void* argv[] = {&a};
	return cudaLaunchKernel(pick_kernel(p.core, p.wg), grid, p.wg * 128, argv, p.smem, stream);
}

}  // namespace nrc_fused
