// nrc_infer.h: frozen-weight NRC inference for wide, deep MLPs on H100 (sm_90a wgmma). See analysis/nrc_infer.md.
//
// Model: the engine's encoding (3D hash grid 16 levels x 2 features, one-blob on 5 dims, identity on 6 dims, padding
// ones to 64), then an MLP without biases: W x 64, (D-1) x [W x W], 16 x W (rows 0..2 used); hidden activation ReLU or
// SiLU; output activation linear (clamped at 0) or exp. W = 64, 128 or 256; D = 1..16.
// Precision: fp16 (weights and activations fp16, fp32 accumulation) or fp8 (layer 0 fp16; the D-1 hidden layers and
// the output layer use e4m3 weights with per-output-channel scales and e4m3 activations with per-layer scales,
// wgmma k32 at twice the fp16 rate; W >= 128).
// Weights come from an NRCINF1 file (format in nrc/bake/stream/bake_stream.py write_nrcinf).
// Encoding "permuto" (header "encoding": "permuto", written by nrc/bake/stream/bake_stream.py export_permuto_fp16): a
// permutohedral lattice grid (nrc/bake/stream/permuto_triton.py, tcnn's PermutoEncoding) on x[0:d] (d = 3, 5 or 8), L levels x F
// features (F = 2, 4 or 8, L x F a multiple of 8), then one-blob (4 bins) on x[d:8], identity on x[8:14], ones to 64.
// permuto_pass writes the whole 64-wide fp16 MLP input (8 x 16 bytes per query) and the MLP kernel copies it.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace nrcw {

enum Act { ACT_RELU = 0, ACT_SILU = 1 };
// SiLU implementation: x * sigmoid(x) with ex2.approx + rcp.approx in fp32 (2 MUFU ops per element), or in f16x2
// with ex2.approx.f16x2 and a Newton reciprocal (1 MUFU op per 2 elements; nrc_infer.cu silu_h2).
enum SiluImpl { SILU_EX2 = 1, SILU_H2 = 2 };
enum OutAct { OUT_LINEAR = 0, OUT_EXP = 1 };
enum Encoding { ENC_HASH = 0, ENC_PERMUTO = 1 };
constexpr int PERMUTO_MAX_LEVELS = 16, PERMUTO_MAX_D = 8;

struct HostModel {
	int width = 64, depth = 5;
	Act act = ACT_RELU;
	OutAct out_act = OUT_LINEAR;
	bool fp8 = false;
	uint32_t log2_hashmap = 19, base_resolution = 16;
	float per_level_scale = 1.5f;
	uint32_t n_entries = 0;
	float level_scale[16];         // per level: scale, resolution, entries, offset, dense (file "levels" or tcnn's)
	uint32_t level_res[16], level_size[16], level_off[17];
	bool level_dense[16];
	Encoding encoding = ENC_HASH;
	// permuto: input dims d, levels, features per level, log2 entries per level; per level l and dim j the lattice
	// coordinate is (x_j + p_shift[l d + j]) * p_scale[l d + j] (fp32, as trained); n_entries = levels << p_log2_T
	int p_d = 0, p_levels = 0, p_features = 0, p_log2_T = 0;
	std::vector<float> p_scale, p_shift;
	std::vector<uint16_t> grid;   // n_entries x 2 fp16 (permuto: n_entries x p_features)
	std::vector<uint16_t> w0;     // W x 64 fp16
	std::vector<uint8_t> hidden;  // (D-1) x W x W: fp16 (2 bytes) or e4m3 (1 byte)
	std::vector<uint8_t> out;     // 16 x W: fp16 or e4m3
	std::vector<float> w_scale;   // fp8: (D-1) x W + 16
	std::vector<float> a_scale;   // fp8: D
	std::string header;           // the file's header JSON
};

// Reads an NRCINF1 file. Returns false and sets err on failure.
bool load_nrcinf(const char* path, HostModel& m, std::string* err);

struct SortWs;
struct Plan {
	int width = 0, depth = 0, nc = 2, cluster = 1, stages = 0, chunks = 0, grid = 0, regs = 0;
	bool fp8 = false, resident = false, rs = false;
	Act act = ACT_RELU;
	SiluImpl silu = SILU_EX2;
	OutAct out_act = OUT_LINEAR;
	size_t smem = 0;
	const void* kernel = nullptr;
	// device buffers
	uint8_t* d_chunks = nullptr;   // C chunk images (W x 128 B each, 128-byte swizzled K-major), in consumption order
	uint8_t* d_out = nullptr;      // output-layer image (W/64 or W/128 chunks of 16 x 128 B)
	float* d_cs = nullptr;         // fp8: per-column epilogue scale of layers 1..D ((D-1) x W + 16)
	float* d_inv_a = nullptr;      // fp8: 1 / a_scale of the D fp8 layers
	__half* d_grid = nullptr;      // hash-grid table
	uint32_t offsets[17], res[16], dense_mask = 0;
	float scale[16];
	uint32_t hash_mask = 0;
	// permuto (encoding == ENC_PERMUTO): always the split grid pass, levels_per_pass levels per sweep (0 = all; create: 8 / F,
	// one 16-byte group);
	// fp16 only, cluster 1
	Encoding encoding = ENC_HASH;
	int p_d = 0, p_levels = 0, p_features = 0, p_log2_T = 0;
	float p_scale[PERMUTO_MAX_LEVELS * PERMUTO_MAX_D], p_shift[PERMUTO_MAX_LEVELS * PERMUTO_MAX_D];
	int levels_per_pass = 4;       // grid pass: levels per sweep (L2 working set); 0 = the MLP kernel gathers itself
	                               // (create sets 0 for W <= 128, 4 for W = 256)
	// Morton sort: infer first sorts the queries by the Morton code of their position (sort_bits bits, 3 per axis;
	// 0 = off), runs on the sorted copy, and scatters the outputs back. Neighbouring queries then share hash-grid
	// sectors in L2. Outputs are bit-identical to the unsorted run. Workspace grows on demand (sort_ws).
	int sort_bits = 0;
	SortWs* sort_ws = nullptr;
};

// nc: consumer warpgroups per CTA (2 or 3; 0 = auto). Uploads the weights. Needs an sm_90 device.
// cluster: 1, or 2 = two CTAs share the weight stream through TMA multicast.
// rs: activations stay in registers as the A operand of the next layer (RS wgmma; fp16, width 128 or 256, cluster 1).
bool create(Plan& p, const HostModel& m, int nc, std::string* err, SiluImpl silu = SILU_EX2, int cluster = 1, bool rs = false);
void destroy(Plan& p);

// out[i*3 + c] = MLP(encode(in[i*16 .. i*16+13]))[c]. enc: workspace of enc_half2(p) * n half2 (grid features; unused
// and may be null when enc_half2(p) == 0).
inline uint32_t enc_half2(const Plan& p) { return p.encoding == ENC_PERMUTO ? 32u : p.levels_per_pass ? 16u : 0u; }
cudaError_t infer(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, __half2* enc);
// The two passes separately (timing): the grid pass, then the MLP kernel that reads enc. enc needs enc_half2 * n half2
// (permuto: 32; split hash grid: 16).
cudaError_t grid_pass(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, __half2* enc);
cudaError_t mlp_pass(const Plan& p, cudaStream_t stream, uint32_t n, const __half* in, float* out, const __half2* enc);

}  // namespace nrcw
