// cycles_nrc.cu: NRC engine v1 on tiny-cuda-nn (see include/cycles_nrc.h for the contract).
#include "cycles_nrc.h"
#include "cycles_nrc_debug.h"
#include "nrc_fused.h"
#include "nrc_infer.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

namespace {

constexpr uint32_t N_USED = 14;  // features 0..13 of NRC_IN; 14..15 are padding
constexpr uint32_t N_OUT = 3;

json default_config() {
	return {
		{"encoding", {
			{"otype", "Composite"},
			{"nested", {
				{{"n_dims_to_encode", 3}, {"otype", "HashGrid"}, {"n_levels", 16}, {"n_features_per_level", 2},
				 {"log2_hashmap_size", 19}, {"base_resolution", 16}, {"per_level_scale", 1.5}},
				{{"n_dims_to_encode", 5}, {"otype", "OneBlob"}, {"n_bins", 4}},
				{{"otype", "Identity"}},
			}},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"}, {"activation", "ReLU"}, {"output_activation", "None"},
			{"n_neurons", 64}, {"n_hidden_layers", 5},
		}},
		{"optimizer", {
			{"otype", "EMA"}, {"decay", 0.99},
			{"nested", {{"otype", "Adam"}, {"learning_rate", 1e-2}, {"beta1", 0.9}, {"beta2", 0.99},
			            {"epsilon", 1e-15}, {"l2_reg", 1e-6}}},
		}},
		{"loss", {{"otype", "RelativeL2"}}},
		{"nrc", {{"warmup_infer_batch", 1 << 18}, {"warmup_train_batch", 1 << 14}, {"seed", 1337}}},
	};
}

// Record j of a batch reads source record (start + j * stride) mod n: inference uses (0, 1) and zero-fills j >= n;
// training step s uses (s, steps), which interleaves the steps and wraps the tail to a multiple of 256.
__global__ void gather_inputs(uint32_t m, uint32_t n, uint32_t start, uint32_t stride, bool zero_tail,
                              const __half* __restrict__ src, float* __restrict__ dst) {
	const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= m) return;
	float* out = dst + (size_t)j * N_USED;
	if (zero_tail && j >= n) {
		for (uint32_t d = 0; d < N_USED; ++d) out[d] = 0.0f;
		return;
	}
	const uint32_t r = (uint32_t)(((uint64_t)start + (uint64_t)j * stride) % n);
	const __half2* in = (const __half2*)(src + (size_t)r * NRC_IN);  // 32-byte rows
	for (uint32_t d = 0; d < N_USED / 2; ++d) {
		const float2 v = __half22float2(in[d]);
		out[2 * d] = v.x;
		out[2 * d + 1] = v.y;
	}
}

__global__ void gather_targets(uint32_t m, uint32_t n, uint32_t start, uint32_t stride,
                               const float* __restrict__ src, float* __restrict__ dst) {
	const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= m) return;
	const uint32_t r = (uint32_t)(((uint64_t)start + (uint64_t)j * stride) % n);
	for (uint32_t c = 0; c < N_OUT; ++c) dst[(size_t)j * N_OUT + c] = src[(size_t)r * N_OUT + c];
}

__global__ void scatter_outputs(uint32_t n, uint32_t out_stride, const precision_t* __restrict__ src,
                                float* __restrict__ dst) {
	const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n) return;
	for (uint32_t c = 0; c < N_OUT; ++c) dst[(size_t)j * N_OUT + c] = fmaxf((float)src[(size_t)j * out_stride + c], 0.0f);
}

uint32_t round_up(uint32_t x) { return (x + BATCH_SIZE_GRANULARITY - 1) / BATCH_SIZE_GRANULARITY * BATCH_SIZE_GRANULARITY; }

void cu_check(CUresult r, const char* what) {
	if (r != CUDA_SUCCESS) {
		const char* s = nullptr;
		cuGetErrorString(r, &s);
		throw std::runtime_error{std::string{what} + ": " + (s ? s : "unknown CUDA driver error")};
	}
}

}  // namespace

struct NrcEngine {
	CUcontext ctx = nullptr;
	cudaStream_t create_stream = nullptr;
	json config;
	std::shared_ptr<Loss<precision_t>> loss;
	std::shared_ptr<Optimizer<precision_t>> optimizer;
	std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;
	std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

	// Persistent staging buffers (cudaMalloc, grown on demand; warm-up sizes them at create time).
	GPUMemory<float> in_f;        // N_USED x cap, column-major (one record per column)
	GPUMemory<float> target_f;    // N_OUT x cap
	GPUMemory<precision_t> out_h; // padded_output_width x cap
	uint64_t n_infer = 0, n_train_steps = 0;

	// nrc-bake (analysis/nrc_bake.md): CYCLES_NRC_LOAD / _SAVE / _FREEZE / _DUMP, read once at nrc_create.
	bool frozen = false;          // nrc_train is a no-op (inference only)
	std::string save_path;        // nrc_destroy writes the cache state here
	FILE* dump = nullptr;         // nrc_train appends its records here (offline bake data)
	uint64_t n_dumped = 0;
	json loaded_meta;             // header of the loaded file
	float out_scale = 1.0f;       // header "output_scale" of the loaded file (per-rule offline calibration)

	// Inference engine: "tcnn" (tcnn's encoding + FullyFusedMLP) or the fused kernel (nrc_fused.cu) on the same EMA
	// weights: "wgmma" (sm_90a; on other GPUs it runs the mma.sync core) or "mma" (mma.sync core on any sm_80+).
	std::string engine = "tcnn";
	bool fused = false;
	nrc_fused::Plan plan;
	GPUMemory<__half2> enc;  // split-mode workspace, 16 x n (engine infer: enc_half2 x n)

	// "engine": "infer": frozen weights from an NRCINF1 file (nrc_infer.cu); no tcnn model, nrc_train is a no-op.
	bool frozen_infer = false;
	nrcw::Plan iplan;
	~NrcEngine() { if (frozen_infer) nrcw::destroy(iplan); }

	void ensure(uint32_t m) {
		if (in_f.size() < (size_t)m * N_USED) in_f.resize((size_t)m * N_USED);
		if (target_f.size() < (size_t)m * N_OUT) target_f.resize((size_t)m * N_OUT);
		const size_t w = network->padded_output_width();
		if (out_h.size() < (size_t)m * w) out_h.resize((size_t)m * w);
	}

	void infer_tcnn(cudaStream_t stream, uint32_t n, const __half* in, float* rgb) {
		const uint32_t m = round_up(n);
		ensure(m);
		gather_inputs<<<(m + 255) / 256, 256, 0, stream>>>(m, n, 0, 1, true, in, in_f.data());
		GPUMatrix<float> x{in_f.data(), N_USED, m};
		GPUMatrix<precision_t> y{out_h.data(), network->padded_output_width(), m};
		network->inference_mixed_precision(stream, x, y, /*use_inference_params=*/true);
		scatter_outputs<<<(n + 255) / 256, 256, 0, stream>>>(n, network->padded_output_width(), out_h.data(), rgb);
	}

	void infer_fused(cudaStream_t stream, uint32_t n, const __half* in, float* rgb) {
		if (plan.levels_per_pass && enc.size() < (size_t)n * 16) enc.resize((size_t)n * 16);  // growth syncs
		CUDA_CHECK_THROW(nrc_fused::infer(plan, stream, n, in, rgb, (const __half*)trainer->params_inference(), enc.data()));
	}

	void infer(cudaStream_t stream, uint32_t n, const __half* in, float* rgb) {
		if (n == 0) return;
		if (frozen_infer) {
			const uint32_t ew = nrcw::enc_half2(iplan);  // 16 (split hash grid), 32 (permuto) or 0
			if (ew && enc.size() < (size_t)n * ew) enc.resize((size_t)n * ew);  // growth syncs; warm-up sizes it
			CUDA_CHECK_THROW(nrcw::infer(iplan, stream, n, in, rgb, ew ? enc.data() : nullptr));
		} else if (fused) infer_fused(stream, n, in, rgb);
		else infer_tcnn(stream, n, in, rgb);
		++n_infer;
	}

	void dump_records(cudaStream_t stream, uint32_t n, const __half* in, const float* target) {
		std::vector<__half> hin((size_t)n * NRC_IN);
		std::vector<float> ht((size_t)n * N_OUT);
		CUDA_CHECK_THROW(cudaMemcpyAsync(hin.data(), in, hin.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(ht.data(), target, ht.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		fwrite(&n, sizeof(uint32_t), 1, dump);
		fwrite(hin.data(), sizeof(__half), hin.size(), dump);
		fwrite(ht.data(), sizeof(float), ht.size(), dump);
		n_dumped += n;
	}

	void train(cudaStream_t stream, uint32_t n, const __half* in, const float* target, uint32_t steps) {
		if (n == 0 || frozen_infer) return;
		if (dump) dump_records(stream, n, in, target);
		if (frozen) return;
		if (steps == 0) steps = 1;
		if (steps > n) steps = n;
		const uint32_t m = round_up((n + steps - 1) / steps);
		ensure(m);
		GPUMatrix<float> x{in_f.data(), N_USED, m};
		GPUMatrix<float> t{target_f.data(), N_OUT, m};
		for (uint32_t s = 0; s < steps; ++s) {
			gather_inputs<<<(m + 255) / 256, 256, 0, stream>>>(m, n, s, steps, false, in, in_f.data());
			gather_targets<<<(m + 255) / 256, 256, 0, stream>>>(m, n, s, steps, target, target_f.data());
			trainer->training_step(stream, x, t);
			++n_train_steps;
		}
	}
};

namespace {

// Make the engine's context current for the duration of a call if the caller has another one current.
struct CtxGuard {
	bool pushed = false;
	explicit CtxGuard(CUcontext ctx) {
		CUcontext cur = nullptr;
		cu_check(cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
		if (cur != ctx) {
			cu_check(cuCtxPushCurrent(ctx), "cuCtxPushCurrent");
			pushed = true;
		}
	}
	~CtxGuard() {
		if (pushed) {
			CUcontext c;
			cuCtxPopCurrent(&c);
		}
	}
};

void report(const char* fn, const std::exception& e) { fprintf(stderr, "cycles_nrc: %s failed: %s\n", fn, e.what()); }

// The fused kernel implements one model shape; read it from the config and check it against tcnn's own sizes.
nrc_fused::Model fused_model(const NrcEngine& e) {
	const json& enc = e.config["encoding"];
	const json& net = e.config["network"];
	auto fail = [](const std::string& why) { throw std::runtime_error{"engine " + why}; };
	if (enc.value("otype", "") != "Composite" || !enc.contains("nested") || enc["nested"].size() != 3)
		fail("fused: encoding must be Composite [HashGrid, OneBlob, Identity]");
	const json& g = enc["nested"][0];
	const json& b = enc["nested"][1];
	const json& i = enc["nested"][2];
	const std::string go = g.value("otype", "");
	if (!(go == "HashGrid" || (go == "Grid" && g.value("type", "Hash") == "Hash")) || g.value("n_dims_to_encode", 0u) != 3 ||
	    g.value("n_levels", 16u) != 16 || g.value("n_features_per_level", 2u) != 2 ||
	    g.value("interpolation", "Linear") != "Linear" || g.value("stochastic_interpolation", false) ||
	    g.value("fixed_point_pos", false))
		fail("fused: grid must be a 3D linear hash grid with 16 levels x 2 features");
	if (b.value("otype", "") != "OneBlob" || b.value("n_dims_to_encode", 0u) != 5 || b.value("n_bins", 16u) != 4)
		fail("fused: second encoding must be OneBlob, 5 dims, 4 bins");
	if (i.value("otype", "") != "Identity" || i.value("scale", 1.0f) != 1.0f || i.value("offset", 0.0f) != 0.0f)
		fail("fused: third encoding must be a plain Identity");
	if (net.value("otype", "") != "FullyFusedMLP" || net.value("n_neurons", 0u) != 64 || net.value("activation", "") != "ReLU" ||
	    net.value("output_activation", "None") != "None")
		fail("fused: network must be FullyFusedMLP, 64 neurons, ReLU, output None");
	nrc_fused::Model m;
	m.hidden = (int)net.value("n_hidden_layers", 0u);
	m.log2_hashmap = g.value("log2_hashmap_size", 19u);
	m.base_resolution = g.value("base_resolution", 16u);
	m.per_level_scale = g.value("per_level_scale", 2.0f);
	uint32_t off[nrc_fused::N_LEVELS + 1];
	const uint32_t entries = nrc_fused::grid_offsets(m, off);
	const size_t n_enc = e.network->encoding()->n_params(), n_net = e.network->n_params() - n_enc;
	if (e.network->encoding()->padded_output_width() != 64 || e.network->padded_output_width() != 16 ||
	    n_enc != (size_t)entries * 2 || n_net != (size_t)m.hidden * 4096 + 1024)
		fail("fused: tcnn sizes differ from the fused layout (encoding width " +
		     std::to_string(e.network->encoding()->padded_output_width()) + ", grid params " + std::to_string(n_enc) +
		     " vs " + std::to_string(entries * 2) + ", network params " + std::to_string(n_net) + ")");
	return m;
}

__global__ void random_queries(uint32_t n, uint32_t seed, __half* out) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n * NRC_IN) return;
	uint32_t h = i * 0x9E3779B9u ^ seed;
	h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
	out[i] = __float2half((i % NRC_IN) < N_USED ? (h >> 8) * (1.0f / 16777216.0f) : 0.0f);
}

// Fused vs tcnn on 2^14 random queries with the current EMA weights. Returns the share of outputs outside
// 2% + 0.01 (fp16 tolerance with margin; a fragment or layout bug moves most outputs by O(1)).
double fused_self_check(NrcEngine& e, cudaStream_t stream, double* max_rel) {
	const uint32_t n = 1 << 14;
	GPUMemory<__half> in((size_t)n * NRC_IN);
	GPUMemory<float> a((size_t)n * 3), b((size_t)n * 3);
	random_queries<<<(n * NRC_IN + 255) / 256, 256, 0, stream>>>(n, 12345u, in.data());
	e.infer_tcnn(stream, n, in.data(), a.data());
	e.infer_fused(stream, n, in.data(), b.data());
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	std::vector<float> ha((size_t)n * 3), hb((size_t)n * 3);
	CUDA_CHECK_THROW(cudaMemcpy(ha.data(), a.data(), ha.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_THROW(cudaMemcpy(hb.data(), b.data(), hb.size() * sizeof(float), cudaMemcpyDeviceToHost));
	size_t bad = 0;
	*max_rel = 0;
	for (size_t k = 0; k < ha.size(); ++k) {
		const double d = std::fabs((double)hb[k] - ha[k]);
		*max_rel = std::max(*max_rel, d / (std::fabs(ha[k]) + 1e-2));
		bad += !(d <= 0.02 * std::fabs(ha[k]) + 0.01);  // NaN counts as bad
	}
	return (double)bad / ha.size();
}

// Selects the fused core for "wgmma" / "mma"; falls back (wgmma -> mma -> tcnn) when setup or the self check fails.
void select_engine(NrcEngine& e, cudaStream_t stream) {
	e.engine = e.config.value("engine", "tcnn");
	if (e.engine == "tcnn") return;
	if (e.engine != "wgmma" && e.engine != "mma") throw std::runtime_error{"unknown engine \"" + e.engine + "\" (tcnn, wgmma, mma)"};
	const nrc_fused::Model m = fused_model(e);
	int dev, major, minor;
	CUDA_CHECK_THROW(cudaGetDevice(&dev));
	CUDA_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
	CUDA_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
	std::vector<nrc_fused::Core> cores;
	if (e.engine == "wgmma" && major == 9 && minor == 0) cores.push_back(nrc_fused::CORE_WGMMA);
	cores.push_back(nrc_fused::CORE_MMA);
	const int wg = e.config["nrc"].value("fused_wg", 0);
	// Grid levels per sweep of the split grid pass; 0 = gather all levels inside the fused kernel; -1 (default) = the
	// most levels whose table slice fits in 1/6 of L2 (A40: 1, H100: 4 at 2^19 entries), so the gathers hit L2.
	int lpp = e.config["nrc"].value("fused_grid_levels", -1);
	if (lpp < 0) {
		int l2 = 0;
		CUDA_CHECK_THROW(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
		lpp = (int)std::clamp<int64_t>((int64_t)l2 / (6 * ((int64_t)4 << m.log2_hashmap)), 1, nrc_fused::N_LEVELS);
	}
	if (lpp > nrc_fused::N_LEVELS) throw std::runtime_error{"nrc.fused_grid_levels must be -1..16"};
	for (nrc_fused::Core c : cores) {
		const char* name = c == nrc_fused::CORE_WGMMA ? "wgmma" : "mma";
		const char* err = nullptr;
		if (!nrc_fused::setup(e.plan, m, c, wg, &err)) {
			fprintf(stderr, "cycles_nrc: fused %s core unavailable: %s\n", name, err);
			continue;
		}
		e.plan.levels_per_pass = lpp;
		if (lpp) e.enc.resize((size_t)round_up(e.config["nrc"].value("warmup_infer_batch", 1u << 18)) * 16);
		e.fused = true;
		double max_rel = 0;
		const double bad = fused_self_check(e, stream, &max_rel);
		fprintf(stderr, "cycles_nrc: engine %s: fused %s core, %d warpgroups x %d CTAs/SM, %d regs, grid %s; self check "
		        "vs tcnn: %.4f%% outside 2%%+0.01, max rel %.3g\n", e.engine.c_str(), name, e.plan.wg, e.plan.blocks_per_sm,
		        e.plan.regs, lpp ? ("pass of " + std::to_string(lpp) + " levels").c_str() : "gathered in the kernel",
		        100 * bad, max_rel);
		if (bad <= 1e-3) return;
		e.fused = false;
		fprintf(stderr, "cycles_nrc: fused %s core FAILED its self check; falling back\n", name);
	}
	fprintf(stderr, "cycles_nrc: engine %s unavailable; using tcnn inference\n", e.engine.c_str());
}

__global__ void scale_rgb(uint32_t n, float s, float* __restrict__ rgb) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) rgb[i] *= s;
}

// ---- Bake file (analysis/nrc_bake.md) ----
// "NRCBAKE1" | uint64 header bytes | header JSON | uint64 blob bytes | blob = msgpack of Trainer::serialize(opt)
// Header: {"format": 1, "config": full engine config, "n_params", "params_type", "n_train_steps", "optimizer": bool,
// "meta": free-form (CYCLES_NRC_META JSON at save time, e.g. scene, records, held-out loss)}.
// The blob's "params_binary" holds the inference (EMA) weights; load writes them to both the training and the EMA
// weights, so a weights-only file gives the same inference as the engine that saved it.
constexpr char BAKE_MAGIC[8] = {'N', 'R', 'C', 'B', 'A', 'K', 'E', '1'};

void save_state(NrcEngine& e, const char* path, bool with_optimizer) {
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	json header = {{"format", 1}, {"config", e.config}, {"n_params", e.network->n_params()},
	               {"params_type", "__half"}, {"n_train_steps", e.n_train_steps}, {"optimizer", with_optimizer},
	               {"n_dumped", e.n_dumped}};
	header["config"].erase("nrc");    // workspace sizes and engine choice belong to the loading side
	header["config"].erase("engine");
	if (const char* m = getenv("CYCLES_NRC_META")) {
		try { header["meta"] = json::parse(m); } catch (...) { header["meta"] = std::string{m}; }
	}
	const std::string hs = header.dump();
	const std::vector<uint8_t> blob = json::to_msgpack(e.trainer->serialize(with_optimizer));
	const std::string tmp = std::string{path} + ".tmp";
	FILE* f = fopen(tmp.c_str(), "wb");
	if (!f) throw std::runtime_error{"cannot write " + tmp};
	const uint64_t hl = hs.size(), bl = blob.size();
	fwrite(BAKE_MAGIC, 1, 8, f);
	fwrite(&hl, 8, 1, f);
	fwrite(hs.data(), 1, hl, f);
	fwrite(&bl, 8, 1, f);
	fwrite(blob.data(), 1, bl, f);
	if (fclose(f) != 0 || rename(tmp.c_str(), path) != 0) throw std::runtime_error{std::string{"cannot write "} + path};
	fprintf(stderr, "cycles_nrc: saved %s (%zu params, %llu steps, optimizer %d, %.1f MB)\n", path,
	        (size_t)e.network->n_params(), (unsigned long long)e.n_train_steps, (int)with_optimizer, (8 + 16 + hl + bl) / 1e6);
}

// Reads a bake file: header JSON and the trainer blob (msgpack).
json read_state(const char* path, std::vector<uint8_t>* blob) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error{std::string{"cannot read "} + path};
	char magic[8];
	uint64_t hl = 0, bl = 0;
	f.read(magic, 8);
	if (!f || memcmp(magic, BAKE_MAGIC, 8) != 0) throw std::runtime_error{std::string{path} + " is not an NRCBAKE1 file"};
	f.read((char*)&hl, 8);
	std::string hs(hl, '\0');
	f.read(hs.data(), hl);
	f.read((char*)&bl, 8);
	blob->resize(bl);
	f.read((char*)blob->data(), bl);
	if (!f) throw std::runtime_error{std::string{path} + " is truncated"};
	return json::parse(hs);
}

bool env_on(const char* name) {
	const char* v = getenv(name);
	return v && *v && atoi(v) != 0;
}

}  // namespace

extern "C" NrcEngine* nrc_create(const char* json_config, void* cu_stream) {
	try {
		auto e = std::make_unique<NrcEngine>();
		cu_check(cuCtxGetCurrent(&e->ctx), "cuCtxGetCurrent");
		if (!e->ctx) throw std::runtime_error{"no CUDA context is current (call inside Cycles' CUDAContextScope)"};

		// The runtime API binds the primary context of the current device. Cycles uses the primary context, so
		// both APIs see one context; refuse anything else instead of silently creating a second context.
		CUdevice dev;
		cu_check(cuCtxGetDevice(&dev), "cuCtxGetDevice");
		CUcontext primary = nullptr;
		cu_check(cuDevicePrimaryCtxRetain(&primary, dev), "cuDevicePrimaryCtxRetain");
		cuDevicePrimaryCtxRelease(dev);
		if (primary != e->ctx) throw std::runtime_error{"current context is not the device's primary context"};
		CUDA_CHECK_THROW(cudaSetDevice((int)dev));  // binds the same (primary) context; makes cudaGetDevice agree

		e->config = default_config();
		const json user = json_config && json_config[0] ? json::parse(json_config) : json::object();
		cudaStream_t stream = (cudaStream_t)cu_stream;
		e->create_stream = stream;
		if (user.value("engine", "tcnn") == "infer") {
			// Frozen inference (nrc_infer.cu): weights from CYCLES_NRC_LOAD or "nrc": {"weights": path} (NRCINF1).
			if (user.contains("nrc")) e->config["nrc"].update(user["nrc"]);
			std::string path = e->config["nrc"].value("weights", "");
			if (const char* lp = getenv("CYCLES_NRC_LOAD"); lp && *lp) path = lp;
			if (path.empty()) throw std::runtime_error{"engine infer needs CYCLES_NRC_LOAD or nrc.weights (an NRCINF1 file)"};
			nrcw::HostModel hm;
			std::string err;
			if (!nrcw::load_nrcinf(path.c_str(), hm, &err)) throw std::runtime_error{err};
			const json& o = e->config["nrc"];
			const nrcw::SiluImpl si = o.value("infer_silu", "ex2") == "h2" ? nrcw::SILU_H2 : nrcw::SILU_EX2;
			if (!nrcw::create(e->iplan, hm, o.value("infer_nc", 0), &err, si, o.value("infer_cluster", 1), o.value("infer_rs", 0) != 0)) throw std::runtime_error{err};
			if (const int lpp = o.value("infer_grid_levels", -1); lpp >= 0) e->iplan.levels_per_pass = std::min(lpp, 16);
			e->iplan.sort_bits = o.value("infer_sort_bits", 0);
			e->frozen_infer = true;
			e->engine = "infer";
			if (const uint32_t ew = nrcw::enc_half2(e->iplan)) e->enc.resize((size_t)round_up(o.value("warmup_infer_batch", 1u << 18)) * ew);
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			fprintf(stderr, "cycles_nrc: engine infer: %s, width %d, depth %d, %s, %s, %d consumer warpgroups, %d of %d "
			        "chunks in %d stages (%s), %zu B smem, %d regs, grid levels per pass %d\n", path.c_str(), hm.width, hm.depth,
			        hm.fp8 ? "fp8" : "fp16", hm.act == nrcw::ACT_SILU ? "silu" : "relu", e->iplan.nc,
			        std::min(e->iplan.chunks, e->iplan.stages), e->iplan.chunks, e->iplan.stages,
			        e->iplan.resident ? "resident" : "streamed", e->iplan.smem, e->iplan.regs, e->iplan.levels_per_pass);
			if (hm.encoding == nrcw::ENC_PERMUTO)
				fprintf(stderr, "cycles_nrc: engine infer: permuto encoding, d %d, %d levels x %d features, 2^%d entries per level\n",
				        hm.p_d, hm.p_levels, hm.p_features, hm.p_log2_T);
			return e.release();
		}
		// CYCLES_NRC_LOAD: the file's model (encoding, network, loss, optimizer) replaces the defaults; the caller's
		// config below may still set "nrc" (workspace) and "engine", but not the model, which must match the weights.
		const char* load_path = getenv("CYCLES_NRC_LOAD");
		std::vector<uint8_t> load_blob;
		json load_header;
		if (load_path && *load_path) {
			load_header = read_state(load_path, &load_blob);
			for (auto it = load_header["config"].begin(); it != load_header["config"].end(); ++it) e->config[it.key()] = it.value();
		}
		for (auto it = user.begin(); it != user.end(); ++it) {
			if (it.key() == "nrc") e->config["nrc"].update(it.value());  // "engine" is a top-level key too
			else if (load_path && *load_path && it.key() != "engine" && it.key() != "optimizer" && it.key() != "loss") continue;  // model (encoding, network) comes from the file
			else e->config[it.key()] = it.value();
		}

		auto model = create_from_config(N_USED, N_OUT, e->config);
		e->loss = model.loss;
		e->optimizer = model.optimizer;
		e->network = std::static_pointer_cast<NetworkWithInputEncoding<precision_t>>(model.network);
		const uint32_t seed = e->config["nrc"].value("seed", 1337u);
		e->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(e->network, e->optimizer, e->loss, seed);

		// Warm-up: size the staging buffers and the stream's workspace arena, and record the training CUDA graph,
		// so steady-state calls neither allocate nor synchronise. A training step without the optimizer only
		// overwrites the gradient buffer, so the weights stay at their initial values.
		const uint32_t wi = round_up(e->config["nrc"].value("warmup_infer_batch", 1u << 18));
		const uint32_t wt = round_up(e->config["nrc"].value("warmup_train_batch", 1u << 14));
		e->ensure(std::max(wi, wt));
		CUDA_CHECK_THROW(cudaMemsetAsync(e->in_f.data(), 0, e->in_f.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(e->target_f.data(), 0, e->target_f.get_bytes(), stream));
		{
			GPUMatrix<float> x{e->in_f.data(), N_USED, wi};
			GPUMatrix<precision_t> y{e->out_h.data(), e->network->padded_output_width(), wi};
			e->network->inference_mixed_precision(stream, x, y, true);
			GPUMatrix<float> xt{e->in_f.data(), N_USED, wt};
			GPUMatrix<float> tt{e->target_f.data(), N_OUT, wt};
			e->frozen = env_on("CYCLES_NRC_FREEZE");
			if (!e->frozen) e->trainer->training_step(stream, xt, tt, nullptr, /*run_optimizer=*/false);
		}
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		if (load_path && *load_path) {
			if (load_header.value("n_params", (size_t)0) != e->network->n_params())
				throw std::runtime_error{std::string{load_path} + ": parameter count differs from the model"};
			e->trainer->deserialize(json::from_msgpack(load_blob));
			e->n_train_steps = load_header.value("n_train_steps", (uint64_t)0);
			e->loaded_meta = load_header;
			e->out_scale = load_header.value("output_scale", 1.0f);
			fprintf(stderr, "cycles_nrc: loaded %s (%zu params, %llu steps, frozen %d, output scale %.4f)\n", load_path,
			        (size_t)e->network->n_params(), (unsigned long long)e->n_train_steps, (int)e->frozen, e->out_scale);
		}
		if (const char* sp = getenv("CYCLES_NRC_SAVE")) e->save_path = sp;
		if (const char* dp = getenv("CYCLES_NRC_DUMP")) {
			if (*dp) {
				e->dump = fopen(dp, "ab");
				if (!e->dump) throw std::runtime_error{std::string{"cannot open CYCLES_NRC_DUMP "} + dp};
			}
		}
		select_engine(*e, stream);
		return e.release();
	} catch (const std::exception& ex) {
		report("nrc_create", ex);
		return nullptr;
	}
}

extern "C" int nrc_infer(NrcEngine* e, int n, const void* d_inputs, float* d_rgb, void* cu_stream) {
	if (!e || n < 0) return -1;
	try {
		CtxGuard g{e->ctx};
		e->infer((cudaStream_t)cu_stream, (uint32_t)n, (const __half*)d_inputs, d_rgb);
		if (e->out_scale != 1.0f && n > 0)
			scale_rgb<<<(3 * (uint32_t)n + 255) / 256, 256, 0, (cudaStream_t)cu_stream>>>(3 * (uint32_t)n, e->out_scale, d_rgb);
		return 0;
	} catch (const std::exception& ex) {
		report("nrc_infer", ex);
		return -1;
	}
}

extern "C" int nrc_train(NrcEngine* e, int n, const void* d_inputs, const float* d_target_rgb, int steps, void* cu_stream) {
	if (!e || n < 0) return -1;
	try {
		CtxGuard g{e->ctx};
		e->train((cudaStream_t)cu_stream, (uint32_t)n, (const __half*)d_inputs, d_target_rgb, (uint32_t)std::max(steps, 1));
		return 0;
	} catch (const std::exception& ex) {
		report("nrc_train", ex);
		return -1;
	}
}

extern "C" void nrc_destroy(NrcEngine* e) {
	if (!e) return;
	try {
		CtxGuard g{e->ctx};
		CUDA_CHECK_THROW(cudaStreamSynchronize(e->create_stream));
		if (!e->save_path.empty()) save_state(*e, e->save_path.c_str(), env_on("CYCLES_NRC_SAVE_OPT"));
		if (e->dump) {
			fclose(e->dump);
			fprintf(stderr, "cycles_nrc: dumped %llu records\n", (unsigned long long)e->n_dumped);
		}
		cudaStream_t s = e->create_stream;
		const bool tcnn_arena = !e->frozen_infer;
		delete e;
		// Release the workspace arena tcnn keeps per stream (it would otherwise live until process exit).
		if (tcnn_arena) free_gpu_memory_arena(s);
	} catch (const std::exception& ex) {
		report("nrc_destroy", ex);
	}
}

// ---- Test-only accessors (cycles_nrc_debug.h; not part of the Cycles ABI) ----

extern "C" int nrc_debug_params(NrcEngine* e, int inference, void* host_fp16, size_t capacity) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	try {
		CtxGuard g{e->ctx};
		const size_t n = e->network->n_params();
		if (!host_fp16) return (int)n;
		if (capacity < n) return -1;
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
		CUDA_CHECK_THROW(cudaMemcpy(host_fp16, inference ? e->trainer->params_inference() : e->trainer->params(),
		                            n * sizeof(precision_t), cudaMemcpyDeviceToHost));
		return (int)n;
	} catch (const std::exception& ex) {
		report("nrc_debug_params", ex);
		return -1;
	}
}

extern "C" int nrc_debug_layout(NrcEngine* e, int* n_network_params, int* encoding_width, int* padded_output_width) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	*n_network_params = (int)e->network->n_params() - (int)e->network->encoding()->n_params();
	*encoding_width = (int)e->network->encoding()->padded_output_width();
	*padded_output_width = (int)e->network->padded_output_width();
	return 0;
}

extern "C" const char* nrc_debug_config(NrcEngine* e) {
	static thread_local std::string s;
	s = e->config.dump();
	return s.c_str();
}

// Runs only the input encoding on n queries and copies the padded encoding (row j = query j) to host as float.
extern "C" int nrc_debug_encode(NrcEngine* e, int n, const void* d_inputs, float* host_out, void* cu_stream) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	try {
		CtxGuard g{e->ctx};
		cudaStream_t stream = (cudaStream_t)cu_stream;
		const uint32_t m = round_up((uint32_t)n);
		e->ensure(m);
		gather_inputs<<<(m + 255) / 256, 256, 0, stream>>>(m, n, 0, 1, true, (const __half*)d_inputs, e->in_f.data());
		auto& enc = e->network->encoding();
		const uint32_t w = enc->padded_output_width();
		GPUMatrix<float> x{e->in_f.data(), N_USED, m};
		GPUMatrixDynamic<precision_t> y{w, m, stream, CM};
		enc->inference_mixed_precision(stream, x, y, true);
		std::vector<precision_t> h((size_t)w * m);
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		CUDA_CHECK_THROW(cudaMemcpy(h.data(), y.data(), h.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
		for (int j = 0; j < n; ++j)
			for (uint32_t d = 0; d < w; ++d) host_out[(size_t)j * w + d] = (float)h[(size_t)j * w + d];
		return 0;
	} catch (const std::exception& ex) {
		report("nrc_debug_encode", ex);
		return -1;
	}
}

// tcnn's inference on the same EMA weights, whatever the selected engine (exactness checks and A/B timing).
extern "C" int nrc_debug_infer_tcnn(NrcEngine* e, int n, const void* d_inputs, float* d_rgb, void* cu_stream) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	try {
		CtxGuard g{e->ctx};
		if (n > 0) e->infer_tcnn((cudaStream_t)cu_stream, (uint32_t)n, (const __half*)d_inputs, d_rgb);
		return 0;
	} catch (const std::exception& ex) {
		report("nrc_debug_infer_tcnn", ex);
		return -1;
	}
}

extern "C" const char* nrc_debug_engine(NrcEngine* e) {
	static thread_local std::string s;
	if (e->frozen_infer) s = "infer w=" + std::to_string(e->iplan.width) + " d=" + std::to_string(e->iplan.depth) +
	                         (e->iplan.fp8 ? " fp8" : " fp16") + " nc=" + std::to_string(e->iplan.nc) +
	                         (e->iplan.encoding == nrcw::ENC_PERMUTO ? " enc=permuto" + std::to_string(e->iplan.p_d) : "");
	else if (!e->fused) s = "tcnn";
	else
		s = std::string{e->plan.core == nrc_fused::CORE_WGMMA ? "wgmma" : "mma"} + " wg=" + std::to_string(e->plan.wg) +
		    " ctas_per_sm=" + std::to_string(e->plan.blocks_per_sm) + " grid=" + std::to_string(e->plan.grid) +
		    " regs=" + std::to_string(e->plan.regs) + " smem=" + std::to_string(e->plan.smem) +
		    " grid_levels=" + std::to_string(e->plan.levels_per_pass);
	return s.c_str();
}

// ---- Bake API (nrc-bake; analysis/nrc_bake.md). Not used by Cycles. ----

extern "C" int nrc_save(NrcEngine* e, const char* path, int with_optimizer) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	try {
		CtxGuard g{e->ctx};
		save_state(*e, path, with_optimizer != 0);
		return 0;
	} catch (const std::exception& ex) {
		report("nrc_save", ex);
		return -1;
	}
}

extern "C" int nrc_set_learning_rate(NrcEngine* e, float lr) {
	if (!e->network) return -1;  // engine infer has no tcnn model
	e->optimizer->set_learning_rate(lr);
	return 0;
}

extern "C" long long nrc_train_steps(NrcEngine* e) { return (long long)e->n_train_steps; }
