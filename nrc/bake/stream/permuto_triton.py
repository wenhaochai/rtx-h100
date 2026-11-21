"""Permutohedral lattice grid encoding in Triton for bake_stream.py --enc permuto5, after tiny-cuda-nn's PermutoEncoding
(include/tiny-cuda-nn/encodings/internal/permuto.h) and PermutoSDF (Rosu and Behnke, CVPR 2023).

  enc = PermutoGrid(d, n_levels, n_features, log2_T, res_lo, res_hi)
  y = enc(x)            # x: B x d in [0,1]^d  ->  y: B x (n_levels * n_features), float32

Per level l, as tcnn's kernel_permuto: shift and scale each input dim, c_j = (x_j + shift_lj) * scale_lj, elevate to the
(d+1)-dim hyperplane sum = 0 (elevated[k] = sum_{j>=k} c_j - k c_{k-1}), round each elevated coordinate to the nearest
multiple of d+1, rank the remainders to find the enclosing simplex, compute the d+1 barycentric weights, build the d+1
vertex keys, hash each key with tcnn's base_convert_hash (h = (h + key_j) * 2531011, uint32) into the level's
2^log2_T-entry table, and blend the vertices' features with the weights. Random per-level shifts in [-5, 5) drawn
from torch's generator with seed 1337; table init uniform +-1e-4. nrc/src/nrc_infer.cu permuto_kernel computes the
same forward in the same fp32 operation order.

Scales: res_lo/res_hi are hash-grid-equivalent resolutions per input dim, geometric over the levels. Resolution r means
as many lattice vertices per unit volume as a cubic grid of r^d cells, so tcnn's scale is (d+1)^((d-1/2)/d) * r_j /
sqrt((j+1)(j+2)).

The backward scatters the table gradient with one fp32 vector atomic per vertex over its F contiguous features
(no input gradient: the inputs are data). Kernel sources are generated per d, with every per-dim value in its own
scalar as tcnn's unrolled loops, and written to $TMPDIR/permuto_triton_gen/ (Triton jit needs a source file).
"""
import hashlib, importlib.util, math, os, tempfile

import torch
import triton
import triton.language as tl  # noqa: F401  (used by the generated kernels)

HASH_P = 2531011   # tcnn base_convert_hash multiplier
BLOCK = {1: 256, 2: 256, 3: 256, 4: 256, 5: 128, 6: 128, 7: 128, 8: 128}


def lattice_scale(d):
    """tcnn scale per unit of hash-grid-equivalent resolution (equal vertex density)."""
    return (d + 1) ** ((d - 0.5) / d)


def _prelude(d):
    D1 = d + 1
    s = ["    pid = tl.program_id(0)", "    lv = tl.program_id(1)",
         "    b = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)", "    m = b < B", "    fo = tl.arange(0, F)"]
    for j in range(d):   # c_j = (x_j + shift) * scale_j  (scale includes rsqrt((j+1)(j+2)))
        s.append(f"    c{j} = (tl.load(x_ptr + b * {d} + {j}, mask=m, other=0.0) + tl.load(sh_ptr + lv * {d} + {j})) "
                 f"* tl.load(sc_ptr + lv * {d} + {j})")
    s.append(f"    e{d} = 0.0 - {float(d)!r} * c{d - 1}")   # elevate, as tcnn: dim = d..1, sum accumulates from the top
    s.append(f"    sm = c{d - 1}")
    for k in range(d - 1, 0, -1):
        s.append(f"    e{k} = sm - {float(k)!r} * c{k - 1}")
        s.append(f"    sm = sm + c{k - 1}")
    s.append("    e0 = sm")
    inv = repr(1.0 / D1)
    for i in range(D1):   # nearest multiple of d+1
        s.append(f"    v{i} = e{i} * {inv}")
        s.append(f"    up{i} = tl.ceil(v{i}) * {float(D1)!r}")
        s.append(f"    dn{i} = tl.floor(v{i}) * {float(D1)!r}")
        s.append(f"    rf{i} = tl.where(up{i} - e{i} < e{i} - dn{i}, up{i}, dn{i})")
        s.append(f"    q{i} = rf{i}.to(tl.int32)")
        s.append(f"    di{i} = e{i} - rf{i}")
        s.append(f"    k{i} = tl.zeros([BLOCK], dtype=tl.int32)")
    s.append("    qs = (" + " + ".join(f"q{i}" for i in range(D1)) + f") // {D1}")
    for i in range(d):    # rank: pairs i < j, the smaller remainder gets +1 (tcnn's order and tie rule)
        for j in range(i + 1, D1):
            s.append(f"    t = (di{i} < di{j}).to(tl.int32)")
            s.append(f"    k{i} += t")
            s.append(f"    k{j} += 1 - t")
    for i in range(D1):   # wrap with rem0's sum
        s.append(f"    k{i} += qs")
        s.append(f"    lo = k{i} < 0")
        s.append(f"    hi = k{i} > {d}")
        s.append(f"    k{i} = tl.where(lo, k{i} + {D1}, tl.where(hi, k{i} - {D1}, k{i}))")
        s.append(f"    q{i} = tl.where(lo, q{i} + {D1}, tl.where(hi, q{i} - {D1}, q{i}))")
        s.append(f"    dl{i} = (e{i} - q{i}.to(tl.float32)) * {inv}")
    for kk in range(d + 2):   # barycentric: bary[d - rank_i] += delta_i; bary[d + 1 - rank_i] -= delta_i (i in order)
        terms = []
        for i in range(D1):
            if 0 <= d - kk:
                terms.append(f"tl.where(k{i} == {d - kk}, dl{i}, 0.0)")
            if 0 <= d + 1 - kk <= d:
                terms.append(f"(0.0 - tl.where(k{i} == {d + 1 - kk}, dl{i}, 0.0))")
        s.append(f"    w{kk} = tl.zeros([BLOCK], dtype=tl.float32)")
        for t in terms:   # rank is a permutation: at most one + and one - term is non-zero per weight
            s.append(f"    w{kk} = w{kk} + {t}")
    s.append(f"    w0 = w0 + (1.0 + w{d + 1})")
    s.append("    lvo = lv.to(tl.int64) * (hmask + 1)")
    return s


def _vertex(d, kk):
    """Lines computing the table row `row` of vertex kk (remainder-kk vertex)."""
    D1 = d + 1
    s = ["    h = tl.zeros([BLOCK], dtype=tl.uint32)"]
    for j in range(d):
        s.append(f"    key = q{j} + {kk} - tl.where(k{j} > {d - kk}, {D1}, 0)")
        s.append(f"    h = (h + key.to(tl.uint32, bitcast=True)) * {HASH_P}")
    s.append("    row = (h & hmask).to(tl.int64) + lvo")
    return s


def _gen_source(d):
    pre = _prelude(d)
    fwd = ["@triton.jit", "def fwd(x_ptr, tab_ptr, out_ptr, sc_ptr, sh_ptr, B, hmask, LF: tl.constexpr, F: tl.constexpr, "
           "BLOCK: tl.constexpr):"] + pre + ["    acc = tl.zeros([BLOCK, F], dtype=tl.float32)"]
    for kk in range(d + 1):
        fwd += _vertex(d, kk)
        fwd.append("    val = tl.load(tab_ptr + row[:, None] * F + fo[None, :], mask=m[:, None], other=0.0)")
        fwd.append(f"    acc += w{kk}[:, None] * val")
    fwd.append("    tl.store(out_ptr + b[:, None] * LF + lv * F + fo[None, :], acc, mask=m[:, None])")
    bwd = ["@triton.jit", "def bwd(x_ptr, gout_ptr, gtab_ptr, sc_ptr, sh_ptr, B, hmask, LF: tl.constexpr, F: tl.constexpr, "
           "BLOCK: tl.constexpr):"] + pre
    bwd.append("    g = tl.load(gout_ptr + b[:, None] * LF + lv * F + fo[None, :], mask=m[:, None], other=0.0)")
    for kk in range(d + 1):   # one relaxed vector atomic per vertex (SASS REDG.E.ADD.F32x4, two for F = 8)
        bwd += _vertex(d, kk)
        bwd.append(f"    tl.atomic_add(gtab_ptr + row[:, None] * F + fo[None, :], w{kk}[:, None] * g, mask=m[:, None], "
                   "sem='relaxed')")
    return "\n".join([f"# generated by permuto_triton.py for d = {d}; do not edit", "import triton",
                      "import triton.language as tl", "", ""] + fwd + ["", ""] + bwd) + "\n"


_KERNELS = {}


def kernels(d):
    """Module with the generated fwd/bwd kernels for input dimension d."""
    if d not in _KERNELS:
        assert 1 <= d <= 8, d
        src = _gen_source(d)
        h = hashlib.sha1(src.encode()).hexdigest()[:12]
        dirn = os.path.join(tempfile.gettempdir(), "permuto_triton_gen")
        os.makedirs(dirn, exist_ok=True)
        path = os.path.join(dirn, f"permuto_d{d}_{h}.py")
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(src)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location(f"permuto_d{d}_{h}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _KERNELS[d] = mod
    return _KERNELS[d]


class _PermutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, meta):
        x = x.contiguous().float()
        d, L, F, hmask, blk = meta["d"], meta["L"], meta["F"], meta["hmask"], meta["block"]
        B = x.shape[0]
        out = torch.empty(B, L * F, device=x.device, dtype=torch.float32)
        if B:
            kernels(d).fwd[(triton.cdiv(B, blk), L)](x, table, out, meta["sc"], meta["sh"], B, hmask, LF=L * F, F=F, BLOCK=blk)
        ctx.save_for_backward(x)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx, gout):
        (x,) = ctx.saved_tensors
        meta = ctx.meta
        d, L, F, hmask, blk = meta["d"], meta["L"], meta["F"], meta["hmask"], meta["block"]
        B, gout = x.shape[0], gout.contiguous().float()
        gtab = torch.zeros(L * (hmask + 1), F, device=x.device, dtype=torch.float32)
        if B:
            kernels(d).bwd[(triton.cdiv(B, blk), L)](x, gout, gtab, meta["sc"], meta["sh"], B, hmask, LF=L * F, F=F, BLOCK=blk)
        return None, gtab, None


class PermutoGrid(torch.nn.Module):
    def __init__(self, d, n_levels, n_features, log2_T, res_lo, res_hi, seed=1337, device="cuda"):
        super().__init__()
        assert 1 <= d <= 8 and n_features in (1, 2, 4, 8) and 1 <= n_levels <= 32 and log2_T <= 30
        lo, hi = [float(v) for v in res_lo], [float(v) for v in res_hi]
        assert len(lo) == d and len(hi) == d
        self.d, self.L, self.F, self.log2_T = d, n_levels, n_features, log2_T
        cd = lattice_scale(d)
        res = [[lo[j] * (hi[j] / lo[j]) ** (l / max(1, n_levels - 1)) for j in range(d)] for l in range(n_levels)]
        sc = [[cd * res[l][j] / math.sqrt((j + 1) * (j + 2)) for j in range(d)] for l in range(n_levels)]
        g = torch.Generator().manual_seed(seed)
        sh = torch.rand(n_levels, d, generator=g, dtype=torch.float32) * 10.0 - 5.0
        self.register_buffer("sc", torch.tensor(sc, dtype=torch.float32, device=device))
        self.register_buffer("sh", sh.to(device))
        self.n_entries = n_levels << log2_T
        self.table = torch.nn.Parameter(torch.empty(self.n_entries, n_features, device=device).uniform_(-1e-4, 1e-4))
        self.block = BLOCK[d]

    @property
    def width(self):
        return self.L * self.F

    def forward(self, x):
        return _PermutoFn.apply(x, self.table, dict(d=self.d, L=self.L, F=self.F, hmask=(1 << self.log2_T) - 1,
                                                    block=self.block, sc=self.sc, sh=self.sh))
