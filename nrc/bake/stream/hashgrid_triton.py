"""Triton hash grid for bake_stream.py: tcnn's 3D grid, 2 features per level, coherent-prime hash, linear interpolation,
dense levels indexed by stride. Forward gathers 8 corners per level; backward scatters with atomics."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grid_fwd(x_ptr, tab_ptr, out_ptr, scale_ptr, res_ptr, size_ptr, off_ptr, dense_ptr, B,
              L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lv = tl.program_id(1)
    b = pid * BLOCK + tl.arange(0, BLOCK)
    m = b < B
    s = tl.load(scale_ptr + lv)
    r = tl.load(res_ptr + lv).to(tl.int64)
    sz = tl.load(size_ptr + lv).to(tl.int64)
    off = tl.load(off_ptr + lv).to(tl.int64)
    dn = tl.load(dense_ptr + lv)
    px = tl.load(x_ptr + b * 3 + 0, mask=m, other=0.0) * s + 0.5
    py = tl.load(x_ptr + b * 3 + 1, mask=m, other=0.0) * s + 0.5
    pz = tl.load(x_ptr + b * 3 + 2, mask=m, other=0.0) * s + 0.5
    fx = tl.floor(px); fy = tl.floor(py); fz = tl.floor(pz)
    rx = px - fx; ry = py - fy; rz = pz - fz
    gx = fx.to(tl.int64); gy = fy.to(tl.int64); gz = fz.to(tl.int64)
    a0 = tl.zeros([BLOCK], dtype=tl.float32)
    a1 = tl.zeros([BLOCK], dtype=tl.float32)
    for c in tl.static_range(8):
        cx = c & 1; cy = (c >> 1) & 1; cz = (c >> 2) & 1
        ix = gx + cx; iy = gy + cy; iz = gz + cz
        w = (rx if cx else 1 - rx) * (ry if cy else 1 - ry) * (rz if cz else 1 - rz)
        idd = ix + iy * r + iz * r * r
        idh = (ix ^ (iy * 2654435761) ^ (iz * 805459861)) & 0xFFFFFFFF
        idx = tl.where(dn != 0, idd, idh) % sz + off
        a0 += w * tl.load(tab_ptr + idx * 2, mask=m, other=0.0)
        a1 += w * tl.load(tab_ptr + idx * 2 + 1, mask=m, other=0.0)
    tl.store(out_ptr + b * (2 * L) + lv * 2, a0, mask=m)
    tl.store(out_ptr + b * (2 * L) + lv * 2 + 1, a1, mask=m)


@triton.jit
def _grid_bwd(x_ptr, gout_ptr, gtab_ptr, scale_ptr, res_ptr, size_ptr, off_ptr, dense_ptr, B,
              L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lv = tl.program_id(1)
    b = pid * BLOCK + tl.arange(0, BLOCK)
    m = b < B
    s = tl.load(scale_ptr + lv)
    r = tl.load(res_ptr + lv).to(tl.int64)
    sz = tl.load(size_ptr + lv).to(tl.int64)
    off = tl.load(off_ptr + lv).to(tl.int64)
    dn = tl.load(dense_ptr + lv)
    px = tl.load(x_ptr + b * 3 + 0, mask=m, other=0.0) * s + 0.5
    py = tl.load(x_ptr + b * 3 + 1, mask=m, other=0.0) * s + 0.5
    pz = tl.load(x_ptr + b * 3 + 2, mask=m, other=0.0) * s + 0.5
    fx = tl.floor(px); fy = tl.floor(py); fz = tl.floor(pz)
    rx = px - fx; ry = py - fy; rz = pz - fz
    gx = fx.to(tl.int64); gy = fy.to(tl.int64); gz = fz.to(tl.int64)
    g0 = tl.load(gout_ptr + b * (2 * L) + lv * 2, mask=m, other=0.0)
    g1 = tl.load(gout_ptr + b * (2 * L) + lv * 2 + 1, mask=m, other=0.0)
    for c in tl.static_range(8):
        cx = c & 1; cy = (c >> 1) & 1; cz = (c >> 2) & 1
        ix = gx + cx; iy = gy + cy; iz = gz + cz
        w = (rx if cx else 1 - rx) * (ry if cy else 1 - ry) * (rz if cz else 1 - rz)
        idd = ix + iy * r + iz * r * r
        idh = (ix ^ (iy * 2654435761) ^ (iz * 805459861)) & 0xFFFFFFFF
        idx = tl.where(dn != 0, idd, idh) % sz + off
        tl.atomic_add(gtab_ptr + idx * 2, w * g0, mask=m, sem="relaxed")
        tl.atomic_add(gtab_ptr + idx * 2 + 1, w * g1, mask=m, sem="relaxed")


class _GridFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, meta):
        x = x.contiguous().float()
        B, L = x.shape[0], meta["L"]
        out = torch.empty(B, 2 * L, device=x.device, dtype=torch.float32)
        BLOCK = 256
        _grid_fwd[(triton.cdiv(B, BLOCK), L)](x, table, out, meta["scale"], meta["res"], meta["size"], meta["off"],
                                             meta["dense"], B, L=L, BLOCK=BLOCK)
        ctx.save_for_backward(x)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx, gout):
        (x,) = ctx.saved_tensors
        meta = ctx.meta
        B, L = x.shape[0], meta["L"]
        gtab = torch.zeros(meta["n_entries"], 2, device=x.device, dtype=torch.float32)
        BLOCK = 256
        _grid_bwd[(triton.cdiv(B, BLOCK), L)](x, gout.contiguous().float(), gtab, meta["scale"], meta["res"],
                                             meta["size"], meta["off"], meta["dense"], B, L=L, BLOCK=BLOCK)
        return None, gtab, None


def patch(grid):
    """Make a bake_stream.HashGrid use the Triton kernels."""
    dev = grid.table.device
    meta = dict(L=len(grid.res), n_entries=grid.n_entries,
                scale=torch.tensor(grid.scale, device=dev, dtype=torch.float32),
                res=torch.tensor(grid.res, device=dev, dtype=torch.int32),
                size=torch.tensor(grid.size, device=dev, dtype=torch.int32),
                off=torch.tensor(grid.offset, device=dev, dtype=torch.int64),
                dense=torch.tensor([int(d) for d in grid.dense], device=dev, dtype=torch.int32))
    grid.forward = lambda x: _GridFn.apply(x, grid.table, meta)
    return grid
