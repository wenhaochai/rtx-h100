"""Offline bake of one scene's radiance cache from dumped training records (nrc/bake/dump.sh), each record used once.
Writes an NRCBAKE1 file for the nrc engine; calibrate it with nrc/bake/calibrate.sh as any other bake.

  bake_stream.py --dump <dir with s61.rec .. s69.rec> --out <dir>/bake.nrc [--config nrc/configs/t24.json]

Model: as configs/t24.json and the engine: the t24 hash grid (16 levels x 2 features, 2^24 entries per level,
resolution 128 to 2048) on position, one-blob (4 bins) on direction, normal and roughness, identity on the albedos,
padded with ones to 64, then a 64 x 5 ReLU MLP without biases and 16 padded outputs. Training: L2 loss, Adam with
eps 1e-15, one learning rate for grid and MLP, 3e-3 decayed to 1e-4 (cosine after a 2% warm-up), beta2 0.999 for the
first 10% of steps then 0.9999, and an exponential moving average of the weights whose decay rises from 0.99 to
0.9999 over 80% of the steps; the average is exported. Batches of 2^18 records, drawn without replacement from a GPU
buffer of 2^25 records filled with whole calls of all dump files in a shuffled order. The MLP runs in bf16 autocast.
Dump format, per engine call: [u32 n][n x 16 fp16 inputs][n x 3 fp32 targets].
"""
import argparse, glob, json, math, os, queue, struct, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import hashgrid_triton as HT

DEV = "cuda"
REC = 16 * 2 + 3 * 4   # bytes per record


class HashGrid(torch.nn.Module):
    """tcnn's multiresolution hash grid; levels whose full grid fits are dense (stride indexing)."""

    def __init__(self, n_levels=16, F_=2, log2_T=24, base=128, scale=1.203025):
        super().__init__()
        self.res, self.scale, self.size, self.offset = [], [], [], []
        off = 0
        for l in range(n_levels):
            s = 2.0 ** (l * math.log2(scale)) * base - 1.0
            r = int(math.ceil(s)) + 1
            p = min(r ** 3, 0x7FFFFFFF)
            p = (p + 7) // 8 * 8
            p = min(p, 1 << log2_T)
            self.res.append(r); self.scale.append(s); self.size.append(p); self.offset.append(off)
            off += p
        self.n_entries = off
        self.table = torch.nn.Parameter(torch.empty(off, F_, device=DEV).uniform_(-1e-4, 1e-4))
        self.dense = [(r <= 0x659 and r ** 3 <= sz) for r, sz in zip(self.res, self.size)]


def quartic_cdf(x, inv_r):
    u = x * inv_r
    u2 = u * u
    return torch.clamp(15.0 / 16.0 * u * (1 - 2.0 / 3.0 * u2 + 0.2 * u2 * u2) + 0.5, 0.0, 1.0)


def one_blob(x, n_bins=4):  # B x D -> B x D*n_bins, tcnn's wrap-around variant
    left = torch.arange(n_bins, device=x.device, dtype=x.dtype) / n_bins
    d = left[None, None, :] - x[:, :, None]
    lc = quartic_cdf(d, n_bins) + quartic_cdf(d - 1, n_bins) + quartic_cdf(d + 1, n_bins)
    rc = torch.cat([lc[..., 1:], lc[..., :1] + 1], -1)
    return (rc - lc).reshape(x.shape[0], -1)


class MLP(torch.nn.Module):
    """tcnn's FullyFusedMLP: depth hidden layers -> depth + 1 matrices, no biases, 16 padded outputs, Xavier init."""

    def __init__(self, n_in=64, width=64, depth=5):
        super().__init__()
        dims = [(n_in, width)] + [(width, width)] * (depth - 1) + [(width, 16)]
        self.layers = torch.nn.ParameterList()
        for fi, fo in dims:
            a = math.sqrt(6.0 / (fi + fo))
            self.layers.append(torch.nn.Parameter(torch.empty(fo, fi, device=DEV).uniform_(-a, a)))

    def forward(self, h):
        for w in self.layers[:-1]:
            h = F.relu(h @ w.t())
        return (h @ self.layers[-1].t())[:, :3]


class Net(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.grid = HT.patch(HashGrid())
        self.mlp = MLP(64, width, depth)

    def forward(self, x):  # x: B x 14 (position 3, one-blob 5, identity 6)
        x = x.float()
        e = torch.cat([self.grid(x[:, 0:3]), one_blob(x[:, 3:8]), x[:, 8:14]], 1)
        e = torch.cat([e, torch.ones(e.shape[0], 64 - e.shape[1], device=e.device)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.training):
            return self.mlp(e).float()


def calls_of(path):
    """(offset, n) of every call in a dump file."""
    out, off, sz = [], 0, os.path.getsize(path)
    with open(path, "rb") as f:
        while off < sz:
            f.seek(off)
            n = struct.unpack("<I", f.read(4))[0]
            out.append((off + 4, n))
            off += 4 + n * REC
    assert off == sz, f"truncated dump {path}"
    return out


def producer(files, buf_records, q, seed):
    """Host buffers of about buf_records records, from whole calls of all files in a shuffled order."""
    rng = np.random.default_rng(seed)
    order = [(i, c) for i, p in enumerate(files) for c in calls_of(p)]
    rng.shuffle(order)
    fh = [open(p, "rb") for p in files]
    xs, ys, k = [], [], 0
    for i, (off, n) in order:
        fh[i].seek(off)
        raw = np.frombuffer(fh[i].read(n * REC), dtype=np.uint8)
        xs.append(raw[: n * 32].view(np.float16).reshape(n, 16)[:, :14]); ys.append(raw[n * 32:].view(np.float32).reshape(n, 3))
        k += n
        if k >= buf_records:
            q.put((np.concatenate(xs), np.concatenate(ys))); xs, ys, k = [], [], 0
    if xs:
        q.put((np.concatenate(xs), np.concatenate(ys)))
    q.put(None)


def pack_msgpack(d):
    """The trainer blob nlohmann::json::from_msgpack reads: a map of strings, ints and one binary."""
    o = bytearray([0xdf]) + len(d).to_bytes(4, "big")
    for k, v in d.items():
        s = k.encode(); o += bytes([0xdb]) + len(s).to_bytes(4, "big") + s
        if isinstance(v, bytes): o += bytes([0xc6]) + len(v).to_bytes(4, "big") + v
        elif isinstance(v, int): o += bytes([0xcf]) + v.to_bytes(8, "big")
        else: s = v.encode(); o += bytes([0xdb]) + len(s).to_bytes(4, "big") + s
    return bytes(o)


def write_bake(path, net, config, steps, meta):
    """NRCBAKE1 (nrc/src/cycles_nrc.cu save_state): magic, header JSON, msgpack blob with the fp16 parameters in
    tcnn's order (input layer, hidden layers, output layer, then the hash grid)."""
    params = np.concatenate([p.detach().half().cpu().numpy().ravel() for p in net.mlp.layers]
                            + [net.grid.table.detach().half().cpu().numpy().ravel()])
    hdr = {"format": 1, "config": config, "n_params": int(params.size), "params_type": "__half",
           "n_train_steps": int(steps), "optimizer": False, "n_dumped": 0, "meta": meta}
    hs = json.dumps(hdr).encode()
    blob = pack_msgpack({"n_params": int(params.size), "params_binary": params.tobytes(), "params_type": "__half"})
    with open(path + ".tmp", "wb") as f:
        f.write(b"NRCBAKE1"); f.write(struct.pack("<Q", len(hs))); f.write(hs); f.write(struct.pack("<Q", len(blob))); f.write(blob)
    os.replace(path + ".tmp", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dir with s61.rec .. s69.rec")
    ap.add_argument("--out", required=True, help="output NRCBAKE1 file")
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "configs", "t24.json"))
    ap.add_argument("--batch", type=int, default=1 << 18)
    ap.add_argument("--buffer", type=int, default=1 << 25)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_end", type=float, default=1e-4)
    ap.add_argument("--max_records", type=float, default=0, help="stop after this many records (0 = all; smoke tests)")
    a = ap.parse_args()
    model = json.load(open(a.config))
    net_cfg = model["network"]
    assert net_cfg["otype"] == "FullyFusedMLP" and net_cfg["activation"] == "ReLU", net_cfg
    config = {k: model[k] for k in ("encoding", "network", "optimizer")}
    config["loss"] = {"otype": "L2"}
    files = sorted(glob.glob(f"{a.dump}/s6[1-9].rec"))
    assert len(files) == 9, files
    total = sum(n for p in files for _, n in calls_of(p))
    if a.max_records: total = min(total, int(a.max_records))
    steps = total // a.batch
    print(f"{len(files)} dump files, {total} records, {steps} steps", flush=True)

    torch.manual_seed(0)
    W, D = net_cfg["n_neurons"], net_cfg["n_hidden_layers"]
    net = Net(W, D)
    ema = Net(W, D); ema.load_state_dict(net.state_dict()); ema.requires_grad_(False)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-15, fused=True)
    warm = max(1, steps // 50)
    q = queue.Queue(maxsize=3)
    threading.Thread(target=producer, args=(files, a.buffer, q, 0), daemon=True).start()
    s, t0 = 0, time.time()
    while s < steps:
        item = q.get()
        if item is None: break
        X = torch.from_numpy(item[0]).to(DEV, non_blocking=True); Y = torch.from_numpy(item[1]).to(DEV, non_blocking=True)
        perm = torch.randperm(X.shape[0], device=DEV)
        for b in range(0, X.shape[0] - a.batch + 1, a.batch):
            if s >= steps: break
            s += 1
            f = s / warm if s <= warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g["lr"] = a.lr_end + (a.lr - a.lr_end) * f if s > warm else a.lr * f
                g["betas"] = (0.9, 0.999 if s < steps // 10 else 0.9999)
            idx = perm[b:b + a.batch]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            d = 0.99 + (0.9999 - 0.99) * min(1.0, s / max(1, steps * 0.8))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), params, 1 - d)
            if s % 2000 == 0 or s == steps:
                if not math.isfinite(float(loss)): raise RuntimeError("diverged")
                print(f"step {s}/{steps} loss {float(loss):.4g} lr {opt.param_groups[0]['lr']:.3g} {time.time() - t0:.0f} s", flush=True)
        del X, Y, perm
    write_bake(a.out, ema, config, s, {"trainer": "nrc/bake/stream/bake_stream.py", "records": s * a.batch,
                                       "dump": os.path.basename(os.path.normpath(a.dump))})
    print(f"wrote {a.out}: {s} steps, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
