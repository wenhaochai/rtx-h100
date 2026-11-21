"""Average the targets of grouped training records, for the nrc-scatter caches (nrc/bake/dump_scatter.sh).

A dump of a -DNRC_GROUP_N=N bake cubin (patches/bake/scatter.diff) carries each record's group key (bit 31 set, pixel,
sample / N) in its padding word (features 14-15). Records with the same key and the same position and direction
(features 0-4) come from the N paths of one camera sample that stashed the same vertex: their targets are independent
estimates of the radiance scattered there (one BSDF sample each), so their mean is an unbiased target with about 1/N of
the variance. The other features (normal, roughness, albedos) may differ inside a set where the shader is stochastic
(e.g. a Bevel or AO node): the set's record takes the features of a member drawn uniformly, independent of the targets,
so the cache learns the mean over the shading randomness at the vertex, which the query's expectation needs. Each set
of m <= N records becomes one record (target = the mean), kept with probability m / N so that every path keeps weight
1/N in expectation (full sets, m = N, always stay; partial sets come from slot misses or from paths whose stash vertex
differs). The kept records are shuffled and written in the dump format, 2^20 per call; features 14-15 hold m as fp16
(unused by training).

  group_avg.py <raw.rec> <out.rec> --n 8 [--seed 0] [--stats out.json]

Stats: raw and kept counts, the histogram of m, sets whose other features vary (varying), the target noise (mean
within-set variance of the luminance over full sets, per raw record), the luminance sum of the raw targets and the
Horvitz-Thompson sum N x sum lum(mean) over kept records (equal in expectation).
"""
import argparse, json, os, sys, time

import numpy as np
import torch

REC = 44
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)


def calls_of(path):
    out, off, sz = [], 0, os.path.getsize(path)
    with open(path, "rb") as f:
        while off < sz:
            f.seek(off)
            n = int(np.frombuffer(f.read(4), np.uint32)[0])
            out.append((off + 4, n))
            off += 4 + n * REC
    assert off == sz, f"truncated dump {path}"
    return out


def read_all(path):
    calls = calls_of(path)
    n = sum(c for _, c in calls)
    X = np.empty((n, 8), np.uint32); Y = np.empty((n, 3), np.float32)
    i = 0
    with open(path, "rb") as f:
        for off, c in calls:
            f.seek(off)
            raw = np.frombuffer(f.read(c * REC), np.uint8)
            X[i:i + c] = raw[:c * 32].view(np.uint32).reshape(c, 8)
            Y[i:i + c] = raw[c * 32:].view(np.float32).reshape(c, 3)
            i += c
    return X, Y


def mix(words, prime, nw=7, chunk=1 << 26):
    """64-bit FNV-style hash of words 0..nw-1 of each row (int64 wrap-around; nw = 2.5: words 0, 1 and the low half of
    word 2 = position and direction), on the GPU in chunks."""
    out = torch.empty(words.shape[0], dtype=torch.int64, device=DEV)
    for i in range(0, words.shape[0], chunk):
        w = torch.from_numpy(words[i:i + chunk, :7].view(np.int32)).to(DEV).to(torch.int64) & 0xffffffff
        if nw == 2.5: w = torch.stack([w[:, 0], w[:, 1], w[:, 2] & 0xffff], 1)
        h = torch.full((w.shape[0],), -3750763034362895579, dtype=torch.int64, device=DEV)
        for j in range(w.shape[1]):
            h = (h ^ w[:, j]) * prime
        out[i:i + w.shape[0]] = h ^ (h >> 29)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    t0 = time.time()
    X, Y = read_all(a.raw)
    n = X.shape[0]
    t_read = time.time() - t0
    key = torch.from_numpy(X[:, 7].view(np.int32)).to(DEV).to(torch.int64) & 0xffffffff
    flagged = int((key >> 31).sum())
    h1 = mix(X, 1099511628211, nw=2.5)
    comp = ((key & 0x7fffffff) << 32) | (h1 & 0xffffffff)
    del key, h1
    g = torch.Generator(device=DEV); g.manual_seed(a.seed)
    r = torch.randperm(n, generator=g, device=DEV)     # a stable sort of a random order: each set's first is uniform
    vals, perm = torch.sort(comp[r], stable=True)
    perm = r[perm]
    del comp, r
    new = torch.ones(n, dtype=torch.bool, device=DEV); new[1:] = vals[1:] != vals[:-1]
    del vals
    gid = torch.cumsum(new.to(torch.int64), 0) - 1
    first = torch.nonzero(new).squeeze(1)          # sorted position of each set's first record
    del new
    G = first.shape[0]
    m = torch.bincount(gid, minlength=G)
    h2 = mix(X, 1469598103934665603 | 1)       # all 14 features: do they vary inside a set?
    h2s = h2[perm]; del h2
    diff = (h2s != h2s[first][gid])
    varying = int(torch.zeros(G, dtype=torch.int64, device=DEV).index_add_(0, gid, diff.to(torch.int64)).gt(0).sum())
    del h2s, diff
    Ys = torch.from_numpy(Y).to(DEV)
    Ys = Ys[perm]                                   # f32: a set has at most N records
    lum_raw = (Ys @ W.float().to(DEV)).double()
    S = torch.zeros(G, 3, dtype=torch.float32, device=DEV).index_add_(0, gid, Ys)
    del Ys
    mean = S.double() / m[:, None].double()
    lm = mean @ W.to(DEV)
    full = m == a.n                                 # target noise: within-set variance of lum over full sets
    ss = torch.zeros(G, dtype=torch.float64, device=DEV).index_add_(0, gid, (lum_raw - lm[gid]) ** 2)
    noise = float((ss[full] / (a.n - 1)).mean()) if bool(full.any()) else float("nan")
    del gid, ss
    keep = torch.rand(G, generator=g, device=DEV, dtype=torch.float64) * a.n < m.double()
    assert int(m.max()) <= a.n or flagged == 0, f"a set has {int(m.max())} > N records"
    kidx = torch.nonzero(keep).squeeze(1)
    kidx = kidx[torch.randperm(kidx.shape[0], generator=g, device=DEV)]
    src = perm[first[kidx]].cpu().numpy()
    Xo = X[src].copy()
    mk = m[kidx].cpu().numpy()
    Xo[:, 7] = mk.astype(np.float16).view(np.uint16).astype(np.uint32)
    Yo = mean[kidx].float().cpu().numpy()
    lum_mean = mean @ W.to(DEV)
    st = dict(raw=a.raw, n=a.n, records_raw=n, flagged=flagged, sets=G, kept=int(kidx.shape[0]),
              m_hist={int(k + 1): int(v) for k, v in enumerate(torch.bincount(m, minlength=a.n + 1)[1:].tolist())},
              varying=varying, noise_var_raw=noise, noise_var_mean=noise / a.n, sum_lum_raw=float(lum_raw.sum()), sum_lum_ht=float(a.n * lum_mean[kidx].sum()),
              sum_lum_sets=float((lum_mean * m.double()).sum()),
              max_lum_raw=float(lum_raw.max()), max_lum_kept=float(lum_mean[kidx].max()),
              var_lum_raw=float(lum_raw.var()), var_lum_kept=float(lum_mean[kidx].var()), t_read=round(t_read, 1))
    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        for i in range(0, Xo.shape[0], 1 << 20):
            c = min(1 << 20, Xo.shape[0] - i)
            f.write(np.uint32(c).tobytes()); f.write(Xo[i:i + c].tobytes()); f.write(Yo[i:i + c].tobytes())
    os.replace(tmp, a.out)
    st["t_total"] = round(time.time() - t0, 1)
    print(json.dumps(st))
    if a.stats:
        json.dump(st, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
