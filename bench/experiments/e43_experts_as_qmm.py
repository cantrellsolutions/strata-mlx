"""Experiment 43: Q2_0 experts through MLX's gather_qmm (codes as they are, scale d, bias -d) against the table-of-sums kernel.

Slower: 4.62 / 7.56 / 16.46 / 47.75 ms against 4.30 / 5.57 / 11.14 / 30.62 for 1 / 2 / 4 / 8 tokens.

    uv run python bench/experiments/e43_experts_as_qmm.py <shard 1>
"""
import sys, time
import mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from strata_mlx.load import load
from strata_mlx import blocks

model, _ = load(sys.argv[1])
L = model.model.layers
glus = [l.mlp.switch_mlp for l in L if hasattr(getattr(l, "mlp", None), "switch_mlp")]
print("layers with experts", len(glus))

def as_q(lin):
    b = lin.blocks  # (E, rows, nblocks*18) uint8
    E, R = lin.num_experts, lin.output_dims
    b = b.reshape(E, R, -1, 18)
    d = b[..., :2].flatten().view(mx.float16).reshape(E, R, -1)
    w = mx.contiguous(b[..., 2:]).reshape(E, R, -1).view(mx.uint32)
    return w, d, -d

Q = []
for g in glus:
    Q.append(tuple(as_q(p) for p in (g.gate_proj, g.up_proj, g.down_proj)))
    mx.eval(Q[-1])
print("converted")

def qmm_glu(x, idx, q):
    (gw, gs, gb), (uw, us, ub), (dw, ds, db) = q
    x = mx.expand_dims(x, (-2, -3))
    kw = dict(transpose=True, group_size=64, bits=2)
    g = mx.gather_qmm(x, gw, gs, gb, rhs_indices=idx, **kw)
    u = mx.gather_qmm(x, uw, us, ub, rhs_indices=idx, **kw)
    h = (g * mx.sigmoid(g)) * u
    return mx.gather_qmm(h, dw, ds, db, rhs_indices=idx, **kw).squeeze(-2)

mx.random.seed(0)
for T in (1, 2, 4, 8):
    x = (mx.random.normal((T, 2560)) * 0.5).astype(mx.float16)
    idx = mx.random.randint(0, 512, (T, 10)).astype(mx.uint32)
    a = glus[0](x, idx); b = qmm_glu(x, idx, Q[0]); mx.eval(a, b)
    err = (mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max() / mx.abs(a.astype(mx.float32)).max()).item()
    res = {}
    for name, f in (("table", lambda i: glus[i](x, idx)), ("qmm", lambda i: qmm_glu(x, idx, Q[i]))) * 1:
        res[name] = []
    for r in range(30):
        for name, f in (("table", lambda i, x: glus[i](x, idx)), ("qmm", lambda i, x: qmm_glu(x, idx, Q[i]))):
            y = x
            t0 = time.perf_counter()
            for i in range(len(glus)):
                y = f(i, y).sum(-2) + x
            mx.eval(y)
            res[name].append((time.perf_counter() - t0) * 1e3)
    q = {k: sorted(v)[len(v) // 4] for k, v in res.items()}
    print(f"T={T}: table {q['table']:.2f} ms  qmm {q['qmm']:.2f} ms  (48 layers, lower quartile)  rel.err {err:.2e}")
