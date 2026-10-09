"""Apple's shader-level matmul over the operand types it lists: right, and how fast, on this GPU."""
import sys, time
import mlx.core as mx, numpy as np

HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
PTR = {"int2b_format": "uchar", "uint2b_format": "uchar", "int4b_format": "uchar", "uint4b_format": "uchar"}
MX = {"half": mx.float16, "float": mx.float32, "int8_t": mx.int8, "uint8_t": mx.uint8, "int32_t": mx.int32}

def kernel(a_t, b_t, c_t, M, N, K, sg=4, tm=64, tn=32):
    src = f"""
    uint tx = threadgroup_position_in_grid.x, ty = threadgroup_position_in_grid.y;
    constexpr auto desc = matmul2d_descriptor({tm}, {tn}, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<{sg}>> op;
    tensor<device {a_t}, dextents<int32_t, 2>, tensor_inline> A((device {PTR.get(a_t, a_t)} *)a, dextents<int32_t, 2>(K, M));
    tensor<device {b_t}, dextents<int32_t, 2>, tensor_inline> B((device {PTR.get(b_t, b_t)} *)b, dextents<int32_t, 2>(N, K));
    tensor<device {c_t}, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
    auto tA = A.slice(0, ty * {tm}); auto tB = B.slice(tx * {tn}, 0); auto tC = C.slice(tx * {tn}, ty * {tm});
    op.run(tA, tB, tC);
"""
    k = mx.fast.metal_kernel(name="mpp_" + "_".join(t.split("_")[0] for t in (a_t, b_t, c_t)), input_names=["a", "b"], output_names=["c"],
                             source=src, header=HEADER + f"constant int M = {M}; constant int N = {N}; constant int K = {K};\n")
    width = 32 * sg
    return lambda a, b: k(inputs=[a, b], grid=(width * (N // tn), M // tm, 1), threadgroup=(width, 1, 1),
                          output_shapes=[(M, N)], output_dtypes=[MX[c_t]], init_value=0)[0]

def best(fn, n=5):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(fn()); t = min(t, time.perf_counter() - t0)
    return t

def pack(codes, bits):  # (K, N) small unsigned codes -> bytes, low bits first along the row
    per = 8 // bits
    c = codes.reshape(codes.shape[0], -1, per).astype(np.uint8)
    return sum(c[:, :, j] << (bits * j) for j in range(per)).astype(np.uint8)

M, N, K = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (2048, 2048, 2048)
rng = np.random.default_rng(0)
ops = 2 * M * N * K
ah = rng.standard_normal((M, K)).astype(np.float16); bh = rng.standard_normal((K, N)).astype(np.float16)
a8 = rng.integers(-127, 128, (M, K), dtype=np.int8); b8 = rng.integers(-127, 128, (K, N), dtype=np.int8)
c2 = rng.integers(0, 4, (K, N)); c4 = rng.integers(0, 16, (K, N))
s2 = np.where(c2 > 1, c2 - 4, c2); s4 = np.where(c4 > 7, c4 - 16, c4)   # the same bits read as signed
cases = [
    ("half", "half", "half", ah, bh, bh, None),
    ("half", "half", "float", ah, bh, bh, None),
    ("float", "float", "float", ah.astype(np.float32), bh.astype(np.float32), bh.astype(np.float32), None),
    ("int8_t", "int8_t", "int32_t", a8, b8, b8, None),
    ("half", "int8_t", "half", ah, b8, b8, None),
    ("half", "uint4b_format", "half", ah, pack(c4, 4), c4, None),
    ("half", "int4b_format", "half", ah, pack(c4, 4), s4, None),
    ("half", "uint2b_format", "half", ah, pack(c2, 2), c2, None),
    ("half", "int2b_format", "half", ah, pack(c2, 2), s2, None),
    ("int8_t", "int4b_format", "int32_t", a8, pack(c4, 4), s4, None),
    ("int8_t", "int2b_format", "int32_t", a8, pack(c2, 2), s2, None),
]
only = int(sys.argv[4]) if len(sys.argv) > 4 else None
if only is None:
    import subprocess
    am, bm = mx.array(ah), mx.array(bh); mx.eval(am, bm)
    print(f"C ({M}x{N}) = A ({M}x{K}) @ B ({K}x{N}). MLX matmul: half {ops / best(lambda: am @ bm) / 1e12:.2f}, "
          f"float {ops / best(lambda: am.astype(mx.float32) @ bm.astype(mx.float32)) / 1e12:.2f} T ops/s", flush=True)
    for i, c in enumerate(cases):
        r = subprocess.run([sys.executable, __file__, str(M), str(N), str(K), str(i)], capture_output=True, text=True)
        line = r.stdout.strip() or next((l for l in r.stderr.splitlines() if "error" in l), "failed")[:170]
        print(f"{c[0]:8} x {c[1]:14} -> {c[2]:8}: {line}", flush=True)
else:
    a_t, b_t, c_t, a, b, b_values, _ = cases[only]
    fn = kernel(a_t, b_t, c_t, M, N, K)
    A, B = mx.array(a), mx.array(b); mx.eval(A, B)
    got = np.array(fn(A, B)).astype(np.float64)
    want = a.astype(np.float64) @ b_values.astype(np.float64)
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"{ops / best(lambda: fn(A, B)) / 1e12:5.2f} T ops/s; largest difference from numpy {err:.1e}")
