"""The shape a prompt gives a layer's experts: 512 experts, each with its own group of tokens, as one kernel of
Apple's matmul primitive (weights packed at 2 bits, or as half), against what MLX offers for it."""
import sys, time
import mlx.core as mx, numpy as np
E, G, K, N = 512, int(sys.argv[1]) if len(sys.argv) > 1 else 64, 2560, 640   # G tokens to every expert
HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
def kernel(b_t, tm):
    ptr = "uchar" if "format" in b_t else b_t
    src = f"""
    uint tx = threadgroup_position_in_grid.x, ty = threadgroup_position_in_grid.y;   // ty: a tile of {tm} tokens
    uint e = ty / ({G} / {tm});                                                         // their expert
    constexpr auto desc = matmul2d_descriptor({tm}, 32, {K});
    matmul2d<desc, execution_simdgroups<4>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A((device half *)a, dextents<int32_t, 2>({K}, {E * G}));
    tensor<device {b_t}, dextents<int32_t, 2>, tensor_inline> B((device {ptr} *)b, dextents<int32_t, 2>({N}, {E * K}));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>({N}, {E * G}));
    auto tA = A.slice<{K}, {tm}>(0, ty * {tm});
    auto tB = B.slice<32, {K}>(tx * 32, e * {K});
    auto tC = C.slice<32, {tm}>(tx * 32, ty * {tm});
    op.run(tA, tB, tC);
"""
    k = mx.fast.metal_kernel(name=f"mpp_experts_{b_t.split('_')[0]}_{tm}", input_names=["a", "b"], output_names=["c"], source=src, header=HEADER)
    return lambda a, b: k(inputs=[a, b], grid=(128 * (N // 32), E * G // tm, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(E * G, N)], output_dtypes=[mx.float16], init_value=0)[0]
def best(fn, n=5):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(fn()); t = min(t, time.perf_counter() - t0)
    return t
rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((E * G, K)).astype(np.float16))
codes = rng.integers(0, 4, (E, K, N))
packed = mx.array(sum(codes.reshape(E * K, N // 4, 4)[:, :, j].astype(np.uint8) << (2 * j) for j in range(4)).astype(np.uint8))
w = mx.array(codes.astype(np.float16).reshape(E * K, N))
mx.eval(x, packed, w)
ops = 2 * E * G * K * N
tm = int(sys.argv[2]) if len(sys.argv) > 2 else 64
print(f"{E} experts x {G} tokens each ({E * G // 10} tokens of prompt), {K} -> {N}: {ops / 1e9:.0f} G ops")
if len(sys.argv) > 3:   # one variant per process: a kernel that does not build ends the process
    which = sys.argv[3]
    fn = kernel("uint2b_format" if which == "2bit" else "half", tm)
    B = packed if which == "2bit" else w
    got = fn(x, B)
    want = (x.reshape(E, G, K) @ w.reshape(E, K, N)).reshape(E * G, N)
    err = mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() / mx.abs(want).max().item()
    t = best(lambda: fn(x, B))
    print(f"  one kernel, weights {'packed at 2 bits' if which == '2bit' else 'as half':18} tiles of {tm} tokens: {t * 1e3:6.1f} ms, {ops / t / 1e12:5.2f} T ops/s (differs from MLX by {err:.0e})")
else:
    idx = mx.array(np.repeat(np.arange(E), G))
    w3 = w.reshape(E, K, N)
    t = best(lambda: mx.gather_mm(x[:, None, :], w3, rhs_indices=idx, sorted_indices=True))
    print(f"  MLX gather_mm over half weights, sorted:           {t * 1e3:6.1f} ms, {ops / t / 1e12:5.2f} T ops/s")
    t = best(lambda: x.reshape(E, G, K) @ w3)
    print(f"  MLX batched matmul over half weights:              {t * 1e3:6.1f} ms, {ops / t / 1e12:5.2f} T ops/s")
