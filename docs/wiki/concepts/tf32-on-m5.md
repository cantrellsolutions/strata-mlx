---
id: tf32-on-m5
kind: hardware
area: precision
status: current
---
# Reduced-precision float32 on an M5

> 中文摘要:MLX 从 0.30 起在带 Neural Accelerators 的芯片(M5)上默认用降精度做 float32 矩阵乘,相对误差约 7e-4,
> 27 个测试因此失败。设 `MLX_ENABLE_TF32=0` 才恢复精确;它也让 float16 的乘法慢很多,所以只在需要精确的地方设。

## What it is

Since MLX 0.30, on chips with neural accelerators in the GPU, float32 matrix products run at reduced precision
unless `MLX_ENABLE_TF32=0` is set before the first computation. Upstream calls it expected behaviour.

## In this engine

`tests/conftest.py` sets it for the tests; `strata_mlx/serve.py` sets it for `--dtype float32`; the benchmarks
that ask for float32 set it too.

## Relations

- invalidates → [[exactness]]: on an M5 a float32 product was off by 7.3e-4 relative, where the M4 Max gives 5.3e-7; 27 tests failed until it was set [#5]
- depends-on → [[gpu-generation]]: the M5's neural accelerators; nothing changes on an M4 [#5]

## What we know

- With it set, all 325 tests passed on the reporter's M5 [#5]. It also made float16 products 13 times slower
  there, which the engine's default float16 path does not set it for [#5] [ozxszktq].
- Pull request 7 made the benchmarks that ask for float32 set it as well [omvllxto].

## Open questions

- A real model in float32 on an M5 has not been run.
