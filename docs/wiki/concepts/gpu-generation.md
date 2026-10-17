---
id: gpu-generation
kind: hardware
area: across
status: current
---
# The GPU's generation

> 中文摘要:不同代、不同核心数的苹果 GPU 上,内核的最佳参数不一样。在 M5 Ultra 上调好的阈值,在 M4 Max 上可能更慢,
> 反过来也一样。所以凡是"调出来的"阈值,都要注明在哪台机器上调的。

## What it is

The chip a pass runs on: its generation (M4, M5), its GPU cores (40 on an M4 Max, 20 on an M5 Pro, 80 on an M5 Ultra), and what
the generation adds (M5's neural accelerators in every GPU core). Settings tuned on one are not settled for another.

## In this engine

Thresholds tuned on one machine: `strata_mlx/engine.py:FUSE_UPTO` (set by hand with `--fusion-limit`,
`strata_mlx/serve.py:fusion_limit`), `strata_mlx/fused.py:WIDE_FROM`, `strata_mlx/blocks.py:lookup_matvec`'s
threads to a row, `strata_mlx/blocks.py:MATVEC_GLU_UPTO`.

## Relations

(Its relations are the ones that point at it: tuned-by from [[hyper-connections]], [[expert-lookup-kernel]],
[[kernel-fixed-cost]], [[draft-window]]; depends-on from [[exactness]].)

## What we know

- MLX names the GPU's architecture (`mx.device_info()["architecture"]`): `applegpu_g16s` on the M4 Max, generation
  16; an M5 is generation 17, which is how MLX itself tells where the neural accelerators are. The hyper-connections'
  fusion limit is chosen by it: 8 from 17, 6 below [#9] [uvunqsoq].
- The generation alone does not choose the fusion limit: on a 20-core M5 Pro (`applegpu_g17s`, 64 GB) fused up to
  eight was the slower, by 1.3-1.7 ms a pass at seven and eight tokens with Q2_0 and 1.1-1.2 with IQ3_S, against a
  control engine; measured by pull request 11's author [e54] [#11]. `--fusion-limit 6` sets it by hand; the
  engine's own choice is unchanged.

- Two thresholds moved between an M5 Ultra and this M4 Max: hyper-connections fused up to eight tokens (faster
  there, slower here at seven and eight) and the table kernel's threads to a row (slower there in a full pass,
  faster here) [e44] [e45].
- MLX on an M5 runs float32 products at reduced precision by default [#5].

## Open questions

- The other thresholds tuned on one machine (`WIDE_FROM`, `MATVEC_GLU_UPTO`) are not chosen by GPU; the threads
  to a row of the table kernel were measured on both and hold on both [#9].
- Generation 17 is taken from MLX's naming; an M5 Pro calls itself `applegpu_g17s` [#11], the M5 Ultra's own string
  has not been seen here.
- What does choose the fusion limit on an M5: the GPU's cores (20 on the M5 Pro, faster at 6; 80 on the Ultra,
  faster at 8)? The letter after the generation may not tell them apart, the M4 Max being `g16s` too, and
  `mx.device_info()` does not give the cores [e54].
