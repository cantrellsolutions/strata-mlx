---
id: gpu-generation
kind: hardware
status: current
---
# The GPU's generation

> 中文摘要:不同代、不同核心数的苹果 GPU 上,内核的最佳参数不一样。在 M5 Ultra 上调好的阈值,在 M4 Max 上可能更慢,
> 反过来也一样。所以凡是"调出来的"阈值,都要注明在哪台机器上调的。

## What it is

The chip a pass runs on: its generation (M4, M5), its GPU cores (40 on an M4 Max, 80 on an M5 Ultra), and what
the generation adds (M5's neural accelerators in every GPU core). Settings tuned on one are not settled for another.

## In this engine

Thresholds tuned on one machine: `strata_mlx/engine.py:FUSE_UPTO`, `strata_mlx/fused.py:WIDE_FROM`,
`strata_mlx/blocks.py:lookup_matvec`'s threads to a row, `strata_mlx/blocks.py:MATVEC_GLU_UPTO`.

## Relations

(Its relations are the ones that point at it: tuned-by from [[hyper-connections]], [[expert-lookup-kernel]],
[[kernel-fixed-cost]], [[draft-window]]; depends-on from [[exactness]].)

## What we know

- Two thresholds moved between an M5 Ultra and this M4 Max: hyper-connections fused up to eight tokens (faster
  there, slower here at seven and eight) and the table kernel's threads to a row (slower there in a full pass,
  faster here) [e44] [e45].
- MLX on an M5 runs float32 products at reduced precision by default [#5].

## Open questions

- Choosing such thresholds by GPU at start. It needs each one measured on both machines the same way.
