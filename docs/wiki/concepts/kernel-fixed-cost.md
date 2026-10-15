---
id: kernel-fixed-cost
kind: constraint
status: current
---
# The fixed cost of a kernel

> 中文摘要:一次前向是几千个小内核,一个接一个地等。每个内核不管算多少都有几微秒的固定开销,
> 所以在 1 到 8 个 token 的窗口里,把小步骤合并成一个内核往往比优化某个大矩阵乘更有效。

## What it is

What a kernel costs beyond its work: launch, a wait for the one before it, the GPU's threads ramping up. In a
pass of a few tokens most kernels do little, so this is a large share of the pass.

## In this engine

The fused kernels of `strata_mlx/fused.py` and `strata_mlx/blocks.py`; `mx.compile` over a layer's halves in
`strata_mlx/engine.py:Engine`.

## Relations

- tuned-by → [[gpu-generation]]: on an M5 Ultra every kernel in the chain costs about 4-5 us at one token; which merges pay depends on the GPU [#9] [e45]

## What we know

- A pass is about 3,300 kernels in 229 command buffers; the GPU is busy 77% of it. Cutting the kernel count by 43%
  bought 15%, because the kernels easy to merge were the cheap ones [ROADMAP: Now: what is left].
- One GPU thread is slow and a kernel takes as long as its longest thread: a norm kernel of 128 threads looping
  160 times costs 21 us for nothing [e08] [e09].
- Pull request 9 merged ten groups of small kernels (router top-k, the end of the MoE block, the delta-net's
  convolution and norms, attention's norms and rope, and more): 1.6 ms less a pass at one token and 3.0 at four
  on the M4 Max [e45] [wykuvxvr].

## Superseded

- "A pass is thousands of small kernels that cost by their number": the bytes, not the count, were most of it
  then [ROADMAP: Faster answers]. Both hold now: the count matters at the margin once the products are near the floor.
