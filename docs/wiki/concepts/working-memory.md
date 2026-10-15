---
id: working-memory
kind: constraint
area: memory
status: current
---
# Working memory

> 中文摘要:进程在权重之外需要的内存:非数组的部分、读提示时的峰值、MLX 的缓冲池。规则里是 5 GB,
> 原来 4 GB,按层读提示之后峰值高了约 1 GB 才加上去。

## What it is

What the process holds beyond its weights: 1.3 GB that are not arrays, the peak of reading a prompt (a layer's
tensors, the activations), MLX's pool of freed buffers.

## In this engine

`strata_mlx/paged.py:WORKING`; MLX's pool is capped at 512 MB where experts are read from the SSD (`mx.set_cache_limit`).

## Relations

- costs → [[memory-rule]]: 5 GB that the experts do not get [utpukvqp]
- depends-on → [[layer-major-prefill]]: a piece of 8,192 tokens peaks 0.9-1.1 GB above a chunk, so the working memory went from 4 GB to 5 [e40] [utpukvqp]

## What we know

- Measured with Q2_0: 1.3 GB not arrays, 2.1 GB at the peak of reading a 1,500-token prompt, MLX's pool 0.5
  [ROADMAP: Macs with 16, 24 and 32 GB].
- Without a cap MLX's pool held 37 GB after 2,048 tokens and the machine swapped; the same speed at 1, 4 and 16 GiB [ROADMAP: Done].
- 32,768 tokens read a layer at a time peaked 5.00 GB above rest, against 3.72 chunk by chunk; most of either is
  the conversation's caches, which the rule does not count [e40].
