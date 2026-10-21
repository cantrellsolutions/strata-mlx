---
id: e56
script: bench/experiments/e56_expert_kernel_variants.py
change: sksmuwnv
date: 2026-10-21
---
# e56: what the experts' kernel waits on

> 中文摘要:把专家的 gate 乘法拆成几个变体,48 层串起来对比。结论:它不受运算限制,也不受发散查表限制;
> 1 个 token 时,"只读字节、不计算"就已经要现有内核 86-89% 的时间,所有读法都在带宽下限的 2.5 倍左右,是小内核读几 MB 的爬坡和延迟;
> 4-8 个 token 时现有内核是下限的 1.4-1.45 倍,权重换成合并读取的平面布局后只读能贴着下限(1.0 倍),配上现有的查表是 1.28-1.36 倍,
> 比现在快 7-9%。所以"换成矩阵单元的新内核能快 2-3 倍"不成立;能动的是 1 个 token 时的内核个数(合并 gate 和 up)和平面布局。

## Question

The experts' table kernel ([[expert-lookup-kernel]]) is 2-3 times the bandwidth floor in a pass. Is that its
arithmetic, its divergent lookups into the table, the way it reads the weights' bytes, or what any kernel of
that size costs? (The knowledge base said arithmetic; e55's analysis with the other agent doubted it.)

## Setup

M4 Max 128 GB, change sksmuwnv; the GPU 24-67% busy with other programs. The gate product of a layer's routed
experts, 512 experts of 640 x 2,560 Q2_0, random weights and routes (as e43), 10 / 40 / 80 pairs (1 / 4 / 8
tokens). 48 layers of their own weights chained, each layer waiting on the one before; a chain of the glue alone
is subtracted. Variants shuffled in one process, 30 rounds, the lower quartile. Results
`bench/results/e56_expert_kernel_variants.json`.

## Result

Microseconds a layer, less the glue (8-9 us); in brackets, times the floor of the bytes at 490 GB/s [e55]:

| variant | 1 token (floor 9.4) | 4 tokens (36.3) | 8 tokens (69.5) |
| :--- | ---: | ---: | ---: |
| the engine's lookup | 26.8 (2.85) | 52.7 (1.45) | 97.4 (1.40) |
| the same grid reading the bytes only | 23.9 (2.54) | 44.3 (1.22) | 91.8 (1.32) |
| the lookup with its table index uniform | 29.3 (3.12) | 51.6 (1.42) | 100.4 (1.44) |
| codes and scales as planes, read side by side, bytes only | 23.4 (2.49) | 37.0 (1.02) | 69.9 (1.01) |
| planes, decoded through a 2 KB table of codes | 31.3 (3.33) | 74.8 (2.06) | 138.7 (2.00) |
| the same, activations in threadgroup memory | 32.2 (3.43) | 72.4 (1.99) | 135.8 (1.95) |
| planes read side by side, the engine's table of activations | 26.8 (2.85) | 49.4 (1.36) | 89.2 (1.28) |

8 threads to a row for the plane variants (4 and 16 were slower). The planes with the engine's table give the
lookup's result within 2-4e-4 of the largest value: the sums are the same, added in another order.

## Conclusion

- **Not arithmetic, not divergent lookups.** Reading the bytes and doing nothing is 86-94% of the kernel at every
  size; a uniform table index changes nothing. The "bound by arithmetic" of [e02]-[e12] is superseded.
- **At one token every reader is 2.5 times the floor**, the bytes alone included: a kernel reading 4.6 MB does
  not reach the bandwidth (190 GB/s). What one token can win is fewer, larger kernels (gate and up read the same
  input and could be one), not a better product.
- **At four and eight tokens the way the bytes are read matters**: as planes, side by side, reading reaches the
  floor; with the engine's table on top the product is 7-9% faster than now, a few tenths of a millisecond a pass
  for gate, about three times that for all three products. Not the 12-16 ms the analysis hoped for.
- **A matrix-unit kernel is not wanted**: nothing here is limited by multiply-adds, and decoding codes in
  registers costs more than the table does.
- Gate only; down, whose table is built in its own threadgroup, and up are not measured here.

## Concepts

[[expert-lookup-kernel]]
