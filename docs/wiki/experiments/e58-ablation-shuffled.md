---
id: e58
script: bench/experiments/e58_ablation_shuffled.py
change: ruspomnz
date: 2026-10-21
---
# e58: what each part of a pass costs now, left out

> 中文摘要:e06 的"逐项去掉"重做一次,按 e57 的办法(各部分一个引擎、共享权重、单次前向随机交替、下四分位,带对照组)。
> 现在 1 个 token 的前向是 17.6 ms。最大的是 delta-net 投影 5.4 ms(按内存里的 8 位是下限的 1.25 倍,文件里只有一半字节);
> 第一次单独量到的 delta-net 递归 2.7 ms,它几乎不读权重(状态 226 MB 读写,下限约 0.5 ms),是新发现的最大"超出";
> 注意力 2.1、路由专家 2.1(4 个 token 7.6)、输出层 1.0、共享专家 0.5、路由器 0.2。去掉之后剩下约 3.5 ms 是超连接和零碎。

## Question

After e57, where does the one-token pass's time over the 10.8 ms floor [e55] go? e06's ablation is from before
pull request 9 and was taken in turns, not shuffled.

## Setup

M4 Max 128 GB, Q2_0 whole in memory, change ruspomnz (submitting every 6 layers, gate and up in one kernel). An engine
for each part left out, sharing the weights, its stand-in (no weights read) in place while it was traced and
during its passes; a 300-token conversation first; every (engine, size) pass shuffled, 120 rounds, the lower
quartile; each pass committed. A control engine of the whole pass. Results `bench/results/e58_ablation_shuffled.json`.

## Result

| left out | 1 token, ms less | 4 tokens, ms less | bytes a token, and their time at 490 GB/s |
| :--- | ---: | ---: | :--- |
| nothing (the pass) | 17.58 | 28.81 | 5.5 GB, 10.8 ms |
| control, the same | 0.02 | -0.01 | |
| delta-net projections | 5.38 | 6.99 | 2.1 GB as held (8-bit), 4.3 ms; about 1.0 GB in the file |
| delta-net recurrence (`gated_delta_update`) | 2.71 | 3.06 | no weights; a state of 3.1 MB a layer read and written, 0.46 ms |
| attention, all of it | 2.11 | 3.77 | 0.6 GB, 1.2 ms |
| routed experts | 2.12 | 7.61 | 0.66 GB, 1.35 ms |
| output layer | 1.02 | 1.96 | 0.5 GB, 1.0 ms |
| shared experts | 0.54 | 0.85 | |
| routers | 0.24 | 0.56 | |

What is left out adds to 14.1 ms of the 17.6; the rest (hyper-connections, 1.27 GB and 2.6 ms of floor, and the
glue) about 3.5.

## Conclusion

- **The delta-net recurrence is 2.7 ms and reads no weights**: about 2.2 ms over what its state takes to read and
  write. Never measured apart before; the largest excess of any part. (Its stand-in also left out the small
  kernels before it, g and beta.)
- **The delta-net projections are 5.4 ms, near the floor of their bytes as held**, but held at 8 bits where the
  file is 3-4: about 2-2.5 ms of bytes that are inflation (e36).
- Routed experts at one token are 2.1 ms, 1.6 times their floor; at four 7.6 ms. The output layer is at its floor.
- So the 3 ms looked for are mostly these two: the recurrence's excess and the projections' inflation.

## Concepts

[[decode-pass]]
