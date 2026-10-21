---
id: e59
script: bench/experiments/e59_recurrence_kernels.py
change: rtvwnktp
date: 2026-10-21
---
# e59: the delta-net recurrence alone

> 中文摘要:把 delta-net 递归单独拿出来量:36 层、1 个 token 只要 1.23 ms(4 个 token 1.35 ms),不是 e58 推断的 2.7 ms。
> 递归内核本身每层约 32 us,mlx-lm 的 packed 和 unpacked 两个版本一样快、输出逐位相同;它读写 6.3 MB,
> 这么小的内核本来就到不了带宽(e56)。所以它能省的只有零点几毫秒。e58 里去掉它省的 2.7 ms,多出的部分应该是
> 它的输入随之不再被需要、被 MLX 一并删掉了:逐项去掉的办法会高估有上游计算的部分。

## Question

e58 left the recurrence out and saved 2.7 ms of a one-token pass, against a floor of 0.46 ms for its state. Is it
the kernel, the small kernels before it (g and beta), or something else?

## Setup

M4 Max 128 GB, mlx-lm's `gated_delta` as installed, change rtvwnktp. 36 layers of the model's shapes (16 key heads,
48 value heads of 128 x 128, float32 state), random inputs, chained (each layer's v waits on the one before).
Variants shuffled in one process, 150 rounds, lower quartile. Results `bench/results/e59_recurrence_kernels.json`.

## Result

ms for 36 layers (the chain's own glue, an add a layer, is in every row; "g, beta only" is about that):

| variant | 1 token | 4 tokens |
| :--- | ---: | ---: |
| gated_delta_update as the engine calls it | 1.23 | 1.35 |
| g and beta, then the unpacked kernel | 1.25 | 1.41 |
| the packed kernel alone | 1.18 | 1.26 |
| the unpacked kernel alone | 1.15 | 1.34 |
| g and beta alone | 0.75 | 0.77 |

The unpacked kernel gives the packed kernel's bits, at both sizes.

## Conclusion

- **The recurrence is about 1.2 ms of a pass, not 2.7.** Its kernel is about 32 us a layer for 6.3 MB read and
  written; e56 found any kernel reading a few MB takes about that. Little to win: a few tenths of a millisecond
  by fusing g and beta in, which would have to reproduce mlx-lm's float16 softplus to keep the bits.
- **e58's 2.7 ms is an overstatement**: with the recurrence's output replaced by zeros, the arrays feeding it are no
  longer wanted and MLX leaves out their computation too. Leaving a part out measures the part and whatever only
  it used; parts with work upstream read high.

## Concepts

[[decode-pass]]
