---
id: hyper-connections
kind: mechanism
area: answers
status: current
---
# Hyper-connections

> 中文摘要:这个模型每层不是一条残差流,而是四条,每层开头和结尾都有一组小矩阵把它们混合。权重是 16 位浮点,
> 共 1.27 GB,占每个 token 读取量的近四分之一。它在小窗口下由融合内核计算,融合到几个 token 为止要按 GPU 来定。

## What it is

The model keeps four residual streams instead of one. Before each half of a layer they are mixed into the
layer's input, and after it the output is written back into them, through small matrices whose weights are
float16: 1.27 GB in all, read every token.

## In this engine

`strata_mlx/fused.py:FusedResidual` (the fused kernels), `strata_mlx/fused.py:PlainResidual` (MLX's operations),
`strata_mlx/engine.py:FUSE_UPTO` (windows up to this many tokens take the fused ones), chosen by
`strata_mlx/engine.py:fuse_upto_for` from `strata_mlx/engine.py:gpu_generation`.

## Relations

- part-of → [[decode-pass]]: 3.7-5.1 ms of a one-token pass, 7.1-8.1 of a four-token one [e06]
- limited-by → [[kernel-fixed-cost]]: unfused it is a chain of small kernels; read once for a window and fused it costs less [e07] [e09] [#9]
- tuned-by → [[gpu-generation]]: fused up to eight tokens an M5 Ultra ran 2.0 ms a pass faster at seven and eight, with both files; on the M4 Max windows of seven and eight took 69.2 and 72.1 ms that way against 64.2 and 66.5 fused up to six; so the limit is 8 from GPU generation 17 and 6 below [e45] [#9] [uvunqsoq]
- trades-against → [[exactness]]: at 8 bits it would spare 0.6 GB a token, about 1.6 ms, and llama.cpp's token at 62 of 64 steps on a 7K prompt instead of 64; fused up to eight, windows of seven and eight give other logits than before [e18] [xtxomuun]

## What we know

- The fused kernels give a token the same bits alone or in a window, so they agree with one token at a time
  better than MLX's operations do at seven and eight tokens [#9].
- On the M4 Max the wide kernel for six to sixteen tokens made no difference [e45].

## Open questions

- 8-bit weights are not used: they change the answers. Whether a user should be allowed to choose them.
