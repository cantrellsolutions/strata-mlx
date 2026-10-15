---
id: bandwidth-floor
kind: constraint
status: current
---
# The memory bandwidth floor

> 中文摘要:一次前向至少要把它用到的权重从内存读一遍,读的时间就是下限。Q2_0 每个 token 读 5.5 GB,
> 在 M4 Max 上单纯把这些数组读一遍要 13.9 ms,而一次前向是 20 ms 左右,差的部分是内核开销和等待。

## What it is

The least time a pass can take: the bytes of weight it reads over what memory gives. Answers are bound by it
because a token's products are matrix-vector, each weight read once and used for one or a few tokens.

## In this engine

Nothing to point at: it is what every product in a pass adds up to.

## Relations

(None measured yet: see the open questions.)

## What we know

- A token of Q2_0 reads 5.5 GB: dense tensors 4.87 (hyper-connections 1.27, delta-net projections 2.1, attention
  0.6, output layer 0.5) and experts 0.66. Summing those arrays once takes 13.9 ms on the M4 Max [ROADMAP: Faster answers].
- A window of several tokens reads the dense weights once, so its floor rises only with the experts its tokens
  add; that is why a window is worth its cost ([[draft-window]]).

## Open questions

- The floor is the chip's memory bandwidth, so it should move with [[gpu-generation]] (Apple gives 546 GB/s for
  an M4 Max and 614 for an M5 Max); not measured here.

- How close MLX's own products come to the floor for each shape at one to eight tokens (some are flat, the
  largest 8-bit one is not).
