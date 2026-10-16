---
id: e52
script: bench/experiments/e52_shed_compressed.py
change: yowlwvyo
date: 2026-10-16
---
# e52: letting go of arrays takes their compression off

> 中文摘要:在压力下放掉一半数组,进程被压缩的量几乎一比一地减少(45.1 → 22.6 GB),所以运行中收缩专家存储能卸掉压力。
> macOS 不区分 GPU 每秒都在读的数组和闲置的数组,两者被压缩的比例一样。GPU 直接生成的数组没有被压缩,CPU 写入再拷进 MLX 的会,
> 而专家正是后一种。

## Question

The lever a store that follows the pressure would pull: under pressure, does letting go of arrays take compression
off the process, and does it matter which arrays?

## Setup

M4 Max, 128 GB. A hold wired first, as much as the machine could still wire (92.7 GiB), then 45 GB of arrays made
as `ExpertStore._read` makes them: random bytes from the CPU, copied into MLX arrays, 10 MiB each. Half of them read
by the GPU every second, half left alone; after 150 s the idle half let go and MLX's cache emptied.

Two earlier runs made the arrays on the GPU (`mx.random`) and the hold after them: 15 and 30 GB short by the
arithmetic, and the process's compression stayed at 0.1 GB both times.

## Result

| when | footprint | compressed |
| :--- | ---: | ---: |
| the arrays made | 45.13 GB | 42.27 GB |
| before letting go | 45.27 | 45.06 |
| after letting go of half | 22.76 | 22.56 |

The hold resident throughout (92.75 GiB at its least).

## Conclusion

- Letting go takes the compression off with the arrays, nearly one for one: a store that sheds experts under
  pressure gives back what macOS was compressing.
- macOS compressed the half the GPU read every second as much as the idle half: the GPU's reads do not keep pages
  warm. So which experts a store lets go of does not matter to macOS; how much does. And the experts in use are
  compressed with the rest, which is what e48 saw.
- Arrays the GPU made were not compressed under 15-30 GB of shortfall; arrays the CPU wrote and MLX copied were
  compressed almost whole. The store's experts are of the second kind.
- MLX's arrays are not in a process's resident size (0.03-0.16 GB here against a footprint of 21-45 GB): a hold's
  resident part says what it says because the hold is numpy's, not MLX's.

## Concepts

[[expert-store]] [[swapping]] [[available-memory]]
