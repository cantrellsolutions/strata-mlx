---
id: layer-major-prefill
kind: mechanism
status: current
---
# A prompt read a layer at a time

> 中文摘要:专家从 SSD 读的机器上,长于一块(2,048 token)的提示按 8,192 token 一段,一段先走完第 1 层的所有块,
> 再走第 2 层,每层的专家只读一次。结果与逐块读逐位相同。真实 32 GB 机器上 8K 提示 45.7 秒到 31.5 秒,64 GB 机器上快 1.98 倍。

## What it is

Where experts are read from the SSD a prompt was read in chunks of 2,048 tokens, each through all the layers,
reading each layer's experts again for every chunk. Read a layer at a time, a piece of 8,192 tokens goes through
layer 1 chunk by chunk, then layer 2, and a layer's experts are read once and kept until its last chunk. What a
chunk computes at a layer is not changed, so the state is the same bit for bit.

## In this engine

`strata_mlx/bylayer.py:read`, `strata_mlx/bylayer.py:PIECE`; `strata_mlx/paged.py:hold`; the server's
`--prompt-by-layer off`.

## Relations

- depends-on → [[expert-store]]: only where experts are read from the SSD, and more than a chunk is to be read [utpukvqp]
- depends-on → [[exactness]]: the order of the loops changes and no sum's order does; Strata's #1670, which groups by expert, is not bit for bit and was not taken [e40] [ROADMAP: A prompt read a layer at a time]
- costs → [[working-memory]]: a piece peaks 0.9-1.1 GB above a chunk [e40]
- limited-by → [[ssd-reads]]: chunk by chunk an 8,192-token prompt read 99-100 GB, 70 of it read before; a layer at a time 26.7 GB [e37] [e40]

## What we know

| where | 8,192 tokens, chunk by chunk | a layer at a time | and read ahead |
| :--- | ---: | ---: | ---: |
| M4 Max as a 32 GB machine, both shards dropped, n-gram rows read ahead | 35.4 s | 28.6 | 25.6 |
| a real 32 GB MacBook Air M5 [#6] | 45.7 | 34.5 | 31.5 |
| a real 64 GB M5 Pro, IQ3_S [#8] | 37.85 | 24.9 | 19.1 |

- A stop is seen after every step (0.26-0.30 s, once 10.4 s unexplained) where chunk by chunk it was seen once a
  chunk; the piece under way is undone exactly [e40].
- With the n-gram table cold the rule set before building (1.5 times, files dropped) was met on the 64 GB M5 Pro
  and not on the M4 Max (1.30-1.38) [uoonprvo] [#8].
