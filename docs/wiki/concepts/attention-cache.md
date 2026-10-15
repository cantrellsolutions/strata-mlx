---
id: attention-cache
kind: component
area: conversations
status: current
---
# The attention caches

> 中文摘要:注意力层为每个 token 存的键和值,每个 token 约 30 KB(128K token 约 3.9 GB),可以截断到任意位置。
> 超过 2,048 个位置后改为稀疏注意力,每次只取被选中的 2,052 个位置。内存规则没有把它算进去。

## What it is

The keys and values of the attention layers, 30 KB a token in 16-bit floats (3.9 GB at 128K), and the indexer's
pooled keys. They can be cut back to any token.

## In this engine

`strata_mlx/attention.py:AttnCache`, `strata_mlx/attention.py:IndexerCache`; past 2,048 cells a pass gathers the
2,052 each query chose (`strata_mlx/attention.py:GATHER_FROM` for chunks).

## Relations

- costs → [[memory-rule]]: not counted by it: 1 GB for 36,000 tokens, which a machine reading experts from the SSD has not been told of [ROADMAP: Done]
- part-of → [[decode-pass]]: with the gather a pass stays 27-29 ms from 8K to 128K cells; it was 26-45 ms when every cell was attended under a mask [ROADMAP: Done]

## What we know

- Against llama.cpp at 3,161, 6,945 and 32,299 tokens: its token at 64 of 64 steps [ROADMAP: Done].
- At 8 bits the caches would be about half; not done.
