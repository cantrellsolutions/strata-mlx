---
id: ngram-table
kind: component
area: prompts
status: current
---
# The n-gram table

> 中文摘要:模型的第二个文件里有一张 28.8 GB 的 n-gram 表,每个 token 按它前面几个 token 的哈希取几行。
> 引擎把它留在盘上按需读取,表不在文件缓存里时每取一行都是一次缺页,写回答和读提示都会被拖慢。

## What it is

A table of per-layer embeddings looked up by hashes of the tokens before each one: 28.8 GB, the second shard of
every size of the model. Its rows land anywhere in it.

## In this engine

`strata_mlx/load.py:PleTable`: mapped from the file, not loaded; its rows read ahead by threads for a prompt
(`_ahead`), and the whole table read into the page cache after loading where memory allows (`warm`).

## Relations

- depends-on → [[page-cache]]: read by page faults; cached on a machine with memory to spare, cold on a small one [uoonprvo] [#10]
- costs → [[decode-pass]]: cold, a step waited 4.0 ms on its page faults, 1.2 with the table cached; an answer came 16% faster on an M5 Ultra [#10] [tousqpoo]

## What we know

- A prompt of 8,192 tokens with the table cold took 13-14 s more on the M4 Max, the same for every way of
  reading the experts; reading its rows' pages ahead with 32 threads cut a chunk's lookup from 6.3 s to 0.22
  [qrlqtqpv].
- Benchmarks that drop only the first shard's cache leave this table cached, which made earlier prompt figures
  look faster than a cold machine's [ROADMAP: A prompt read a layer at a time].

## Open questions

- How much a first prompt that arrives while the table is still being warmed competes with it [#10].
