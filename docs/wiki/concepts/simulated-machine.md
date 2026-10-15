---
id: simulated-machine
kind: method
area: memory
status: current
---
# A smaller machine, simulated

> 中文摘要:在 128 GB 的机器上用 `--machine-memory` 按小内存机器的规则限制专家预算来测。它能测读盘和计算,
> 测不到内存压力和换页;同时必须先把模型文件(两个分片)清出文件缓存,否则会测得过快。

## What it is

Running as a machine of less memory would: the engine works out its budget from `--machine-memory GiB` instead of
the machine's own, and the files are dropped from the page cache first. It is how every small-machine figure
here was measured until users ran the engine on theirs.

## In this engine

`--machine-memory` in `strata_mlx/serve.py`, `bench/paged.py`, `bench/prompt_by_layer.py`; `bench/drop_cache.py`.
`bench/paged.py --hold auto` makes the machine smaller in fact: another process holds what the machine has beyond
`--machine-memory` in random bytes (`bench/memory.py:Hold`), and the run is valid only if the hold stayed whole.

## Relations

- limited-by → [[page-cache]]: the files must be dropped from it first, both shards; a run that dropped only the experts' shard left the n-gram table cached [e37] [uoonprvo]

## What we know

- What a simulated run measures holds where memory is not short: the disk, the kernels, how many experts are read.
- What it cannot measure by itself: what macOS and other programs hold, pressure, swapping [#6]. Held memory is
  meant to close that: TensorFold's harness does the same and rejects a run whose hold died. As a 96 GB machine
  with 32 GiB held, the hold stayed whole through the run (50 samples) and nothing swapped [wumtsnom].
- Real machines heard from so far: a 32 GB MacBook Air M5 [#6], a 48 GB M4 Pro [xukwosuo], a 64 GB M5 Pro [#8].

## Open questions

- A run as a 32 GB machine with 96 GiB held, which is what would have shown issue 6's swapping here [wumtsnom].
