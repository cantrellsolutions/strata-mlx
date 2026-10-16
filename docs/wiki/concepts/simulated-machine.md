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
`--machine-memory` in random bytes (`bench/memory.py:Hold`), and the run is valid only if the hold stayed resident;
`bench/memory.py:own` counts this process's memory compressed or swapped out. `--hold quiet` holds less by what
the other programs hold beyond a quiet macOS (6.1 GB). The hold is wired: unwired it is
the coldest memory on the machine and is pushed out first [e47] [zzzlsupr].

## Relations

- limited-by → [[page-cache]]: the files must be dropped from it first, both shards; a run that dropped only the experts' shard left the n-gram table cached [e37] [uoonprvo]

## What we know

- What a simulated run measures holds where memory is not short: the disk, the kernels, how many experts are read.
- What it cannot measure by itself: what macOS and other programs hold, pressure, swapping [#6]. Held memory is
  meant to close that: TensorFold's harness does the same and rejects a run whose hold died. As a 96 GB machine
  with 32 GiB held, the hold stayed whole through the run (50 samples) and nothing swapped; as a 48 GB one with 80 GiB held, the store filled, answers came at 25-34 tok/s [e46] [wumtsnom].
- Real machines heard from so far: a 32 GB MacBook Air M5 [#6], a 48 GB M4 Pro [xukwosuo], a 64 GB M5 Pro [#8].

## Superseded

- A hold left unwired [tukzwlpx]: as a 32 GB machine 96 GiB were held and 8.9 stayed resident [e47].

- A hold judged whole by its footprint [wumtsnom]: the footprint counts compressed and swapped pages; it is judged
  by its resident part since [tukzwlpx].

## Open questions

- Runs under held memory one after another are not alike: what the next one finds available depends on what the
  last one pushed out [e51].

- With the memory wired the two ways of setting the budget can be compared on this machine: the rule's
  against what is available [e48] [e49].

- A quiet small machine: a wired hold leaves the other programs of this machine inside the simulated one. As a
  48 GB machine with about 38 GB of them open, the engine had 31 GB compressed or swapped out [e48]. Held as a
  quiet 32 GB machine it still had 11.9 GB compressed at once and wrote a quarter slower than the real Air with
  nothing open: active programs keep asking for memory, which no hold takes out [e50].

- Which process a system-wide swap-out came from: as a 48 GB machine with 80 GiB held, 25 GB were swapped
  out and the hold stayed whole [e46]. This process's own page-ins would say.

- A run as a 32 GB machine with 96 GiB held, which is what would have shown issue 6's swapping here [wumtsnom].
