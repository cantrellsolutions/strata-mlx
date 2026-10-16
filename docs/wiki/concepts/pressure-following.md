---
id: pressure-following
kind: mechanism
area: memory
status: current
---
# The budget following the pressure

> 中文摘要:运行中按内存压力调整专家份额:引擎进程被压缩超过 1 GB(连续两次读数)或系统压力达到"警告"时,
> 把份额降到"当前占用 − 被压缩的量 − 0.5 GB",每秒最多降 2 GB;压力消失 30 秒后每次涨回 1 GB,不超过启动时规则给的值。
> 默认关闭。e53:打开后不再换页、读提示更快,但份额降得过多,专家改从 SSD 读,写回答没有变快。

## What it is

The experts' budget moved while the engine runs, by what macOS is doing to the process: down when the process is
being compressed or the system reports memory pressure, back up after a quiet spell. It answers what a budget set
at start cannot see.

## In this engine

`strata_mlx/paged.py:Follow` (the rule, `decide`), `strata_mlx/paged.py:follow_pressure`,
`strata_mlx/paged.py:shrink_to`, `strata_mlx/paged.py:compressed_bytes`, `strata_mlx/paged.py:pressure_level`;
the server's `--follow-pressure on`, `bench/paged.py --follow-pressure`. Off by default.

## Relations

- depends-on → [[expert-store]]: it moves the store's budget and lets the least recently used experts go; which ones go does not matter to macOS, how many do [e52] [yytmkrlk]
- trades-against → [[swapping]]: experts let go are read from the disk again when wanted, in place of being decompressed [e52] [yytmkrlk]
- trades-against → [[ssd-reads]]: shed experts are read from the disk; it shed so far that reading took 34-66% of an answer [e53]
- depends-on → [[available-memory]]: the start-time bound sets where it starts; it is what follows after [e49] [e51]

## What we know

- The lever holds: under pressure, letting go of arrays took their compression off nearly one for one (45.1 to
  22.6 GB compressed for half the arrays let go) [e52].
- The signal is the process's own compression, not the pressure level alone: a quiet 32 GB machine had 11.9 GB
  of the engine compressed with nothing swapped [e50].
- Under pressure (a 48 GB machine wired, e48's condition) it took the engine's compression off at measuring time
  and the machine stopped swapping out (0-0.42 GB against 24.7-27.4), prompts read 2-4 times faster, but answers
  were not faster (9.5-12.2 tok/s medians on, 9.2-9.7 off): the budget fell to 3-7 GB and the experts came from
  the disk, 34-66% of the time [e53].
- Answers are the same bit for bit with the budget cut in the middle of one (`tests/test_engine.py`) [yytmkrlk].

## Open questions

- Shedding less: the target takes all of what is compressed off the store, which overshot to the least; stopping
  as compression starts to fall, or shedding in proportion, is untried [e53].
- Against `--expert-memory available`, and on a quiet machine: not measured.
