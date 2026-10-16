---
id: available-memory
kind: mechanism
area: memory
status: current
---
# Available memory at start

> 中文摘要:`--expert-memory available`:启动前读一下当时空闲的内存和文件缓存,专家的份额不超过这个数减去加载量和工作内存。
> 它针对"开着很多别的程序"的情况,换来的是不换页和稳定,平均速度并不更快。是可选项,不是默认。

## What it is

An option that bounds the experts' budget by what was free or in the file cache before anything was loaded, less
what is loaded and the [[working-memory]]; the [[memory-rule]]'s figure stays the upper bound. It answers what a
constant cannot: other programs holding more than the rule leaves them.

## In this engine

`strata_mlx/paged.py:available_memory` (from `sysctl`), `strata_mlx/paged.py:fit` (`available=`).

## Relations

- depends-on → [[page-cache]]: it counts file-backed pages as available, which macOS gives up before it swaps; the model's own files fill the cache as it loads, so it counts before loading [trprrzsw] [ooosmwnq]
- trades-against → [[swapping]]: with apps open on a 32 GB Mac, 8.08 GB of experts swapped nothing and wrote 17.1 tok/s; 10 GB by hand swapped 0.09 GB and wrote 18.9; 12 GB 1.24 GB and 20.8 [#6]

## What we know

- Counted after loading, the first version took 9 GB of the model's own just-read files as available; they were
  given up within 20 s, with no swap [ooosmwnq].
- Kept an option: it buys no swapping and steadier answers (15.9-17.7 tok/s against 10.8-24.7 with the fixed
  rule), not a higher mean [#6].

- Under held, wired memory as a 48 GB machine with a full desktop: 20.9 GB of experts where the rule gives
  28.3, answers 17.4-25.6 tok/s where 10.5-14.5, the machine swapping 0.96 GB where 19.9 [e48] [e49].
- It leaves no margin: the process grew to 29.9 of the 31.3 GB available at its start, and 23.3 GB of it was
  compressed at once at the most [e49].

## Superseded

- Counting after loading [trprrzsw]: before, since [ooosmwnq].

## Open questions

- Following the pressure at run time, not only at start: letting go of experts would take the compression
  off [e52].

- A reading at start says nothing of what other programs will ask for after it: with the smaller budget of the
  two, a run during which they came back in did worse [e51].

- How much to leave out of what is available, or how much of the file cache to count [e49]. Tried at 8, 0 and
  4 GB on a busy machine: what was available moved by 10.7 GB between runs 30 s apart, more than any margin [e51].
  `strata_mlx/paged.py:AVAILABLE_MARGIN` is 0.
