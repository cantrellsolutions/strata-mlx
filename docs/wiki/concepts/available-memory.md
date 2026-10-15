---
id: available-memory
kind: mechanism
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

## Superseded

- Counting after loading [trprrzsw]: before, since [ooosmwnq].
