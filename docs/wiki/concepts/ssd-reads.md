---
id: ssd-reads
kind: hardware
area: memory
status: current
---
# Reads from the SSD

> 中文摘要:专家从 SSD 读的速度。本机 1 TB 内置盘 8 个以上线程约 6.5 GB/s;读法本身(片大小、线程数)影响不大,
> 关键是缓冲区要按页对齐才能绕过文件缓存。小内存 Mac 常见的 256-512 GB 盘慢很多,还没有测过。

## What it is

Reading experts that are not in memory: random places, expert-sized or layer-sized, past the page cache.

## In this engine

`strata_mlx/paged.py:ExpertStore` reads with `os.preadv` into page-aligned buffers, 1 MiB a piece, by a pool of threads.

## Relations

- depends-on → [[page-cache]]: what decides whether a read goes through the cache is the buffer, not the size: preadv into a page-aligned buffer leaves nothing there at 16 MiB a piece, os.pread's own bytes left 4.23 of 5.66 GB at 4 MiB [e38]

## What we know

- This machine's 1 TB internal SSD: 1.9 GB/s from one thread, 6.5 from eight or more [ROADMAP: Macs with 16, 24 and 32 GB].
- A layer's experts (0.71 GB) come at 5.9-6.5 GB/s whatever the piece (1-16 MiB), the threads (4-32) or the
  buffer; in the engine 4.0-4.8 GB/s, the rest not found [e38].
- A read in a thread of its own beside the GPU keeps 99-100% of its speed and slows the GPU's work by 1-7% [e39].

## Superseded

- "macOS puts any read of 2 MB or more through the page cache, F_NOCACHE or not": it was the buffer [e38].

## Open questions

- The 256-512 GB SSDs of small MacBooks give a quarter to a half of this one, not measured here.
