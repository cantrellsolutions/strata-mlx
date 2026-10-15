---
id: layer-read-ahead
kind: mechanism
area: memory
status: current
---
# The next layer read ahead

> 中文摘要:按层读提示时,算这一层的同时由另一个线程读下一层的专家。8K 的一段里一层算 0.29-0.34 秒,读一层 0.13 秒,
> 所以读盘几乎全藏在计算后面。PR 7 之后只读内存里缺的那部分。

## What it is

While a prompt's piece goes through one layer, a thread reads the next layer's experts and makes their arrays;
the main thread takes them when it comes to that layer. It evaluates nothing of MLX on its own.

## In this engine

`strata_mlx/paged.py:ahead`; the server's `--read-ahead off`.

## Relations

- part-of → [[layer-major-prefill]]: its second stage [utpukvqp]
- depends-on → [[ssd-reads]]: a layer's read (0.13 s here) hides behind a layer of an 8,192-token piece (0.29-0.34 s) but not behind one of a short piece (0.12 s at 2,560 tokens) on a disk of a quarter this speed [e40]
- depends-on → [[expert-store]]: since pull request 7 it reads only what memory lacks: as a 64 GB machine with IQ3_S, the 1,180-token prompt read 1.4 GB where it read 32 [#7] [volxnpkv]

## What we know

- Read ahead, an 8,192-token prompt waits 0.2 s for the disk in all; the reads take 8.6-9.3 s beside the GPU's
  work, at 3.7-4.0 GB/s [e40].
- With the n-gram table cold it buys less (38.0 to 36.2 s): the page faults and the reads ask the same disk [uoonprvo].

## Superseded

- "Reading only what memory lacks saves no time on this disk and waits for a machine with a slower one" [e40]:
  built in pull request 7, which counts on the 64 GB machines that hold most of a file [#7].
