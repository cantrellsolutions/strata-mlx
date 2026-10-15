---
id: page-cache
kind: concept
status: current
---
# The page cache

> 中文摘要:macOS 把读过的文件内容留在内存里,下次读直接给。它让基准测得过快(必须先清掉两个分片),
> 也是 n-gram 表和"可用内存"计算依赖的东西。内存紧张时 macOS 先放掉它,再换页。

## What it is

The file pages macOS keeps in memory after a read, which answer the next read of them. It is given up before
anything swaps.

## In this engine

The model's files are read past it where experts come from the SSD; `bench/drop_cache.py` empties it for a file.

## Relations

- invalidates → [[simulated-machine]]: a run with the file in the cache made the same run 2.4 times faster; a 1,537-token prompt read at 273-287 tok/s with the file cached and 149-162 without [ROADMAP: The dense tensors of two types]

## What we know

- The engine's n-gram table is read through it, by page faults: a machine with memory to spare has it cached,
  a 32 GB one cannot hold it, and with it cold an 8,192-token prompt takes 13-14 s more [uoonprvo]. See [[ngram-table]].
- Loading the model leaves its own files in it for about 20 s, which is why available memory is counted before
  loading [ooosmwnq].
