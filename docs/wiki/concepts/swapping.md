---
id: swapping
kind: metric
area: memory
status: current
---
# Swapping

> 中文摘要:内存不够时 macOS 把内存页写到 SSD 上。它会让写回答的速度掉一半以上,而且在 128 GB 的机器上模拟小内存
> 是看不到的。目前所有关于换页的数据都来自用户的真实机器。

## What it is

macOS writing pages of memory out to the SSD when it runs short, and reading them back when they are wanted.
Counted system-wide (`vm_stat`, `sysctl vm.swapusage`), so a count says that the machine swapped, not which
allocation made it.

## In this engine

Nothing in the engine sees it; `--check` and the [[memory-rule]] are meant to keep it from happening.

## Relations

- costs → [[decode-pass]]: a real 32 GB Mac that swapped 106 GB in a run wrote answers at 5.8-19.2 tok/s, and 23.4-26.8 with the budget that did not swap [#6]
- invalidates → [[simulated-machine]]: a budget capped on a 128 GB machine never swaps; the README's "23-38 tok/s as a 32 GB machine" was 5.8-19.2 on a real one that did [#6]

## What we know

- What the engine loses first is its own memory compressed, not reads: as a 48 GB machine wired, with a full
  desktop, its store read next to nothing once filled and answers came at 10-15 tok/s, 31 GB of the process
  compressed or swapped out [e48].

- 32 GB MacBook Air M5: with apps open and the fixed rule, 16.8 GB swapped out; with `--expert-memory available`,
  1.70 and 0.41 GB [#6] [trprrzsw].
- 48 GB M4 Pro: 4.98 GiB swapped out at the start with 30.1 GB of experts, none with 24 [xukwosuo].
- 64 GB M5 Pro: an IQ3_XXS file held whole with 2.3 GiB to spare swapped 7.4-8.2 GiB out during startup and the
  warm-up, and nothing with 4 GB of parked conversations or later [#8].

## Open questions

- macOS pushes out the coldest memory first: an idle hold of 96 GiB went down to 8.9 resident while the
  engine lost 1.57 GB [e47].

- Here, as a 48 GB machine with 80 GiB held by another process and a full desktop open, 25 GB swapped out with
  the hold "whole" by a footprint that counts swapped pages too: the hold's, the engine's or the other programs',
  not told apart [e46].

- Which allocation the startup swaps come from.
