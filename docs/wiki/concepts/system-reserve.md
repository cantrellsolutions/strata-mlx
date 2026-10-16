---
id: system-reserve
kind: constraint
area: memory
status: current
---
# What is left to macOS

> 中文摘要:内存规则留给 macOS 和其他程序的那一份。它从 4.5 GB 改到 8.5 GB,再改成内存的四分之一(至少 8.5 GB),
> 每次改动都来自一台真实的小内存机器在默认设置下换页。目前的依据只有两台机器。

## What it is

The part of a machine's memory the [[memory-rule]] does not give to the model: macOS itself and whatever else the
user runs. It cannot be measured from inside the engine, so it is a rule fitted to machines that were heard from.

## In this engine

`strata_mlx/paged.py:left_to_system` (a quarter of the memory, `strata_mlx/paged.py:SYSTEM` at least).

## Relations

- costs → [[memory-rule]]: every gigabyte left here is a gigabyte of experts less on a machine that reads them from the SSD [xukwosuo]
- limited-by → [[simulated-machine]]: a capped budget on a 128 GB machine never runs short, so no simulated run could have found it [#6]
- trades-against → [[swapping]]: too little left and the machine swaps; too much and fewer experts stay in memory [#6] [xukwosuo]

## What we know

- 4.5 GB was the first figure. A real 32 GB MacBook Air M5 swapped 106 GB out in a run with the 20.5 GB of experts
  that gave, answers at 5.8-19.2 tok/s; with 16 GB of experts it was about twice as fast. With nothing else open
  macOS itself held 6.1 GB [#6] [zqxnwnnk].
- 8.5 GB then. A real 48 GB M4 Pro swapped 4.98 GiB out as it started with the 30.1 GB of experts that gave, and
  not with 24; Strata's own Metal engine chose 25.6 there [xukwosuo].
- A quarter of the memory, 8.5 GB at least, fits both: the same for 32 GB, 12.9 GB for 48 [xukwosuo].
- As a 48 GB machine with the memory held, the rule gave the experts 28.3 GB and the answers came at 25-34 tok/s
  once they were in memory, those of the real M4 Pro capped at 24 [e46].

## Superseded

- 4.5 GB [zqxnwnnk]. 8.5 GB for every machine [xukwosuo].

## Open questions

- Two machines. A 64 GB M5 Pro did not swap at 93% of IQ3_S's experts, which the quarter cuts to 76%: the share
  may want to fall as the memory grows.
