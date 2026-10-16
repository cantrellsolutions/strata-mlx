---
id: expert-store
kind: component
area: memory
status: current
---
# The expert store

> 中文摘要:内存装不下全部专家时,专家放在一个按"最久未用先淘汰"管理的存储里。每层等路由器选出专家,缺的从 SSD 读,
> 读完再继续。即使专家全在内存,这种跑法每个 token 也要多等 10 ms,所以装得下的机器不走这条路。

## What it is

Where experts are read from the SSD in part: the experts in memory within a budget, the least recently used
dropped first. A layer waits for its router, reads what is missing and goes on.

## In this engine

`strata_mlx/paged.py:ExpertStore`, `strata_mlx/paged.py:PagedSwitchGLU`; filled at start with the experts used
most (`strata_mlx/paged.py:preload`), by the engine's profile and then the machine's own.

## Relations

- costs → [[decode-pass]]: the wait for each layer's router costs 10 ms a token even when nothing is read: 22 tok/s with every expert resident this way, 43 held whole [ROADMAP: Macs with 16, 24 and 32 GB]
- limited-by → [[ssd-reads]]: with 2 GB of experts in memory the disk is 65-71% of an answer's time, with 16 GB 6-14% [e35]
- depends-on → [[memory-rule]]: its budget [ROADMAP: Macs with 16, 24 and 32 GB]

## What we know

- A token uses 480 of the 24,576 experts, not the same few. With the most used elsewhere in memory and the least
  recently used replaced, misses a token: 226 with 2.1 GB, 147 with 4.1, 66 with 8.3, 21 with 16.6
  [ROADMAP: Macs with 16, 24 and 32 GB].
- The draft head helps here by a third to a half: the wait at each layer is paid once for a window.
- Filling the budget at start with the experts used most buys little (29.5 to 31.5 tok/s for the first answer
  as a 32 GB machine).

## Open questions

- Shedding experts under pressure takes their compression off nearly one for one, whichever experts they are:
  macOS compresses those the GPU reads every second as much as idle ones [e52].

- The store counts an expert as in memory when macOS has compressed it: misses near nothing and answers at
  10-15 tok/s as a wired 48 GB machine with a full desktop [e48].

- The 64 GB M5 Pro with IQ3_S, 93% of its experts in memory, wrote answers at 17-44 tok/s: the router wait costs a
  machine that nearly holds a file as much as the disk does [#8].
