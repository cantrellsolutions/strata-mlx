---
id: e47
script: bench/paged.py
change: zzzlsupr
date: 2026-10-16
---
# e47: as a 32 GB machine, the memory held but not wired

> 中文摘要:模拟 32 GB 机器,另一个进程占住 96 GiB。新加的检查判定这次运行无效:占住的内存最少时只剩 8.9 GiB 常驻,
> 其余被 macOS 压缩或换出,因为它一直闲着,是全机最冷的内存。结论:被动占住内存模拟不了压力,要锁住(mlock)才行。

## Question

As e46, as a 32 GB machine: 96 GiB held, the rule's 15.4 GB of experts.

## Setup

As e46, `--machine-memory 32`; the machine had about 94 GB free before the run and 18 GB of swap in use.

## Result

| round | prose | code | reasoning | edit |
| :--- | ---: | ---: | ---: | ---: |
| first, tok/s | 15.7 | 18.0 | 19.9 | 16.3 |
| again, tok/s | 18.9 | 18.3 | 23.5 | 19.2 |
| again, experts read a token | 27.6 | 17.9 | 21.6 | 40.7 |

- The hold: 8.91 GiB resident at its least over 146 samples, of 96. The run is invalid as a test of pressure.
- This process: footprint peak 23.81 GB; 1.57 GB of it compressed or swapped out during the run.
- The machine swapped out 33.88 GB, and its swap in use grew from 18 to 24 GB.

## Conclusion

- Memory held and left alone is the coldest on the machine: macOS compressed and swapped out nearly all of it
  before it touched the engine's. So the machine was not 32 GB, and the answers (18-24 tok/s once the store had
  turned over) are a simulated 32 GB machine's of the old kind, with the page faults of a busy system.
- A hold must be wired to stand for memory a machine does not have. `bench/memory.py` wires it since [zzzlsupr]; a user
  process may wire up to `vm.user_wire_limit`, 116.8 GB here.
- The check that found this is the one added in e46's correction: the hold judged by its resident part.

## Concepts

[[simulated-machine]] [[swapping]] [[measurement-noise]]
