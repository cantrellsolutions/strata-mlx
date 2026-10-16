---
id: e48
script: bench/paged.py
change: szyzmoou
date: 2026-10-16
---
# e48: as a 48 GB machine, the memory held and wired

> 中文摘要:锁住 80 GiB 后,机器真的只剩 48 GB,锁住的内存全程完整。但开跑前其他程序和缓存已占约 38 GB,
> 剩下的装不下引擎的 36 GB:引擎被压缩或换出 31 GB,回答只有 10-15 tok/s。所以这是"48 GB 的 Mac 开着这一整套桌面程序",
> 不是一台安静的 48 GB 机器;要模拟后者,占住的量要扣掉当时其他程序已经用掉的那部分。

## Question

As e46, with the hold wired (e47 found an unwired hold pushed out first): what does a 48 GB machine under the rule
do when it is 48 GB in fact?

## Setup

As e46. The hold wired by `mlock`. Before the run the machine had 99 GB free of 137 and 22.5 GB of swap in use:
the author's other programs and caches held about 38 GB, and stayed open.

## Result

| round | prose | code | reasoning | edit |
| :--- | ---: | ---: | ---: | ---: |
| first, tok/s | 18.6 | 14.2 | 12.5 | 11.0 |
| again, tok/s | 10.5 | 11.0 | 10.5 | 14.5 |
| again, experts read a token | 0.1 | 0.6 | 0.0 | 0.5 |

- The hold: 80.0 GiB resident at its least over 214 samples: whole.
- This process: footprint peak 36.12 GB; 31.34 GB of it compressed or swapped out during the run, 30.29 at once
  at the most. The machine swapped out 19.92 GB.
- The rule's budget 28.3 GB of experts; 27.99 held at the end.

## Conclusion

- A wired hold makes the machine smaller: the method works, and it shows what a budget capped alone never did,
  the engine's own memory pushed out (cf. e46, the same rule, 25-34 tok/s with an unwired hold).
- The store reads next to nothing from the disk once filled, and the answers are slow all the same: its experts
  are "in memory" and macOS has compressed them, so each use waits for them to be decompressed.
- This machine was 48 GB with about 38 GB of other programs and caches in it, the engine's 36 GB on top: a 48 GB
  Mac with a full desktop open. A quiet one would be simulated by holding what the machine has beyond 48 GB plus
  what the other programs hold now, less what a quiet macOS holds (6.1 GB on issue 6's Air).

## Concepts

[[simulated-machine]] [[swapping]] [[system-reserve]] [[expert-store]]
