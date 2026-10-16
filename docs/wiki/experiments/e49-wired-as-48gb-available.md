---
id: e49
script: bench/paged.py
change: kzynqwqp
date: 2026-10-16
---
# e49: as a 48 GB machine wired, the experts bounded by what was available

> 中文摘要:与 e48 相同(锁住 80 GiB,开着一整套桌面程序),只加了 `--available`。专家份额从 28.3 降到 20.9 GB,
> 第二轮回答从 10.5-14.5 升到 17.4-25.6 tok/s,整台机器换出从 19.9 降到 0.96 GB。但引擎同一时刻仍有最多 23.3 GB 处于压缩状态:
> 这个选项把启动时的可用内存几乎用尽,没有给其他程序留余量。

## Question

Does `--expert-memory available` keep the engine in memory where the fixed rule did not (e48)?

## Setup

As e48, with `--available`: the experts' budget no more than what was free or in the file cache when the hold had
started and before loading, less what was loaded and the working memory.

## Result

| round | prose | code | reasoning | edit |
| :--- | ---: | ---: | ---: | ---: |
| first, tok/s | 21.1 | 13.3 | 21.4 | 14.7 |
| again, tok/s | 17.4 | 22.9 | 25.6 | 21.9 |
| again, experts read a token | 14.7 | 8.8 | 13.1 | 17.5 |

- Before loading: 24.6 GB free and 6.7 GB of file cache; the experts' budget 20.94 GB, where the rule gives 28.30.
- The hold: 80.0 GiB resident throughout (137 samples).
- This process: footprint peak 29.86 GB; 23.3 GB of it compressed at once at the most, 41.8 GB over the run.
- The machine swapped out 0.96 GB, where e48's swapped 19.92.

## Conclusion

- Under pressure the option does what it is for: answers about twice e48's, and next to no swapping.
- It takes all that was available: the process came to 29.9 GB of a 31.3 GB start, and macOS made room for the
  other programs by compressing it. A margin left out of what is available, or the file cache counted in part,
  is the next thing to try; this run does not say how much.

## Concepts

[[available-memory]] [[swapping]] [[simulated-machine]] [[system-reserve]] [[pressure-following]]
