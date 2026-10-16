---
id: e50
script: bench/paged.py
change: kmznkvlw
date: 2026-10-16
---
# e50: as a quiet 32 GB machine, the memory wired

> 中文摘要:用 `--hold quiet` 模拟一台只跑这个模型的 32 GB 机器(锁住 84.4 GiB,扣掉本机其他程序多占的 12.4 GB)。
> 没有换页,但引擎同一时刻仍有最多 11.9 GB 被压缩,回答 18.1-21.8 tok/s,比真实的 32 GB Air 关掉其他程序时的 24.8-28.7 慢约四分之一。
> 本机其他程序一直在活动、不断要内存,这一点在模拟里扣不掉,所以结果偏悲观。

## Question

Made a quiet 32 GB machine in fact, does the engine under the rule match the real 32 GB MacBook Air of issue 6
with nothing else open?

## Setup

As e48, `--machine-memory 32 --hold quiet`: other programs held 18.5 GB beyond the free memory and the file
cache, and a quiet macOS 6.1, so 12.4 GB less was held, 84.4 GiB wired. The default rule, no `--available`.

## Result

| round | prose | code | reasoning | edit |
| :--- | ---: | ---: | ---: | ---: |
| first, tok/s | 16.4 | 17.7 | 19.8 | 17.4 |
| again, tok/s | 18.1 | 20.1 | 21.8 | 19.1 |
| again, experts read a token | 28.7 | 20.3 | 23.1 | 40.8 |

- The experts' budget 15.42 GB. The hold: 84.4 GiB resident throughout (136 samples).
- This process: footprint peak 24.85 GB; 11.9 GB of it compressed at once at the most, 18.1 over the run.
- The machine swapped out nothing.

## Conclusion

- The real Air with nothing else open, the rule giving it 16.2-16.5 GB of experts: 24.8-28.7 tok/s, 0.89 GB
  swapped [#6]. Here 18.1-21.8 on a GPU four times its size: a quarter slower.
- What makes the difference is the engine compressed while there was room by the arithmetic: this machine's other
  programs stay active and keep asking for memory, which a hold cannot take out of the simulation. So a quiet
  hold is nearer a quiet machine than a hold of all but N GiB, and still pessimistic where the desktop is busy.

## Concepts

[[simulated-machine]] [[swapping]] [[system-reserve]] [[pressure-following]]
