---
id: e51
script: bench/paged.py
change: zxmztxsx
date: 2026-10-16
---
# e51: a margin on what is available, not settled

> 中文摘要:在 e49 的条件下给 `--available` 留 8、0、4 GB 余量,连跑三次。三次之间只隔 30 秒,启动时的可用内存却从 15.5 变到 26.2 GB,
> 这个波动比余量本身的影响大得多,所以定不出余量取多少,默认保持 0。真正的发现是:繁忙机器上启动时读到的"可用内存"很不稳定。

## Question

How much of what is available at start should `--expert-memory available` leave to other programs (e49 took all
of it and had the engine compressed)?

## Setup

As e49 (48 GB machine, 80 GiB wired, the author's desktop open), `--available --margin` 8, 0 and 4 GB, one run
each in that order, 30 s apart.

## Result

| margin | available at start | experts' budget | again, tok/s | compressed at once, most | machine swapped out |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 GB | 4.4 + 11.1 GB | 0.5 GB, the least | 5.1-7.1 | 2.6 GB | 0 |
| 0 | 10.4 + 12.9 | 12.9 | 17.1-19.1 | 5.8 | 0 |
| 4 GB | 21.4 + 4.8 | 11.9 | 10.9-13.7 | 13.8 | 15.7 GB |

(available: free + file cache.) The hold was whole in every run.

## Conclusion

- What was available at start moved by 10.7 GB between runs 30 s apart: after a hold of 80 GiB is let go, the
  programs it pushed out come back in, and the machine is not the same twice. That moves the budget by more than
  any margin tried, so the runs do not compare and the margin is not chosen. `AVAILABLE_MARGIN` stays 0.
- The 4 GB run had the smaller budget and did worse than the 0 GB one: the other programs were coming back in
  through it (15.7 GB swapped out), which no budget set at start can see.
- Settling a margin wants a quiet machine, or runs spaced until what is available is steady.

## Concepts

[[available-memory]] [[measurement-noise]] [[simulated-machine]]
