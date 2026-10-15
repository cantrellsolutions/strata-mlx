---
id: e45
script: none kept (it lived in a worktree that was removed)
change: zryukqou
date: 2026-10-14
---
# e45: pull requests 9 and 10 checked on the M4 Max

> 中文摘要:在同一个进程里给三种设置各建一个引擎(共享权重),每次前向随机选一种,比较合并前、合并 PR 9 和 10、
> 再加 e44 线程规则三种情况;并核对 PR 作者请我们看的阈值。结论:合并,但超连接的融合上限改回 6。

## Question

Pull request 9 (ten kernel changes, tuned on an 80-core M5 Ultra) and 10 (the n-gram table warmed after
loading): what do they give on this 40-core M4 Max, and do its thresholds suit it?

## Setup

Q2_0, M4 Max. Separate processes in turns were tried first and told nothing: another program kept the GPU 18-76%
busy, and one version timed 19.4 to 30.8 ms a token. So, as the pull request's author did: one process, an
engine for each setting sharing one copy of the weights, each built after its setting's switches were set (some
are fixed when a layer is compiled), the switches set again before each of its passes, passes in random order,
150 of each size, lower quartile.

## Result

| tokens | main's paths | with 9 and 10 | and e44's thread rule |
| ---: | ---: | ---: | ---: |
| 1 | 22.46 ms | 20.83 | 20.83 |
| 4 | 35.02 | 32.02 | 31.60 |
| 8 | 53.28 | 52.43 | 49.48 |

The thresholds, a second round under heavier load (compare along a row):

| tokens | main | as merged | hyper-connections fused up to 6 |
| ---: | ---: | ---: | ---: |
| 6 | 68.0 | 59.6 | 58.5 |
| 7 | 73.6 | 69.2 | 64.2 |
| 8 | 77.3 | 72.1 | 66.5 |

The wide kernel on or off: no difference. Pull request 10 not measured: this machine's table had been cached for days.

## Conclusion

Both merged as they were [wykuvxvr] [tmyqkuqv], then [zryukqou]: hyper-connections fused up to six tokens, and
e44's rule kept for the rows of 2,560 inputs only, since the down rows' table is now built in their own kernel.

## Concepts

[[decode-pass]] [[hyper-connections]] [[expert-lookup-kernel]] [[kernel-fixed-cost]] [[gpu-generation]]
[[measurement-noise]] [[draft-window]]
