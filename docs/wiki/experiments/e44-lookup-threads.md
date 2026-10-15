---
id: e44
script: bench/experiments/e44_lookup_threads.py
change: skxmzozu
date: 2026-10-14
---
# e44: threads to a row of the table kernel

> 中文摘要:查表内核每行用几个线程,原先按"少于 21 对用 16,否则 4"。扫描后改成"40 对以内用 8,否则 2",
> 在 M4 Max 上专家部分快 3% 到 27%,端到端带草稿头 8 组对比赢 7 组,约 6%。

## Question

A window of four to eight tokens is 40 to 80 (token, expert) pairs; the table kernel took 11 ms of 48 layers at
four tokens and 30 at eight. Is its split of rows across threads right for those sizes?

## Setup

Q2_0, M4 Max, the 48 expert layers chained. A sweep of threads to a row for the gate and up rows (2,560 inputs)
and the down rows (640), then the chosen rule alternated with the old one, six rounds each. Then
`bench/spec_decode.py` through four prompts, in the order new, old, old, new.

## Result

| tokens | old rule | new rule |
| ---: | ---: | ---: |
| 1 | 4.19 ms | 4.07 (+3%) |
| 4 | 8.92 | 8.18 (+8%) |
| 8 | 75.35 | 56.73 (+25%) |
| 128 | 607.08 | 505.20 (+17%) |

The machine slowed through the run (the eight-token round took 75 ms where it had taken 20); the rules were
alternated, so the comparison holds and the figures do not.

Whole answers, drafts as sure as the head is, tok/s, new / old in two pairs: prose 77.2 / 76.4 and 74.0 / 68.2,
code 77.5 / 74.5 and 71.4 / 66.3, reasoning 100.4 / 92.6 and 89.9 / 84.3, editing 99.9 / 84.3 and 77.6 / 80.8.
Guesses kept: prose 63-69%, code 59-69%, reasoning 78-89%, editing 77-89%.

## Conclusion

Taken: 8 threads up to 40 pairs and 2 beyond for rows of 2,560 inputs. The down rows' part was undone by
pull request 9 (see e45).

## Concepts

[[expert-lookup-kernel]] [[gpu-generation]] [[draft-window]] [[guess-acceptance]] [[measurement-noise]]
