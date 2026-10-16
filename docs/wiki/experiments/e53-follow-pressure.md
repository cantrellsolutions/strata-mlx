---
id: e53
script: bench/paged.py
change: lptnxxvz
date: 2026-10-16
---
# e53: the budget following the pressure, off and on

> 中文摘要:模拟 48 GB(锁住 80 GiB、开着桌面程序),按关、开、开、关各跑一次。打开之后,引擎被压缩的量在测量时降到 0,
> 整台机器几乎不再换页(0–0.42 GB,关闭时是 24.7–27.4 GB),读提示也快了;但写回答没有变快,
> 因为份额一路降到 3–7 GB 甚至最低值,专家改从 SSD 读,读盘占了 34–66% 的时间。收缩得太多,默认保持关闭。

## Question

Does following the pressure at run time ([[pressure-following]]) make answers faster on a machine under pressure,
and does the budget settle?

## Setup

M4 Max 128 GB, Q2_0 with the draft head, `bench/paged.py --machine-memory 48 --hold auto` (80 GiB wired, the
author's desktop open, as e48), the fixed rule's budget 28.3 GB at start. Off, on, on, off, each run after the
machine had settled (60-270 s). Change lptnxxvz. Results `bench/results/e53_follow_*.json`.

## Result

The second round of each prompt (prose, code, reasoning, edit):

| run | answers, tok/s | prompt, tok/s | compressed at the row | budget at the row | reading from the disk | compressed at once, most | machine swapped out |
| :--- | :--- | :--- | ---: | :--- | :--- | ---: | ---: |
| 1 off | 10.6, 12.9, 7.5, 8.7 | 4.7-15.3 | 30.0-30.3 GB | 28.3 | 0-1% | 30.3 GB | 24.7 GB |
| 2 on | 10.0, 11.5, 12.9, 12.9 | 12.0-69.7 | 0.0 | 4.0-6.2 | 43-48% | 18.7 | 0 |
| 3 on | 9.8, 9.1, 9.8, 5.3 | 10.1-30.2 | 0.0-0.5 | 6.8-7.9, then 0.5 | 34-66% | 18.4 | 0.42 |
| 4 off | 9.5, 9.7, 8.9, 8.6 | 4.4-16.0 | 30.2-30.6 | 28.3 | 0-4% | 30.7 | 27.4 |

The hold was whole in every run. With it on, the budget went down 2 GB a second from 28.3 to 3-4.7 GB, then came
back 1 GB at a time; in run 3 it fell to the least (0.5 GB) again during the last prompt.

## Conclusion

- It does what it is for: the engine's compression is off at measuring time, and the machine stops swapping out
  (0-0.42 GB against 24.7-27.4). Prompts read 2-4 times faster.
- Answers are not faster: medians 9.2-9.7 tok/s off, 9.5-12.2 on, and 5.3 where it fell to the least. Compression
  was traded for reading from the SSD, which took 34-66% of an answer's time.
- It shed too much. Reading of mine, not measured apart: the target is the store's size less what is compressed,
  and what is compressed (30 GB) is more than the store held, so the target was the least each second and only the
  2 GB cap slowed it; compression comes off later than experts are let go, so it went on shedding after enough.
- Off by default still. Next: stop shedding as soon as compression starts to fall, shed in proportion to the
  excess, and measure on a quiet machine, where e50 had 11.9 GB compressed with nothing swapped.

## Concepts

[[pressure-following]]
