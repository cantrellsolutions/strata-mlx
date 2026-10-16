---
id: e46
script: bench/paged.py
change: vrnvzqql
date: 2026-10-16
---
# e46: as a 48 GB machine, with the memory held

> 中文摘要:第一次真正"变小"地模拟 48 GB 机器:另一个进程占住 80 GiB 随机字节,全程没有被换出。引擎按规则拿到 28.3 GB 专家,
> 进程峰值 36.3 GB;第二轮回答 25-34 tok/s,和一台真实 48 GB M4 Pro 封顶 24 GB 时的 23.6-34.1 相近。
> 但测量期间整台机器换出了 25.3 GB,分不清是谁的:当时判断占内存进程"完整"用的是 footprint,而 footprint 把已压缩、已换出的页也算在内,
> 所以连占内存的进程本身也不能排除。之后改为按常驻内存判断,并记录引擎进程自己被压缩或换出的量。

## Question

Does the rule now (a quarter of the memory left to macOS) keep a 48 GB machine from swapping, when the machine is
made smaller in fact and not only in the budget?

## Setup

M4 Max, 128 GB, Q2_0, `bench/paged.py --machine-memory 48 --hold auto --engines head`: 80 GiB of random bytes held
by another process for the whole run, the draft head, four prompts of 200 tokens, twice over (the store starts
empty and keeps what the first round brought in), the experts' shard dropped from the page cache first. Before
the run the machine had about 61 GB free and 18 GB of swap in use; the author's own programs (browsers, an
editor, Xcode, a dozen agent sessions) stayed open, and they are part of what the machine held.

## Result

| round | prose | code | reasoning | edit |
| :--- | ---: | ---: | ---: | ---: |
| first, tok/s | 15.6 | 19.2 | 22.5 | 23.2 |
| again, tok/s | 31.1 | 24.8 | 31.6 | 33.9 |
| again, experts read a token | 0.1 | 1.0 | 0.0 | 0.2 |

- The experts' budget by the rule: 28.3 GB; held at the end 28.05 GB. MLX's arrays peaked at 35.6 GB, the kernel's
  footprint of the process at 36.33 GB.
- The hold: 80.06 GiB of footprint at its least over 120 samples. That was taken for "whole", and it does not
  say so: a footprint counts pages compressed or swapped out as well. Since [tukzwlpx] a hold is judged by what of it
  is resident.
- The machine swapped out 25.28 GB while the run went on.

## Conclusion

- The answers once the store is filled, 25-34 tok/s, are those of a real 48 GB M4 Pro with the experts capped at
  24 GB (23.6-34.1, three prompts of 128 tokens), reported on X after `9f37174`.
- 25 GB swapped out. It may have been the hold's (see above), this process's or the other programs'; whether it
  was any one of them a system-wide count does not tell. Since [tukzwlpx] the run counts this process's own memory
  compressed or swapped out, and the hold's resident part. With 85.9 GB held, 36.3 in this process and the author's programs, the machine had little left, so
  this is a 48 GB machine with a full desktop open, not a quiet one.

## Concepts

[[simulated-machine]] [[swapping]] [[system-reserve]] [[measurement-noise]]
