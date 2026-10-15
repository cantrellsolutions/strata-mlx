---
id: gpu-idle
kind: mechanism
area: prompts
status: current
---
# The GPU after a rest

> 中文摘要:GPU 空闲约 2 秒后会放掉没有锁定的数组,之后的计算慢一半以上;约 4 秒后休眠,第一个内核要等 250-320 ms。
> 所以短提示在空闲后要等半秒。PR 7 的做法:请求后 600 秒内每秒给 GPU 一个很小的运算,同时锁定数组,两样必须一起开。

## What it is

What the GPU does between requests: after about two seconds of nothing it lets go of arrays that are not wired,
and pieces then compute at half speed or less; after about four it sleeps, and the first work waits 250-380 ms.
A short prompt, whose reading takes a fraction of a second, pays all of it.

## In this engine

`strata_mlx/serve.py:KeepAwake`; the server's `--keep-awake SECONDS` (600) and `--wired off`.

## Relations

- costs → [[prompt-chunk]]: 64 tokens took 179 ms one after another and 430-580 ms after 2-8 s of nothing; prompts of 27-49 tokens 618-835 ms before the fix and 148-215 after, on the M4 Max [#1] [ssmutmqs]
- depends-on → [[gpu-generation]]: measured on an M5 Ultra and confirmed on the M4 Max [#7]
- depends-on → [[memory-rule]]: arrays are wired only where the model is held whole; near the recommended working set, as on a 64 GB machine, wiring may want room to spare [#7] [#8]

## What we know

- Kept awake alone or wired alone bought nothing on the M4 Max; wired and left alone was the slowest of all (525-819
  ms). Both together: 148-216 ms [#7].
- Answers are no slower. Idle, the small addition cost the GPU a median of 12 mW on an M5 Ultra; a laptop on
  battery not measured [#7].

## Open questions

- `--wired auto`: wiring only with room to spare.
