---
id: e57
script: bench/experiments/e57_submit_in_parts.py
change: rmxrzznt
date: 2026-10-21
---
# e57: a pass handed to the GPU in parts, and gate with up in one kernel

> 中文摘要:前向建图时每隔 6 层先把建好的部分交给 GPU(Engine(submit_every=6)),1 个 token 的前向省 2.4-3.0 ms(约 10-12%),
> 4 个 token 省 1.8-2.7 ms;每 3、4、8 层效果相同。专家的 gate 和 up 合成一个内核再省 0.4-0.7 ms。两项都不改变输出,已默认开启。
> 按整段回答轮换的测法这台机器测不出来(长时间满负载会降速,同一设置前后差 4-7 ms),要按单次前向随机交替、取下四分位。

## Question

Does handing a pass to the GPU in parts while the CPU builds the rest (`mx.async_eval` every N layers) take the
1.8-2.1 ms of building [e55] off a pass? And gate with up in one kernel, since at one token a kernel reading a
layer's experts does not reach the bandwidth [e56]?

## Setup

M4 Max 128 GB (a 14-inch MacBook Pro), Q2_0 whole in memory. Engines sharing the weights in one process, each
compiled with its own setting: `submit_every` 0 (as before) and its control, 3, 4, 6, 8, 12, 16; gate and up in
one kernel (`blocks.GATE_UP_ONE_KERNEL`, read when an engine's halves are traced). First tried as whole answers
in turn ([qmvxvpnt]); then single passes of 1 and 4 tokens after a 300-token conversation, every (engine, size)
shuffled, 200 rounds, the lower quartile. Music and the like closed; WindowServer at half a core. Results
`bench/results/e57_submit_in_parts.json` (answers), `e57_passes.json`, `e57_passes_2.json`.

## Result

Whole answers in turn: a step 33-42 ms against e55's 19, "now" and its control 4.4-7 ms apart; not a measurement.

Single passes, lower quartile, ms (the second round throttled throughout: compare within a round):

| setting | round 1: 1 token | 4 tokens | round 2: 1 token | 4 tokens |
| :--- | ---: | ---: | ---: | ---: |
| as before | 19.87 | 30.21 | 28.98 | 58.49 |
| its control | 19.90 | 29.85 | 29.16 | 58.94 |
| every 3 | | | 26.02 | 55.82 |
| every 4 | | | 26.14 | 56.44 |
| every 6 | 17.45 | 28.41 | 26.26 | 55.77 |
| every 8 | | | 26.53 | 56.21 |
| every 12 | 18.17 | 27.95 | | |
| every 16 | 18.25 | 28.37 | | |
| gate and up in one kernel | 19.20 | 29.48 | | |
| every 6 and gate+up | | | 25.72 | 55.40 |

## Conclusion

- **Submitting in parts takes 2.4-3.0 ms off a one-token pass (10-12%) and 1.8-2.7 ms off a four-token one**,
  nearly all of the CPU's building time e55 found. Where the parts fall hardly matters (3 to 8 layers within
  0.5 ms). On by default, every 6 layers; not where experts are read from the SSD (a pass stops at every layer).
- **Gate and up in one kernel: 0.4-0.7 ms more**, the same bits (`tests/test_blocks.py`). On by default.
- Both keep the graph or the sums, so the bits; a token a pass, every setting gave the reference's tokens.
- **Answers in turn do not measure this machine**: under minutes of load it slows (bench/quiet.py), and the same
  engine twice differed by more than the effect. Single passes shuffled, lower quartile, do.
- Not measured on an M5; the overlap should hold wherever building a pass takes the CPU time it does here.

## Concepts

[[decode-pass]] [[expert-lookup-kernel]] [[measurement-noise]]
