---
id: decode-pass
kind: concept
area: answers
status: current
---
# A decode pass

> 中文摘要:写回答时模型的一次前向。不带草稿头时一次一个 token,带草稿头时一次 2 到 8 个 token(一个窗口)。
> 写回答的速度 = 每次前向定下的 token 数 ÷ 一次前向的耗时,所以这里的每一毫秒都直接换算成 tok/s。

## What it is

One forward of the model over the tokens of a window: one token without the draft head, up to eight with it
([[draft-window]]). Answers are written a pass at a time, so tokens a second are the tokens a pass settles over
the time a pass takes.

## In this engine

`strata_mlx/engine.py:Engine` (`_forward`, `_commit`); the parts are built by `strata_mlx/load.py:load`.

## Relations

- limited-by → [[bandwidth-floor]]: a token reads 5.5 GB of weights, 4.87 of them dense tensors and 0.66 experts; reading them once takes 10.8 ms, the GPU's part of a one-token pass 16.8 ms [ROADMAP: Faster answers] [e55]
- limited-by → [[kernel-fixed-cost]]: a pass is about 3,300 kernels, most of them small and each waiting for the one before [ROADMAP: Now: what is left] [#9]
- depends-on → [[measurement-noise]]: its time moves by half within minutes with other programs on the GPU, so changes are compared in one process [e45]

## What we know

Parts, measured by leaving each out of a real pass (Q2_0, M4 Max, ms for one / four tokens) [e06]:

| part | at the start | 2026-10-14 |
| :--- | ---: | ---: |
| routed experts ([[expert-lookup-kernel]]) | 4.1 / 12.6 | 4.7 / 12.3 |
| delta-net projections | 6.0 / 8.7 | 5.4 / 9.0 |
| [[hyper-connections]] | 5.1 / 8.1 | 3.7 / 7.1 |
| attention projections | 2.3 / 3.9 | 1.4 / 5.1 |
| output layer | 2.3 / 3.1 | 2.4 / 11.1 (a slow round) |
| routers | 2.0 / 2.7 | 0.4 / 4.0 |
| shared experts | 1.8 / 2.5 | 0.8 / 1.4 |

The same left out again in one process, passes shuffled with a control, lower quartile, after e57 (a pass of
17.6 / 28.8 ms): delta-net projections 5.4 / 7.0, the delta-net recurrence 2.7 / 3.1 (no weights read; but
alone it is 1.2 / 1.35 ms [e59]: left out, what only fed it went too), attention 2.1 / 3.8, routed experts 2.1 / 7.6, output layer 1.0 / 2.0, shared experts 0.5 / 0.9,
routers 0.2 / 0.6 [e58]. Leaving a part out also leaves out what only it used, so a part
with work upstream reads high [e59].

The 2026-10-14 column was taken while the machine slowed; the whole pass went from 21 to 24 ms through it, so it
places the parts and does not time them.

Whole passes, Q2_0, M4 Max, one process, passes shuffled, lower quartile [e45]:

| tokens | before pull request 9 | since [wykuvxvr] and [zryukqou] |
| ---: | ---: | ---: |
| 1 | 22.46 ms | 20.83 |
| 4 | 35.02 | 31.60 |
| 8 | 53.28 | about 46-49, not measured as merged |

A step, the engine timed in place [e55]: a token a pass 19-20 ms, of which the GPU 16.8-17.6 and the CPU alone
2.3-2.6 (building the graph 1.8-2.0, committing 0.4); with the draft head the CPU alone is 3.8-4.3 ms a step, the
head itself 3.8-6.6 ms. The CPU's share is 10-12% of a step.

Handing the pass to the GPU every 6 layers while the rest is built takes 2.4-3.0 ms off a one-token pass and
1.8-2.7 off a four-token one, the same bits; on by default (`engine.SUBMIT_EVERY`) [e57].

## Superseded

- "A pass is not far from its floor; no single large target is left": the bytes were then put at 2 GB a token,
  and they are 5.5 [ROADMAP: Now: what is left].

## Open questions



- What the delta-net projections (the largest dense part) would give to a kernel that reads their weights once
  for a window; MLX's product is not flat from one token to four for the 10,240 x 2,560 shape (50 to 83 us).
