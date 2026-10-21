---
id: e57
script: bench/experiments/e57_submit_in_parts.py
change: qmvxvpnt
date: 2026-10-21
---
# e57: a pass handed to the GPU in parts, not measured

> 中文摘要:让前向每隔 6/12/16 层先交给 GPU(Engine(submit_every=N)),和现在、对照组在同一进程里随机交替。
> 这次运行无效:其他程序让 GPU 有 42-84% 被占用,一次 1 个 token 的步长从 e55 的 19 ms 变成 33-48 ms,对照组和"现在"相差 7 ms,
> 比要测的差别大得多。只说明了一件事:不带草稿头时,分段提交的输出和现在逐位相同。

## Question

Does handing a pass to the GPU in parts while the CPU builds the rest (`mx.async_eval` every N layers) take the
1.8-2.1 ms of building off a step [e55], and is the output the same?

## Setup

M4 Max 128 GB, Q2_0 whole in memory, change qmvxvpnt. One process, engines sharing the weights: `submit_every` 0 (as
now) and its control, 6, 12, 16. Four prompts, 160 tokens, four rounds shuffled. The GPU was 42% busy with other
programs at the start and 84% after (WindowServer, Music). Results `bench/results/e57_submit_in_parts.json`.

## Result

| engine | now | control | every 6 | every 12 | every 16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| a token a pass, step ms | 41.5 | 34.6 | 39.9 | 48.3 | 32.6 |
| draft head, step ms | 80.7 | 82.0 | 83.3 | 77.2 | 78.6 |

## Conclusion

- **Not measured**: the control and "now", the same engine, differ by 7 ms a step, more than the 2 ms looked for;
  a step was twice e55's. To be run again with the machine quiet.
- A token a pass, every setting gave the reference's tokens. With the draft head even "now" did not repeat its own
  tokens: how many guesses a pass takes depends on timings (drafting.py), and a different window rounds float16
  differently, so the check means nothing there.

## Concepts

[[decode-pass]]
