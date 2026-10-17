---
id: e54
script: bench/experiments/e54_shared_weight_controls.py
change: uvuuktql
date: 2026-10-15
---
# e54: the fusion limit on an M5 Pro, against a control engine

> 中文摘要:PR 11 的作者在一台 20 核 GPU 的 M5 Pro(64 GB)上,用同一进程、共享权重、带对照引擎的方法测了超连接的
> 融合上限:7、8 个 token 的窗口里,融合到 8 比融合到 6 慢 1.3-1.7 ms(Q2_0)和 1.1-1.2 ms(IQ3_S),和 80 核
> M5 Ultra 上的结果相反。所以只看 GPU 的代数选不出这个上限。合并时加了 `--fusion-limit` 手动指定,自动规则不变。
> 数字是作者机器上的,不是这里测的。

## Question

The hyper-connections are fused up to eight tokens from GPU generation 17, because an 80-core M5 Ultra was faster
that way [#9]. Does that hold on a 20-core M5 Pro, which is generation 17 too, and is the difference larger than
what a control engine of the same setting varies by?

## Setup

Measured by the pull request's author on their machine, not here [#11]: a Mac mini, M5 Pro, 20 GPU cores, 64 GB
(issue 8's kind of machine), `applegpu_g17s`, macOS 27.0.1, MLX 0.32.3, main at `2842c51` (so the engine's own
choice there was 8), float16, `MLX_ENABLE_TF32` unset. Q2_0, held whole, then IQ3_S, its experts from the SSD with
39.8 GB of them in memory; one file loaded at a time. Other work on the machine was paused for the run by the
author's own tooling.

One process, four engines sharing the weights: fused up to 6, fused up to 8, a control fused up to 6, and a
reference with pull request 9's eight kernel switches off, fused up to 6. At windows of 1, 4, 7 and 8 tokens after
a fixed prompt, 20 rounds, the four in a new random order each round; each pass timed, its logits hashed and their
top tokens kept. The figure is the median of the 20 paired differences, with a bootstrap 95% interval. Then three
rounds of eight prompts, the order shuffled per prompt: five that answer in JSON and are graded, prose, code and
Chinese; greedy, thinking left empty, at most 256 tokens.

These figures came from `bench/results/e54-m5pro/legacy-harness.py`, kept beside them with their summaries.
`bench/experiments/e54_shared_weight_controls.py` is its cleaned form (the experts' share worked out from the
machine's memory, a new output folder only, phases recorded, logits checked finite, the draft policy reset for each
request); it ran a small check only: Q2_0, windows of 1 and 8, two rounds, two graded prompts, 16 passes and 8 of 8
right (`bench/results/e54-cleaned-smoke`). `bench/compare_results.py` checks that a run has every planned sample
before it summarizes one.

## Result

Ms a pass, the median of 20; "8 slower by" is the median of the paired differences:

| file | tokens | fused up to 6 | up to 8 | 8 slower by | 95% | control's spread (\|p95\|) |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: |
| Q2_0 | 1 | 25.84 | 25.93 | 0.05 | -0.10 to 0.13 | 0.52 |
| | 4 | 36.41 | 36.44 | 0.06 | -0.02 to 0.09 | 0.33 |
| | 7 | 46.99 | 48.32 | 1.34 | 1.27 to 1.43 | 0.37 |
| | 8 | 48.55 | 50.29 | 1.71 | 1.58 to 1.84 | 0.24 |
| IQ3_S | 1 | 38.30 | 38.27 | -0.06 | -0.18 to 0.04 | 0.72 |
| | 4 | 55.13 | 55.17 | -0.01 | -0.24 to 0.33 | 1.02 |
| | 7 | 77.91 | 79.01 | 1.09 | 1.02 to 1.21 | 1.14 |
| | 8 | 80.82 | 81.97 | 1.19 | 0.95 to 1.39 | 0.99 |

- Bits: fused up to 8 gave other logits at seven and eight tokens with both files, and with IQ3_S other top
  tokens there too; the same at one and four. The control gave fused-up-to-6's bits everywhere.
- Pull request 9's switches off, a pass was slower at every size: 1.6, 3.1, 4.3 and 4.8 ms with Q2_0, 1.4, 2.1, 2.5
  and 2.4 with IQ3_S.
- Answers: the graded ones 12 of 15 right for every engine with Q2_0 (the invoice answered inside a ```` ```json ````
  fence every time) and 15 of 15 with IQ3_S. At the 256-token cap: the code prompt with Q2_0 (3 of 24 a setting),
  prose, code and Chinese with IQ3_S (9 of 24). The control's tokens differed from fused-up-to-6's in 3 of 24
  answers with Q2_0 and 4 of 24 with IQ3_S; in that run the draft policy carried its history from request to request.
- Memory, system-wide: with Q2_0 2.3 GiB swapped out while the draft head loaded, none during the passes and
  answers; with IQ3_S none. The n-gram table was not warmed with either file.

## Conclusion

- On this M5 Pro, fusing up to eight is 1.3-1.7 ms a pass slower at seven and eight tokens with Q2_0 (about 3%) and
  1.1-1.2 ms with IQ3_S (about 1.5%), and the same below: the opposite of the M5 Ultra, a generation 17 as well.
  The GPU's generation does not choose the limit alone. Its cores may (20 here, 80 there), but two M5s are not a
  rule.
- With Q2_0 the difference is 3.6 and 7.3 times the control's spread. With IQ3_S, whose experts come from the SSD, the
  control varied as much as the difference (1.0-1.1 ms), though the paired intervals stay above zero.
- Answers between engines are not the same work, the control's included, so their rates are not compared here.
- Merged with `--fusion-limit 6|8` to override the engine's choice; `auto`, the default, is still 8 from generation 17.

## Concepts

[[hyper-connections]] [[gpu-generation]] [[measurement-noise]]
