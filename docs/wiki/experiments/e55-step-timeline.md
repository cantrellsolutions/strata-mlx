---
id: e55
script: bench/experiments/e55_step_timeline.py
change: sylqlors
date: 2026-10-20
---
# e55: where a step's time goes, and the bandwidth measured again

> 中文摘要:不改引擎,给写回答的每一步分阶段计时,并重新量带宽。带宽实测约 490 GB/s,5.5 GB 读一遍是 10.8 ms,
> 不是一直沿用的 13.9 ms,所以一次前向离下限比原来以为的更远:GPU 跑一次 1 个 token 的前向约 16.8 ms,比下限多约 6 ms。
> CPU 单独干活、GPU 空等的时间每步约 2.3 ms(不带草稿头)到 4 ms(带草稿头),约占一步的 10-12%;其中建图 1.8-2.1 ms 最大。
> n-gram 表读取几乎为零(表在缓存里)。结论:GPU 那边的 6 ms 比 CPU 这边的 2-4 ms 大,内核(实验 2)优先;流水线值 10% 左右。

## Question

How much of a decoding step is the GPU running, and how much is the CPU working with nothing submitted (building
the graph, committing, deciding guesses)? And what is the memory bandwidth, measured by a kernel that only reads,
against the 13.9 ms of `mx.sum` every floor in this knowledge base hangs on?

## Setup

M4 Max 128 GB, Q2_0 whole in memory, change sylqlors. Methods of `Engine` wrapped for timing in the script's process
only (the engine unchanged); every `mx.eval` in a step summed as the wait for the GPU. Four prompts of
`bench/spec_decode.py`, 160 tokens, greedy, two rounds; one engine a token a pass, one with the draft head.
Medians over steps. **Not a quiet machine**: the GPU was 49% busy with other programs before the run, two Emacs
processes each held a core. Results `bench/results/e55_step_timeline.json`.

## Result

The bandwidth, 5.5 GB in arrays of 512 MB:

| reader | ms | GB/s |
| :--- | ---: | ---: |
| `mx.sum` | 10.83 | 496 |
| a kernel reading 16-byte vectors, 40K to 2.6M threads | 10.88-11.25 | 477-493 |

A step, ms (median of the four prompts' medians, two rounds):

| engine | step | build | n-gram rows | wait for the trunk (`_choose`) | commit | draft head (of it waiting) | all waits | CPU with nothing waited on | tokens a step | tok/s |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| a token a pass | 19.0-20.2 | 1.75-2.02 | 0.03-0.07 | 16.8-17.6 | 0.39-0.45 | - | 16.8-17.6 | 2.25-2.58 | 1 | 49-53 |
| draft head | 35.5-50.0 | 1.83-2.11 | 0.07-0.11 | 27.0-42.8 | 0.38-1.26 | 3.8-6.6 (3.2-5.7) | 31.5-46.5 | 3.8-4.3 | 3.2-5.4 | 73-115 |

## Conclusion

- **The floor is 10.8 ms, not 13.9.** `mx.sum` itself reads at 496 GB/s here, and a read-only kernel does no
  better: the 13.9 ms of ROADMAP was taken on a slower day or another layout. Every "distance to the floor" in this
  knowledge base was too small.
- **The GPU's part of a one-token pass is about 16.8 ms**, 6 ms over the floor (the step's wait, which is all
  GPU with the graph already built). That is where most of the time to win is: kernels, not the CPU.
- **The CPU alone takes 2.3 ms a step** without the draft head (build 1.8, commit 0.4) and **about 4 ms** with it
  (building the head's graph and deciding guesses on top): 10-12% of a step. Overlapping the build with the GPU
  (async_eval in parts) can take at most that; worth doing, smaller than the kernels.
- The n-gram rows cost nothing while the table is in the page cache (a machine where it is not was e55's blind spot).
- The machine was busy: the GPU numbers may be high; the proportions and the bandwidth (which came out higher than
  ever measured here) stand.

## Concepts

[[bandwidth-floor]] [[decode-pass]]
