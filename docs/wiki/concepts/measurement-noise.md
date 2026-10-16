---
id: measurement-noise
kind: method
area: across
status: current
---
# Measurement noise

> 中文摘要:这台机器的计时会因为别的程序占 GPU、机器发热而漂移,两次运行之间能差一倍。所以比较两种做法时,
> 要在同一个进程里交替(每次前向随机选一种),而不是分两次运行各跑一遍。

## What it is

What moves a timing besides the change being measured: other programs on the GPU, the machine slowing as it
heats, where the weights land in memory. On this machine it is often larger than the change.

## In this engine

The yardstick `bench/experiments/e05_pass_rate.py` alternates one-token and four-token passes in one process.
A comparison of settings: an engine for each sharing the weights, passes shuffled, and a control engine.

## Relations

- costs → [[decode-pass]]: other programs on the GPU move a pass by half within minutes; the same version timed 19.4 to 30.8 ms a token in separate processes on one afternoon [ROADMAP: Faster answers] [e45]

## What we know

- Compared in turns by process, three versions could not be told apart: another program kept the GPU 18-76%
  busy. In one process, an engine for each setting sharing the weights and the passes shuffled, they could [e45].
- Separate processes differ by about 1 ms a pass on an M5 Ultra, in two clusters [#9].
- A control: one more engine with the same setting as the baseline, in the same shuffle. What it differs from the
  baseline by is what the run cannot tell apart. On the M5 Ultra it was 0.00-0.13 ms, so the 2.0 ms between two
  fusion limits held; a run without one had found up to 4 ms between two settings at window sizes where both ran
  the same code, and was thrown away [#9].
- Whole answers are not the same text from run to run even greedy, so tokens a second through whole answers
  compare less cleanly than passes [#9] [e44].

## Open questions

- A measurement under held memory needs its own check that the hold held: one that did not passed for whole by
  its footprint [e46] [e47].

- A system-wide count (swap-outs) cannot say which process paid; a run under held memory with other programs open
  measures the machine as it is that day [e46].

- A script for the one-process comparison that any change can use, with the control built in, instead of one
  written per experiment.
