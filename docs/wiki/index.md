# Index

> 中文摘要:所有概念的目录,按领域分组。目前只有"写回答的速度"这一块(试点)。

## Writing answers

- [A decode pass](concepts/decode-pass.md): one forward of a window; what tokens a second are made of.
- [The memory bandwidth floor](concepts/bandwidth-floor.md): the least a pass can take, from the bytes it reads.
- [The fixed cost of a kernel](concepts/kernel-fixed-cost.md): what thousands of small kernels cost beyond their work.
- [The draft window](concepts/draft-window.md): guesses checked in one pass, and how many to make.
- [Guess acceptance](concepts/guess-acceptance.md): the share of guesses kept, the content's more than the engine's.
- [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md): experts by a table of sums, and its threads.
- [Hyper-connections](concepts/hyper-connections.md): four residual streams mixed at every layer, 1.27 GB a token.
- [The n-gram table](concepts/ngram-table.md): 28.8 GB looked up by hashes, on the disk.

## Across areas

- [Exactness](concepts/exactness.md): the same bits for the same input; the rule every change keeps or reports.
- [The GPU's generation](concepts/gpu-generation.md): thresholds tuned on one chip are not settled for another.
- [Measurement noise](concepts/measurement-noise.md): why changes are compared in one process, passes shuffled.

## Experiments

- [e43: the experts through MLX's quantized product](experiments/e43-experts-as-qmm.md): slower, not used.
- [e44: threads to a row of the table kernel](experiments/e44-lookup-threads.md): 3-27% on the M4 Max.
- [e45: pull requests 9 and 10 checked on the M4 Max](experiments/e45-pull-request-9-check.md): merged, fused up to six.

Earlier experiments, e01-e42, are in [ROADMAP.md](../../ROADMAP.md) and `bench/experiments/`.

The relations as a diagram: [graph.md](graph.md).
