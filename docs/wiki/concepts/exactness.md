---
id: exactness
kind: constraint
area: precision
status: current
---
# Exactness

> 中文摘要:同样的输入是否给出逐位相同的输出。本项目把它当作硬约束:改速度的内核改动必须逐位不变,
> 否则要单独说明并量出差异。float32 下带草稿与不带草稿答案相同;float16 下约每 250 个 token 有一个落在近乎平局处的不同。

## What it is

Whether a change leaves the model's logits the same bits. The engine holds to it: a change for speed either
keeps every bit, or says where it does not and measures what that does to the answers.

## In this engine

The tests compare new kernels with MLX's operations bit for bit (`tests/test_fused.py`, `tests/test_blocks.py`);
`--dtype float32` is the exact path.

## Relations

- depends-on → [[gpu-generation]]: on an M5, MLX runs float32 products at reduced precision unless `MLX_ENABLE_TF32=0`, and 27 tests failed until it was set [#5]

## What we know

- In float32 the answers with and without drafts are the same. In float16 a window and a single token round
  differently, and the answer can part at a near-tie: about once in 250 tokens, each time at a token the model
  found nearly as probable [e29].
- Pull request 9's ten kernel changes keep the logits the same bits one and four tokens at a time [#9].

## Open questions

- Whether to offer a non-exact mode (8-bit [[hyper-connections]]) as an opt-in.
