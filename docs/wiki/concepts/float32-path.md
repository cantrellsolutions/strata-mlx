---
id: float32-path
kind: mechanism
area: precision
status: current
---
# The float32 path

> 中文摘要:`--dtype float32`:全部用 32 位浮点计算,带草稿与不带草稿答案逐 token 相同,是本项目的"精确路径"。
> 代价是速度:Q2_0 慢一半(查表内核只有 16 位版本),IQ3_S 慢一到两成。

## What it is

Running the model in float32 instead of float16: the answers with and without drafts, parked or not, are then the
same token for token, which makes it the path every exactness test is held to.

## In this engine

`--dtype float32` in `strata_mlx/serve.py`; the Q2_0 table kernel is float16 only (`strata_mlx/blocks.py:lookable`).

## Relations

- trades-against → [[decode-pass]]: half the speed with Q2_0 (44-49 tok/s against 80-103), a tenth to a fifth with IQ3_S (53-73 against 69-80) [ROADMAP: Now: what is left]
- depends-on → [[tf32-on-m5]]: on an M5 it is float32 only with MLX's reduced precision turned off [#5] [ozxszktq]
- part-of → [[exactness]]: the reference that float16 is compared against [e29]

## What we know

- In float32 with drafts and the guess list the engine wrote the model's own answer in all eight answers [e29].
