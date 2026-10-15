---
id: checkpoints
kind: mechanism
area: conversations
status: current
---
# Checkpoints

> 中文摘要:在消息边界等位置把递归状态复制一份,后续请求从最近的检查点接着读,只读新的部分。系统提示结尾那个"根"检查点一直保留,
> 其余按最久未用淘汰;`ckpt=0` 表示这个请求不会被延续,`pin=N` 表示前 N 个 token 是要保留的前缀。

## What it is

Copies of the [[recurrent-state]] taken at places a later prompt may part from this one: where the last two
messages begin, every 16,384 tokens, before a prompt's last token, and where the system prompt ends (the root).
A follow-up goes on from the latest checkpoint at or before the point where it parts from what was read.

## In this engine

`strata_mlx/engine.py:_checkpoint`, `strata_mlx/engine.py:resume`, `strata_mlx/engine.py:PERIODIC`,
`strata_mlx/engine.py:ROOT`; request options `ckpt=0` and `pin=N`.

## Relations

- depends-on → [[recurrent-state]]: 110 MiB each [yumvopms]
- depends-on → [[layer-major-prefill]]: read a layer at a time, each layer's state is kept as it passes a checkpoint's place and the checkpoints are put together at the piece's end, the same bit for bit [e40]

## What we know

- Kept as Strata keeps them: the root stays, the others leave by least recent use, the one before a prompt's last
  token first [yumvopms].
- The engine kept its eight newest before: a chat of several turns, or eight newer checkpoints, lost the system
  prompt's [ROADMAP: Done].
