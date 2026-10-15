---
id: recurrent-state
kind: concept
area: conversations
status: current
---
# The recurrent state

> 中文摘要:模型的 36 个 delta-net 层各有一份递归状态,一共约 110 MiB,和对话长度无关。它不能截断到任意位置,
> 只能回到存过检查点的地方。这是检查点、对话停放和按层读取消都要围着它设计的原因。

## What it is

The delta-net layers carry a state from token to token instead of a cache of every token: 110 MiB in all,
whatever the length. Unlike the attention caches it cannot be cut back to an earlier token; only a copy taken
there gives it back.

## In this engine

`Engine.gdn` in `strata_mlx/engine.py:Engine`, with the convolution's history (`ple_conv`) and `context`.

## Relations

- costs → [[conversation-parking]]: every checkpoint and the state a parked conversation stopped in are 110 MiB each [yumvopms]
- depends-on → [[checkpoints]]: the only way back to an earlier token [ROADMAP: Done]

## What we know

- Why a prompt read a layer at a time is undone whole when stopped: until a piece's end only rows of the attention
  caches have been written, and those are cut back in 0.01 ms [e40].
- The block-level reuse of other engines (vLLM, oMLX) needs a state that can be sliced by token; this one cannot.
