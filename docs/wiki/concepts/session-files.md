---
id: session-files
kind: component
area: conversations
status: current
---
# A conversation on disk

> 中文摘要:`SAVE` / `RESTORE`:把一段对话的全部状态存成一个文件,1,515 token 约 264 MiB,0.6 秒存完。
> 换了模型文件存的会被拒绝。用的是服务端的 `/slots/0?action=save|restore`。

## What it is

A conversation's state written to one file and read back: the caches, the recurrent state, the checkpoints.
Written beside its place and renamed when whole; refused if it was saved with other model files.

## In this engine

`strata_mlx/engine.py:save`, `strata_mlx/engine.py:restore`; the line protocol's `SAVE` and `RESTORE`.

## Relations

- depends-on → [[recurrent-state]]: 1,515 tokens are 264 MiB, about 45 MB of them caches and the rest the state and its checkpoints [ROADMAP: Done]

## Open questions

- Restoring from disk against reading the prompt again: where one becomes faster than the other is not measured.
