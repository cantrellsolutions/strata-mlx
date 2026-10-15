---
id: stopping
kind: mechanism
area: conversations
status: current
---
# Stopping a request

> 中文摘要:服务端发 STOP 时引擎在哪里停、停下后保留什么。逐块读时每块之后才能停(本机一块可达 8-13 秒);按层读时每一步都能停,
> 正在读的那一段精确撤销。`DONE` 行报告实际读完的 token 数。

## What it is

What the engine does when the server sends `STOP`: where it can see it, what stays read, and what `DONE` says.

## In this engine

`strata_mlx/serve.py` (the line protocol's `STOP` and `DONE`); a read layer at a time undoes the piece under way.

## Relations

- depends-on → [[layer-major-prefill]]: a stop is seen after every step (0.26-0.30 s) where chunk by chunk it is seen once a chunk; the piece under way is undone exactly, the pieces before it stay [e40]
- depends-on → [[recurrent-state]]: undoing a piece is possible because until its end only attention cache rows are written [e40]

## What we know

- `DONE` said a stopped read had read the whole prompt; it says what was read since [pkmuzlkl].
- Strata's server allows an engine 300 s of silence and resets the clock on any line; a progress line a layer
  keeps a long piece well inside it.
