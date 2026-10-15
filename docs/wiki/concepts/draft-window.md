---
id: draft-window
kind: mechanism
area: answers
status: current
---
# The draft window

> 中文摘要:草稿头先猜后面几个 token,模型在一次前向里把猜测连同当前 token 一起验证,猜对几个就一次定下几个。
> 窗口越长每次前向越贵,猜错的越多,所以猜几个由"猜中的把握"和"前向的耗时"共同决定。

## What it is

The draft head (the model's own multi-token-prediction layer) guesses the next tokens; the trunk checks them in
one pass over a window of up to eight tokens, and keeps the guesses up to the first wrong one. A window settles
more tokens a pass and costs more than a single token.

## In this engine

`strata_mlx/drafting.py:Policy` decides how many guesses; `strata_mlx/drafting.py:MAX_WINDOW` is eight;
`strata_mlx/mtp.py:Mtp` is the head.

## Relations

- depends-on → [[guess-acceptance]]: three guesses in any case, then two more at a time while the head's sureness says they will be kept [e13]
- trades-against → [[decode-pass]]: a window of four costs about 1.5 times a token, of eight about 2.4 times on the M4 Max (31.6 and 49 ms against 20.8) [e45]
- tuned-by → [[gpu-generation]]: which window lengths take which kernels was set on one GPU and moved on another [e45] [zryukqou]

## What we know

- Three guesses then more while sure, against three always, replayed: 6% more tokens a second; measured: 2% on
  prose, 5% on code, 6% on reasoning, 10% rewriting a file [e13].
- With the head, Q2_0 writes 74-100 tok/s against 41-45 a token at a time (M4 Max, 256 tokens, greedy; a slow
  machine that day) [e44].

## Open questions

- Code settled 5.2-5.9 tokens a pass with seven guesses always against 4.2-4.5 as sure, in a run that could not
  time the passes; whether deeper windows are faster for code is not known.
