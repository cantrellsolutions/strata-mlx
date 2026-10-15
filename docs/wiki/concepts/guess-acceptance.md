---
id: guess-acceptance
kind: metric
status: current
---
# Guess acceptance

> 中文摘要:草稿头的猜测被模型接受的比例。它主要由内容决定:改写文件、推理高,散文低。
> 同一个引擎在不同内容上的接受率能差一倍,所以别人报告的接受率不能直接拿来比。

## What it is

The share of the head's guesses the trunk keeps, and with it the tokens a pass settles. It belongs to the text
being written more than to the engine.

## In this engine

Counted by `strata_mlx/drafting.py:Policy` as the run goes, by how sure the head was of each run of guesses.

## Relations

(Its relations are the ones that point at it: see [[draft-window]].)

## What we know

- Q2_0, M4 Max, 256 tokens, greedy, guesses as sure as the head is: prose 63-69% kept, code 59-69%, reasoning
  78-89%, rewriting a file 77-89% [e44].
- The head's sureness of a run of guesses predicts whether they are kept, well for answers that repeat or reason
  step by step, poorly for prose [e13].
- Elsewhere: oMLX reports 97% on Python code with long context, and 68.8% on another model page; the figure is
  the content's, not a target.

## Open questions

- Acceptance by content type was measured once in a session (prose 58%, code 81%, long code 88-92%, a rewrite
  100%, three guesses always) and not kept as a result file.
