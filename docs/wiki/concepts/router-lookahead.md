---
id: router-lookahead
kind: mechanism
status: current
---
# Experts guessed before their router: not built

> 中文摘要:写回答时提前一层猜下一层要用哪些专家并预先读入。量过,最多 1.35-1.39 倍,低于事先定的 1.5 倍,没有做。
> 猜测只能把读盘挪到计算旁边,不会让读的字节变少;Strata 那边的同类实现也是持平或变慢。

## What it is

Guessing, from an earlier state, which experts a layer's router will choose, and reading them before it has run,
so an answer's token waits less for the disk. Measured and not built.

## In this engine

Nothing: the recording and the replay are `bench/experiments/e35_route_ahead.py`.

## Relations

- limited-by → [[ssd-reads]]: a guess moves reads beside the GPU's work and does not make them fewer; where the disk is two thirds of the time that is the most it can give [e35]
- depends-on → [[expert-store]]: what decides how much is read is how many experts stay in memory, not when they are asked for [e35]

## What we know

- A layer's router on the residual a layer ahead holds 77% of the ten chosen in its best ten, 89% in its best 16
  (with 107% more read for nothing) [e35].
- Replayed against a store: at most 1.35-1.39 times, against the 1.5 set before [e35].
- Elsewhere: slotstream measured 1.1 times for its lookahead with this model; in Strata's issues the same idea
  was even or slower in practice, and a trained predictor did not carry over between conversations.

## Open questions

- Strata's author spoke of small trained projection layers for it; no code or figures have been published.
