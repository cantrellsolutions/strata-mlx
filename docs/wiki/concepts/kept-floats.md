---
id: kept-floats
kind: component
area: prompts
status: current
---
# The floats kept for prompts

> 中文摘要:Q6_K 层读提示时要先解包成浮点数。原来每块都重新算,慢一成;现在第一次读提示时算好并保留:IQ3_S 多占 3.3 GB,
> Q2_0 只多 0.2 GB。PR 7 之后内存规则把它算进去,只在装得下时保留。

## What it is

The layers of type Q6_K (1.88 GB in the IQ3_S file, 84 MB in Q2_0) are multiplied, for a chunk of prompt, from
floats. Made anew for every chunk they cost a tenth of the reading speed; so they are made once, at the first
prompt, and kept.

## In this engine

`strata_mlx/dense.py:ScaledLinear`, `strata_mlx/dense.py:kept_floats`.

## Relations

- costs → [[memory-rule]]: 3.3 GB more held with IQ3_S, 0.2 with Q2_0; the rule did not count them until pull request 7, and keeps them only where the model and they fit [#3] [wysxmzsy]
- part-of → [[prompt-chunk]]: made anew for each prompt they cost a tenth of the reading speed (320 against 355-369 tok/s) [ROADMAP: Against llama.cpp's server]

## What we know

- Where they do not fit they are made for each chunk, as before: prompts about a tenth slower [wysxmzsy].
