---
id: reference-agreement
kind: method
area: precision
status: current
---
# Agreement with llama.cpp

> 中文摘要:用 llama.cpp 存下的答案作参照,逐步比较"模型认为最可能的 token"是否一致,通常 64 步里 63-64 步相同。
> 必须用模型自己会回答的提示:强行越过结束符后,同一文件的三种实现会差半个 nat,比较就失去意义。

## What it is

How a change that alters numbers is judged: the engine's top token compared, step by step, with llama.cpp's
saved answers to the same prompt and file, a token at a time and in windows.

## In this engine

`bench/compare_llama.py`; the results are cited in ROADMAP.md wherever a change touched the numbers.

## Relations

- part-of → [[exactness]]: what a change that is not bit for bit is measured by [ROADMAP: Done]
- depends-on → [[hyper-connections]]: rounding them to 8 bits kept llama.cpp's token at 64 of 64 steps on the chat prompt and 62 of 64 on the 7K-token one (64 before); at 4 bits 60 and 54 [e18]

## What we know

- Use a prompt the model answers. Forced past its own end token, three implementations of the same file disagree
  by half a nat and the comparison says nothing [ROADMAP: Done].
- Typical agreement: 64 of 64 after 1,537 and 3,161 tokens, 63-64 of 64 after 6,945, a token at a time or four a
  pass [ROADMAP: Why prompts are read slower].
- llama.cpp's own Metal and CPU backends differ from each other by about as much as this engine differs from
  either [ROADMAP: Now: what is left].
