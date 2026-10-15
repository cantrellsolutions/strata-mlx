---
id: prompt-chunk
kind: mechanism
area: prompts
status: current
---
# A prompt read in chunks

> 中文摘要:提示按 2,048 token 一块读,每块走完全部层,只为最后一个 token 算输出层。一块是几个毫秒级的大内核,
> 专家部分占一半左右的时间。读 1,180 token 的提示,本机比 llama.cpp 和 LM Studio 都快。

## What it is

A prompt goes through the model in chunks of 2,048 tokens, each through all the layers, the output layer for the
last token only. A chunk is a few kernels each milliseconds long, unlike a [[decode-pass]].

## In this engine

`strata_mlx/engine.py:Engine` (`read`, `chunk=2048`); where experts are read from the SSD and more than a chunk
is to be read, [[layer-major-prefill]] instead.

## Relations

- depends-on → [[expert-tiles]]: the ten chosen experts were 52-54% of a 1,024-token chunk before the tiles and 49-54% after [e30] [e33]
- depends-on → [[ngram-table]]: with the table cold an 8,192-token prompt took 13-14 s more [uoonprvo]
- costs → [[working-memory]]: a chunk of 2,048 peaks 4.6 GB over the weights; MLX's pool of freed buffers had no limit and held 37 GB after 2,048 tokens [ROADMAP: Done]
- depends-on → [[gpu-idle]]: a short prompt after a few seconds of nothing waited half a second for its first token [#1]

## What we know

Reading speed through the server, a 1,180-token prompt, Q2_0, M4 Max [ROADMAP: Against llama.cpp's server]:

| engine | tok/s |
| :--- | ---: |
| llama-server 0.4.1 | 494-495 |
| LM Studio 0.4.25 | 426-446 |
| this engine, as it serves | 570-578 |

- Prompts of 27-49 tokens are read slower than by llama.cpp (31-112 against 92-148 tok/s before pull request 7's
  fix for the idle GPU) [ROADMAP: Against llama.cpp's server].
- All else of a pass but the experts would read 1,200 tokens a second; the GPU does 14.3 T operations a second in
  one large product, and the experts' small products run at 4.9-6.1 [e30] [e33].
