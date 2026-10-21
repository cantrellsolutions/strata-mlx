---
id: e60
script: bench/compare_llama.py
change: oyvuuyuk
date: 2026-10-21
---
# e60: the dense tensors as the file's blocks, against llama.cpp

> 中文摘要:D3 动手之前先测一致率。用 llama.cpp 对同一个 Q2_0 文件在 1,537、3,161、6,945 token 的提示后各写 64 个 token 作参照,
> 再让引擎分别用现在的 8 位数组和文件块格式(现有内核)走一遍。引擎的一致率:8 位数组 62-63/64,文件块 62-64/64,
> 合计 374 对 378 of 384(一次 1 个和一次 4 个 token 各一半);和 llama.cpp top-5 对数概率的最大差也在同一量级。
> 按文件格式存放稠密张量不损害一致率,D3 可以做。

## Question

Before D3 (the dense tensors held at the file's 3-4 bits with a kernel for windows): does holding them as the
file's blocks, the weights the same and the sums in another order, move the engine away from llama.cpp?

## Setup

M4 Max 128 GB, Q2_0, change oyvuuyuk. llama-server 0.4.1 (build 10964) on the same file, greedy, 64 tokens after
the first 1,537, 3,161 and 6,945 tokens of ROADMAP.md, saved (`bench/results/e60/llama_ref_*.json`) and the
server stopped (the two do not fit together). Then `bench/compare_llama.py --dense arrays|blocks --windows 1 4`:
the model's own passes and the engine a token and four tokens a pass; "blocks" with the kernels of today
(`dense.BlockLinear`). Whole output `bench/results/e60/agreement.txt`. (llama-server must run with
`MTL_DEBUG_LAYER=0`: with Xcode's Metal debug layer it aborts on an assertion at the first long prompt.)

## Result

Top token equal to llama.cpp's, of 64 steps, the engine (1 a pass / 4 a pass):

| prompt | dense as 8-bit arrays (now) | dense as the file's blocks |
| ---: | :--- | :--- |
| 1,537 | 63 / 63 | 62 / 62 |
| 3,161 | 62 / 62 | 64 / 64 |
| 6,945 | 62 / 62 | 63 / 63 |
| all | 374 of 384 | 378 of 384 |

The largest gap of log-probability among llama.cpp's top five, engine: 0.83-3.06 as arrays, 0.53-1.33 as blocks.
The model's own passes (one pass, token by token): 63-64 as arrays, 62-64 as blocks.

## Conclusion

- **Holding the dense tensors as the file's blocks does not cost agreement with llama.cpp**: 378 of 384 against
  374, the difference within what one prompt moves. Their sums in another order are noise of the same size as
  the noise already there (the engine and llama.cpp part at a near-tie a few times in 64 either way).
- D3 can go on: the kernel to write must be at least as fast as MLX's 8-bit product from four tokens on (e36's
  was slower), and its agreement is to be measured again the same way, since its order of sums will differ again.

## Concepts

[[dense-blocks]] [[reference-agreement]]
