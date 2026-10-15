---
id: e43
script: bench/experiments/e43_experts_as_qmm.py
change: skxmzozu
date: 2026-10-14
---
# e43: the Q2_0 experts through MLX's own quantized product

> 中文摘要:把 Q2_0 专家的字节原样交给 MLX 的 gather_qmm(编码字节正好是 MLX 的 2 位格式),和查表内核比。更慢,没有采用。

## Question

Can MLX's `gather_qmm` replace the table kernel for answers? A Q2_0 block's 16 code bytes are MLX's packed 2-bit
words as they are, its scale d and bias -d, so the stacks need no conversion.

## Setup

Q2_0, M4 Max, the real model's 48 expert layers chained, random inputs and routes, windows of 1, 2, 4 and 8
tokens, lower quartile of 30 runs. Outputs checked against the table kernel (relative error 6e-4 to 2e-3).

## Result

| tokens | table kernel | gather_qmm |
| ---: | ---: | ---: |
| 1 | 4.30 ms | 4.62 |
| 2 | 5.57 | 7.56 |
| 4 | 11.14 | 16.46 |
| 8 | 30.62 | 47.75 |

## Conclusion

Slower at every size; not used.

## Concepts

[[expert-lookup-kernel]]
