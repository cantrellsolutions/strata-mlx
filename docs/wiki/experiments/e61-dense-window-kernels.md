---
id: e61
script: bench/experiments/e61_q3k_window_kernel.py
change: yrkwxyqy
date: 2026-10-21
---
# e61: window kernels for the dense tensors at the file's bits

> 中文摘要:D3 的第一个内核。把 Q3_K 块在加载时重排成对齐的平面(高位、2 位码、16 个缩放各一个数组,字节数基本不变),
> 一个线程一次处理半块。在 18 个真实的 10240×2560 Q3_K 张量上,1/2/4/8 个 token 都比 MLX 的 8 位快(65-67/64/72/109 对
> 80/81/83/116 us),结果也更接近解码出的权重。IQ4_XS 的平面内核因为 16 个码本值要查表,1-2 个 token 时不如现有的块内核
> (e36 的,60/65 对 MLX 84/85 us),4-8 个 token 时和 MLX 持平。两种内核都离文件字节的下限很远(23-28 us):受解码计算限制。
> 按张量数估算,整次前向 1 个 token 约省 1.3 ms、4 个 token 约 0.7 ms,低于原先估计的 2-2.5 ms。

## Question

D3 holds the dense Q3_K and IQ4_XS tensors at the file's 3-4 bits where they are now 8-bit arrays (e36, e58).
Can a kernel for windows of 1 to 8 tokens be faster than MLX's 8-bit product at every size (e36's was slower
from four tokens)?

## Setup

M4 Max 128 GB, change yrkwxyqy. The 18 Q3_K and 13 IQ4_XS attn_qkv tensors (10,240 x 2,560) of the Q2_0 file, each
type chained (a tensor's input waits on the one before's output). MLX's 8-bit form as the engine holds it
(float16 scales); e36's `blocks.window_matvec`; and planes: the blocks rearranged once into aligned arrays (Q3_K:
high-bit masks, 2-bit codes, the 16 scales less 32 as int8, d; IQ4_XS: 4-bit codes, the 8 scales less 32, d),
a thread taking half a block (128 weights), P threads to a row. Shuffled, 60 rounds, the lower quartile.
Results `bench/results/e61_dense_window_kernels.json`.

## Result

Microseconds a tensor:

| tokens | Q3_K: MLX 8-bit | e36 blocks | planes, P=2 | planes, P=4 | IQ4_XS: MLX 8-bit | e36 blocks | planes, P=2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80.5 | 81.1 | 67.3 | 65.1 | 84.0 | 60.2 | 69.2 |
| 2 | 81.0 | 74.2 | 64.2 | 69.0 | 85.3 | 65.2 | 72.6 |
| 4 | 83.2 | 92.6 | 71.8 | 79.5 | 87.9 | 84.6 | 81.6 |
| 8 | 115.8 | 134.4 | 109.2 | 131.2 | 120.7 | 127.7 | 113.6 |

Bytes at 490 GB/s: 53.5 us as 8-bit, 23.0 (Q3_K) and 28.4 (IQ4_XS) as the file's blocks. Against the weights
unpacked, the largest difference was 2.3-2.7e-4 of the largest value for every block kernel and 0.8-1.2e-3 for
MLX's 8-bit.

## Conclusion

- **Q3_K as planes is faster than MLX's 8-bit at every size**: 13-17 us a tensor at one and two tokens, 11 at
  four, 7 at eight. IQ4_XS: e36's kernel at one and two tokens (20-24 us less), planes at four and eight (6-7 us).
- **None of them is near the file's bytes**: 65-110 us against 23-28. They are bound by decoding; e56's lesson
  (the bytes' reading) is not what limits these, which read 3-4 bits a weight against the experts' 2.
- Scaled by the tensors of each type in a pass (my estimate, other shapes not measured): about 1.3 ms of a
  one-token pass, 1.4 at two, 0.7 at four, 0.5 at eight; and 1.7 GB of memory. Less than the 2-2.5 ms hoped.

## Concepts

[[dense-blocks]]
