---
id: expert-tiles
kind: component
area: prompts
status: current
---
# A prompt's experts in tiles

> 中文摘要:读提示时,专家按"16 个 token × 64 行"的小块用苹果的 matmul2d 原语计算,直接从文件的块格式解包,所有量化类型通用。
> 1,600 对以上走这条路,以下用查表或逐对的内核。配对排序也搬到了 GPU 上,省掉每层等 CPU 的时间。

## What it is

For a chunk of a prompt the experts are multiplied a tile at a time: a threadgroup takes 16 tokens and 64 rows of
the tile's expert, unpacks 64 columns of the file's blocks into threadgroup memory, and Apple's `matmul2d`
multiplies them. The pairs are laid out in tiles by array operations on the GPU, not sorted on the CPU.

## In this engine

`strata_mlx/blocks.py:tile_matmul`, `strata_mlx/blocks.py:tiles_of`, `strata_mlx/blocks.py:TILES_FROM`.

## Relations

- part-of → [[prompt-chunk]]: the ten chosen experts, half a chunk's time [e33]
- limited-by → [[kernel-fixed-cost]]: a tile costs the same with one pair as with sixteen, and 26% of the slots are filling at 1,024 tokens [e30] [e33]
- depends-on → [[gpu-generation]]: `matmul2d` is Metal 4's; on an M5 it runs on the neural accelerators, not measured here [ROADMAP: What the Apple GPU can do]

## What we know

- One product of 10,220 pairs on real layers: 4.5-5.3 ms whatever the type, 6.3-7.4 T operations a second [e33].
- Sorting the pairs on the CPU made a pass stop at every layer for the routers' choice: a 1,024-token piece took
  2,723 ms that way, 1,832 with the GPU laying out the tiles, 1,821 with the choice known beforehand [e32].
- A 1,180-token prompt: 555-560 tok/s with Q2_0 where it was 370-386 [e33] [nwoukvou].

## Superseded

- "Reading prompts is near what this GPU can do; the gap to Strata is the hardware's": it held for a third of the pass [e30].
