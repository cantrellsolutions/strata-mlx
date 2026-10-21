---
id: expert-lookup-kernel
kind: component
area: answers
status: current
---
# The Q2_0 experts' table kernel

> 中文摘要:写回答时 Q2_0 专家的乘法内核。它不逐个乘权重,而是先把每两个输入能组成的 16 种和算成一张表,
> 再按 2 位编码查表相加。e56 推翻了"受算术限制":时间几乎全在读字节的方式上;1 个 token 时是小内核读几 MB 达不到带宽,4-8 个 token 时离下限 1.4 倍。

## What it is

A Q2_0 weight is -1, 0, 1 or 2 times its block's scale, so two weights of two inputs make one of 16 sums. The
kernel builds that table of sums for every pair of inputs and adds table entries by the weights' 2-bit codes:
about three operations a weight instead of eight.

## In this engine

`strata_mlx/blocks.py:lookup_matvec`, `strata_mlx/blocks.py:lookup_table`, and since pull request 9
`strata_mlx/blocks.py:lookup_down_glu`, which builds the inner activations' table inside the down product.

## Relations

- part-of → [[decode-pass]]: routed experts are 4.7 ms of a one-token pass and 12.3 of a four-token one [e06]
- tuned-by → [[gpu-generation]]: threads to a row: 8 up to 40 pairs and 2 beyond is faster on the M4 Max, slower in a full pass on an M5 Ultra [e44] [e45] [#9]
- limited-by → [[kernel-fixed-cost]]: the table for the down product was its own kernel until pull request 9 built it inside down, 0.35 ms a pass on an M5 Ultra [#9] [wlqvqpzr]
- limited-by → [[bandwidth-floor]]: at four and eight tokens 1.4 times the floor, and as planes read side by side its bytes reach it [e56]
- depends-on → [[exactness]]: a sum through the table carries about one more float16 rounding than a product; any change to how rows are split across threads changes the bits [e44]

## What we know

- Bound by how its bytes are read, not by arithmetic: reading the same bytes and doing nothing is 86-94% of the
  gate product at 1 / 4 / 8 tokens, and a uniform table index changes nothing. At one token any kernel reading the
  4.6 MB of a layer's ten experts is 2.5 times the floor; at four and eight tokens the kernel is 1.4 times it, and
  codes read as a plane side by side reach the floor (1.0) and, with the same table, a product 7-9% faster [e56].
- MLX's own quantized gather product on the same bytes (a Q2_0 block's code bytes are MLX's 2-bit words; scale
  d, bias -d) is slower: 4.6 / 7.6 / 16.5 / 47.8 ms for 48 layers at 1 / 2 / 4 / 8 tokens against 4.3 / 5.6 /
  11.1 / 30.6 [e43].
- Threads to a row for the rows of 2,560 inputs, swept and alternated: 3% faster at one token, 8% at four, 17-27%
  from eight on the M4 Max [e44]; on top of pull request 9, 0.4 ms less a pass at four tokens and 3.0 at eight
  [e45] [zryukqou].

- Gate and up in one kernel (`blocks.lookup_gate_up`, behind `GATE_UP_ONE_KERNEL`, off): the same bits as two
  (`tests/test_blocks.py`); its time not measured yet [qmvxvpnt].

## Superseded

- "Bound by arithmetic, not bytes: about eight operations a weight at all the GPU has, whatever the inner loop,
  threads to a row or vector loads, until the table cut it to three" [e02] [e03] [e10] [e11] [e12]: changing the
  arithmetic left it unchanged because it waits on its reads; the bytes alone take nearly all its time [e56].

- "The down rows want 2 threads throughout" [skxmzozu]: since pull request 9 the down rows' table is built in
  `lookup_down_glu`, whose bits the separate kernel must give, and they keep the old rule [zryukqou].
