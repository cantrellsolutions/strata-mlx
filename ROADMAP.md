# Roadmap

## Done

- The model straight from Strata's GGUF files, without changing a weight (`strata_mlx/gguf.py`, `load.py`).
- Experts computed on the file's own blocks (`blocks.py`): nine block types, each checked against gguf-py's
  dequantizer on random blocks. Nothing is converted at start (22 s to load Q2_0, was 34-57), and in a pass the
  kernels are as fast as MLX's own 2-bit gather or a little faster (43 tok/s a token at a time, was 38-40), so
  Q2_0 uses them too. A prompt of 500 tokens or more unpacks each stack to floats for the length of one product.
  When comparing with llama.cpp, use a prompt the model answers: forced past its own end token (`ignore_eos`),
  three implementations of the same file disagree by half a nat and the comparison says nothing.
- Draft-and-verify with the model's draft head, for greedy and for sampled answers (`engine.py`, `mtp.py`,
  `sampler.py`): a seed writes the same text with and without drafts.
- Guesses copied from earlier in the conversation, used where they settle more tokens per millisecond
  (`drafting.py`).
- A conversation read once: checkpoints at message boundaries, a follow-up reads only what is new.
- Prompts read in chunks, the output layer for the last token only.
- Strata's line protocol (`serve.py`, `./strata-mlx`): its server, web app and OpenAI / Anthropic APIs run on this
  engine unchanged.
- The sparse-attention path past 2,048 tokens, checked against llama.cpp at 2,100 and 6,000 tokens.
- Long conversations (`attention.py`, `bench/long_context.py`). The port attended to every cached cell under a
  mask and pooled the indexer's keys of the whole conversation again each pass; with several queries MLX's
  attention kernel then spreads the two key heads over the 24 query heads, 1.6 GB a layer at 128K cells. Now a
  pass gathers the 2,052 cells each query chose, and a block's pooled key is computed once. With the caches
  filled to a length (Q2_0, M4 Max; before -> now):

      cells cached    one token a pass    draft head, 4 a pass    a 2,048-token chunk     its peak above rest
          8K           26-31 -> 29 ms        69-71 -> 59 ms        260-409 -> 405 tok/s      3.8 -> 3.8 GB
         32K              32 -> 27             105 -> 56               288 -> 321            5.7 -> 3.8
         64K              36 -> 28             166 -> 59               219 -> 306            8.9 -> 4.1
        128K              45 -> 29             285 -> 60               115 -> 280           15.7 -> 5.0

  With the draft head a pass of four tokens at 128K was slower than four passes of one; now the head pays at
  every length. (A later run of the same, a browser busy on the GPU: 30-32 ms, 63-83 ms and 172-233 tok/s, at
  every length alike.) A chunk gathers from 12K cells on (8.6 GB of keys and values copied for 2,048 queries, a
  layer; under the mask the cost grows with the length, and inside a real read faster than measured alone). The
  queries of a chunk do not choose the same blocks, so there is no smaller set to gather once.
  Against llama.cpp on chat prompts of 3,161, 6,945 and 32,299 tokens: its token at 64 of 64 steps, a token or
  4 or 8 a pass. The 32,299-token prompt is read at 154-246 tok/s over five runs (llama.cpp on Metal: 267; the
  6,945-token one at 320-436) and the passes after it start at 31 ms, as with caches filled; llama.cpp writes
  11.5 tok/s there, 33-35 after 3K and 7K tokens. That evening other programs kept the GPU busy: 400 passes in
  a row took 29-65 ms by twenties at 2K cells and 37-51 at 32K, so the slower 32K answers measured (22-34 tok/s
  a token at a time) say nothing about the length.
- MLX's pool of freed buffers has a limit of 4 GiB (`load.POOL`). It had none: a chunk of prompt frees buffers
  of sizes the next layer does not ask for again, 37 GB after 2,048 tokens and 65 GB after 7,000, beside 41 GB
  of weights. With other programs' memory that left the machine swapping: a 7K-token prompt read at 60-100
  tok/s and a six-token chunk took six seconds. With the limit it reads at 320-415 tok/s, and the first long
  prompt after a start as fast as the next (it took twice as long: the pool being filled). Passes are the same
  at 1, 4 and 16 GiB and with no limit.
  Not done: the caches are 30 KB a token (3.9 GB at 128K) in 16-bit floats; at 8 bits they would be about half,
  which matters to the machines that page experts, whose budget does not count the conversation at all.
- A conversation on disk (`SAVE` / `RESTORE`, the server's `/slots/0?action=save|restore`): one file, written
  beside its place and renamed when whole, refused if it was saved with other model files. 1,515 tokens are
  264 MiB, saved in 0.6 s.

## Now: what is left of Strata that is worth having on a Mac

- **The larger model sizes.** IQ3_S runs: 58 GB of weights (60 with the draft head; 84 at the peak of loading,
  72 while reading 8,192 tokens), 32 tok/s a token at a time, 51-64 with the draft head, 83 rewriting a file,
  about 440 tok/s reading a 1,500-token prompt. On a chat prompt it picks llama.cpp's token at 63 of 64 steps
  and its log-probabilities are closer to either llama.cpp backend (median 0.05 and 0.09) than those are to each
  other (0.09); llama.cpp on Metal writes 33 tok/s from the same file. All six expert types of the file unpack
  to gguf-py's weights exactly. IQ3_XXS and IQ2_XS will not be run: there is no disk space for more model
  files. Their tensors all have a path through the loader and their block types pass the same tests.
- **The same answer with and without drafts at full speed.** True in float32 (about 48 tok/s); in float16
  (56-105 tok/s) a window and a single token round differently and the answers can part at a near-tie.
- **Prompt reading speed.** About 440 tok/s for a 1,515-token prompt with the draft head (was 220; the first
  prompt after a start took four times as long until the loader's leftover buffers were released), 240-330 for
  8,192 random tokens. For the ceiling see the section on the GPU below.
- **The fixed cost of a pass.** A single-token pass is now 24 ms (was 28), a four-token window about 40 (was 43-47):
  layer halves compiled, hyper-connections as two Metal kernels for windows of one or two tokens. What was learned
  on the way: a pass is about 3,300 kernels in 229 command buffers, the GPU is busy 77% of it, and MLX's
  command-buffer limits change nothing. Cutting the kernel count by 43% bought 15%, because the kernels that are
  easy to merge are the cheap ones; hand-written kernels were slower than MLX's matrix products from three tokens
  up. No single large target is left: the expert gather already reads at 130-310 GB/s inside a pass. Getting
  near Strata's numbers would take a pass of a few hundred large kernels, each as good as MLX's own. The block
  kernels showed the same thing again: three times faster than MLX's gather alone, 6-10% faster in a pass.
  (Superseded: see "Faster answers" below. The pass was near the floor of the bytes it reads, and "no single
  large target" was wrong: the dense weights are seven times the experts' bytes.)
- Pictures (`GENI`), several conversations decoded together (`BGEN`).

## Faster answers: experiments (bench/experiments, results in bench/results/e*.json)

The yardstick is `e05_pass_rate.py`: passes of one token and of four, alternated, lower quartile. On a quiet
machine it repeats within 1%; changes are compared in one process, the two ways alternated.

    step                                           one token    four tokens    code / edit with the draft head
    start                                           22.3 ms       35.8 ms         69-70 / 77 tok/s
    hyper-connections read once for a window        22.1          34.0            72.8 / 81.9
    Q2_0 experts by table lookup                    22.1          31.5            74.9 / 85.3

What the measurements said, in the order they corrected each other:

- **A pass is not thousands of small kernels that cost by their number.** A token reads 5.5 GB of weights: 4.87
  of dense tensors (hyper-connections 1.27, delta-net projections 2.1, attention 0.6, output layer 0.5) and 0.66
  of experts. Summing those arrays once takes 13.9 ms; a one-token pass waits 20 ms for the GPU. An earlier note
  here put the bytes at 2 GB and the floor at 5 ms, and three rounds of merging kernels bought 15%.
- **Parts timed alone do not add up to a pass** (e01, e04, e05). MLX commits its command buffer after every
  50 MB of inputs, 23 us each time, so a kernel with a 236 MB stack of experts as input costs 25 us alone; in a
  pass the GPU is what is waited for and the limits (20 to 500 operations, any size) change nothing. Parts are
  measured by leaving them out of a real pass (e06): of 23.8 / 38.1 ms for one / four tokens, routed experts
  4.1 / 12.6, delta-net projections 6.0 / 8.7, hyper-connections 5.1 / 8.1, attention projections 2.3 / 3.9,
  output layer 2.3 / 3.1, routers 2.0 / 2.7, shared experts 1.8 / 2.5.
- **One GPU thread is slow, about 13 ns an instruction; a kernel takes as long as its longest thread** (e08,
  e09). A matrix of 324 by 10,240 times four tokens: 38 us with 32 threads to a row, 21 with 1,024. A norm
  kernel of 128 threads looping 160 times: 21 us for nothing. An array local to a thread: six times slower.
  From this the hyper-connections' three kernels (fused.py).
- **The expert kernel is bound by arithmetic, not by bytes or loads** (e02, e03, e10, e11): about eight
  operations a weight at all this GPU has, whatever the inner loop, the threads to a row, four values to a load
  or rows sharing their activations. A table of the 16 sums two weights make of two inputs is three operations
  a weight; a table of 256 sums for four weights does not stay in a cache and is four times slower (blocks.py,
  `lookup_matvec`; e12).
- MLX's own switches (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`,
  `MLX_BFS_MAX_WIDTH`) change nothing.

Not yet tried: the dense tensors as the file's blocks (Q3_K, IQ4_XS and Q6_K are held at 8 bits or as floats,
1.5 GB a token more than the file has); the routers as one kernel; a smaller output layer for the draft head's
guesses (it reads the 477 MB one three times a pass); how many guesses to make by how sure the head is.
Changes weights, so not done: the hyper-connections' 1.27 GB of 16-bit floats at 8 bits.

## What the Apple GPU can do, measured (M4 Max, macOS 27)

Looked into because prompts are read three to six times slower here than by Strata on an RTX 5070, and the
question was whether frameworks leave the Apple GPU's speed unused. They do not, on this generation:

- Metal 4 has a matrix product for use inside a shader (`mpp::tensor_ops::matmul2d`, Metal Performance
  Primitives), with int8 x int8 -> int32 and with weights packed at 4 or 2 bits as an operand. It builds and runs
  inside `mx.fast.metal_kernel`. At 4096 x 4096 x 4096 (`bench/gpu_matmul_types.py`, results checked against
  numpy): half 14.3 T operations a second, float 9.0, int8 12.8, half x packed 2-bit 13.4, int8 x packed 2-bit
  11.8. MLX's own matmul: half 14.3-14.7, float 12.3. So the integer forms are no faster than half on this GPU,
  and MLX already runs at what Apple's own primitive reaches.
- An earlier note put this GPU's ceiling for reading prompts at 650-750 tok/s, from 8-9.5 TFLOPS measured at
  2048^3. At the larger size it is 14.3: about 1,100 tok/s if every product ran at that rate, which the small
  per-expert ones do not.
- The hardware, from the reverse-engineered instruction set (M1 generation; nothing public for M4): scalar,
  32 threads to a SIMD group, fused multiply-add in 32 and 16 bits, integer multiply-add and bit-field
  instructions, no dot-product instruction. Apple's guide says the primitive uses "GPU neural accelerators in
  the Apple M5 chip"; MLX and llama.cpp both ship kernels built on it and switch them on by GPU generation.
  Not tested: there is no M5 here.
- The CPU has SME2 with 512-bit vectors (sysctl). Accelerate's sgemm measures 3.3 TFLOPS here, a quarter of the
  GPU; a paper measures 8 T int8 operations a second from two threads on an M4 Pro.
- For this engine, built (`blocks.packed_matmul`): a prompt's Q2_0 experts go through the primitive as 2-bit
  codes. The scale of a Q2_0 block sits inside its 18 bytes and differs every 64 weights, so the stack is split
  into a plane of codes and one of scales (1.3 ms, 0.24 GB instead of 1.7 unpacked), the product is taken 64
  weights at a time into registers and each part scaled as it is added. In tiles of 16 tokens by 64 outputs it
  runs at 10.5 T operations a second (tiles of 64 tokens: 4.6). Real model, back to back after a warm-up: a
  700-token prompt 346 against 281 tok/s, 1,537 tokens the same 430-446, 4,611 tokens 367-424 against 280-360,
  8,192 random tokens 218-365 against 162-342 (this machine's timings move that much between rounds; in every
  pair the packed way was the faster or equal). The peak of memory above rest while reading: 2.5-4.3 GB against
  7.4-11.4. Same tokens as before against llama.cpp (64/64 and 63/64). Read from the SSD on a small machine it
  replaces the pair-at-a-time kernel: 136, 146 and 159 tok/s for the 1,537-token prompt as a 16, 24 and 32 GB
  machine (were 103, 124, 135); most of that time is reading every layer from the disk.
  Not for float32 (the primitive has no float32 by 2-bit product) nor for the codebook types of the larger files,
  which still unpack to floats.
- None of this touches how fast answers are written: a token's products are matrix-vector, bound by memory and
  by the number of kernels in a pass.

## Macs with 16, 24 and 32 GB: a first version runs

On a machine whose memory does not hold the model, only part of the experts stays in memory, the least recently
used dropped first; a layer waits for its router, reads what is missing from the SSD and goes on
(`strata_mlx/paged.py`). How much stays is worked out from the machine's memory: 4.5 GB are left to macOS and
other programs, 4 GB to what this process needs beyond its weights, the dense weights and the draft head take
7.4, the experts get the rest. `--expert-memory <GB>` sets it by hand; `--machine-memory <GiB>` runs as a
machine of that size would, which is how these were measured, on the M4 Max with Q2_0, the engine process driven
over the line protocol with the draft head, the file dropped from the page cache and reads going past it:

    machine    experts in memory    process (peak)    answers           1,537-token prompt
     16 GB          1.5 GB          10-11 GB (12)      6-7 tok/s         136 tok/s
     24 GB         10.1 GB          18-19 GB (21)     19-23              146
     32 GB         18.7 GB          27 GB (29)        28-47              159
    128 GB       all, 34 GB         41 GB             64-83 (116 rewriting a file)   440

Tests hold it to the same tokens as the model with every expert in memory, at a budget of nothing, of five
experts and of all, and with a window spread over several kernel calls. By budget, a token at a time and with
the draft head (`bench/paged.py`): 2 GB 7.5 and 10 tok/s, 210 misses a token; 4 GB 7-8 and 12, 148; 8 GB 12 and
20, 63; 16 GB 18 and 34-41, 17. The disk here gives 6.5 GB/s; a slower one lowers every row but the last. A page
cache answering for the disk made the same run 2.4 times faster, which is why the cache is dropped first.

What it took, and what is still wrong:

- One expert swapped into a large array copies the array (2 ms), so each expert is three arrays of its own and the
  kernels take a token's ten as ten inputs; joining them first cost 8 ms a token.
- The wait for each layer's router costs 10 ms a token even when nothing is read: with every expert resident this
  way of running gives 22 tok/s, not 43. A machine that holds everything should not use it.
- The loader left 7 GB behind for 5.6 GB of dense weights (what numpy frees stays with the process) and peaked
  26 GB above the draft head's 1.6: both now convert a few rows at a time, to the same weights bit for bit.
- macOS puts any read of 2 MB or more through the page cache, F_NOCACHE or not: whole tensors are read in 1 MiB
  pieces.
- Tried and closed: reading straight into the buffer of an array that already exists (to spare making arrays).
  The bytes arrive, the GPU goes on seeing the old ones. A new array for every read costs nothing measurable.
- A knob not yet turned for the prompt peak below: smaller chunks (`--prefill 512`) hold less at once and read
  every layer four times as often.
- Reading a prompt held every delta-net layer's q, k and v of every token through the pass, and afterwards
  the whole convolution input behind each few-row slice of it: 4.6 GB at the peak and 1.4 GB left for 1,500
  tokens, now 2.1 and 0.4.
- The budget is filled at the start with the experts used most (`ExpertStore.preload`): by a profile of seven
  conversations that comes with the engine (`strata_mlx/expert_use.npy`), then by the machine's own answers,
  counted in `~/.cache/strata-mlx/expert-use-<model>.npy` (`--expert-profile off` for neither). It reads 1,080,
  7,293 and 13,507 experts in 0.3, 1.8 and 3.3 s as a 16, 24 and 32 GB machine. It buys little: the first answer
  after a start goes from 29.5 to 31.5 tok/s as a 32 GB machine (34.8 started again with the profile the same
  three requests left; 45 for an answer that follows one like it), and stays at 22 and at 6-7 as a 24 and a
  16 GB one. Which experts a conversation needs is not what was used most before: the profile's 13,500 hold 91%
  of all uses, and one miss in a layer is a read that the whole pass waits for.
- Not done: on 16 GB the dense weights' 5.6 GB as arrays are 2 GB more than in the file, which kernels on the
  file's blocks, as the experts have, would give to the experts; a layer still waits for its router and for
  its reads, where the experts the last token used could be asked for ahead; no prompt past 12K tokens has
  been read this way, where attention gathers 0.5 GB at a time and the conversation's caches (30 KB a token)
  come out of memory the budget does not count; and none of this has run on a machine that small or on a
  256-512 GB SSD.

What was measured before building it:

- **The disk.** Expert-sized reads at random places that are not in the page cache: 1.9 GB/s from one thread,
  6.5 GB/s from eight or more, which is about 4,700 Q2_0 experts a second. This is a 1 TB internal SSD; the
  256-512 GB ones in 16 GB MacBooks are slower, and that has not been measured.
- **How spread out the routing is.** Over seven conversations (prose, code, reasoning, an edit, Chinese, a
  summary, SQL) a token uses 480 of the 24,576 experts, and they are not the same few: with the experts most used
  in the other six conversations resident and the least recently used one replaced at each miss,

      resident experts     share    memory    misses per answer token (range over conversations)
           1,500             6%     2.1 GB    226  (201-260)
           3,000            12%     4.1 GB    147  (120-174)
           6,000            24%     8.3 GB     66  (43-95)
          12,000            49%    16.6 GB     21  (15-40)

  A fixed set, never replaced, misses about twice as often.
- **What that leaves a 16 GB machine** (as estimated then; the table above is what was measured since). Some
  4 GB for experts was the guess; the dense weights turned out to be 5.6 GB as arrays, not the file's 3.6, so with
  the draft head it is nearer 2 GB, the first row. The guess that the draft head would not help was wrong too:
  it does, by a third to a half, because the wait at each layer is paid once for a window.

Settled: the target file is Q2_0, the one on disk (no space for the Coder or any other file), behind a memory
budget that one build serves 16, 24 and 32 GB machines with; it is measured here by capping the budget, reading
experts past the page cache. 24 and 32 GB machines are the easier case: the 6,000 and 12,000 rows.
