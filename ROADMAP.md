# Roadmap

## Done

- The model straight from Strata's GGUF files, without changing a weight (`strata_mlx/gguf.py`, `load.py`).
- Experts computed on the file's own blocks (`blocks.py`): nine block types, each checked against gguf-py's
  dequantizer on random blocks. Nothing is converted at start (22 s to load Q2_0, was 34-57), and in a pass the
  kernels are as fast as MLX's own 2-bit gather or a little faster (43 tok/s a token at a time, was 38-40), so
  Q2_0 uses them too. A chunk of a prompt multiplies from the blocks too, a tile at a time (e33, below); it
  unpacked each stack to floats for the length of one product before.
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

- **The larger model sizes.** IQ3_S runs: 56 GB of weights (61 with the draft head), 38-45 tok/s a token at a
  time, 52-81 with the draft head (see "The IQ3_S file" below), about 440 tok/s reading a 1,500-token prompt. On a chat prompt it picks llama.cpp's token at 63 of 64 steps
  and its log-probabilities are closer to either llama.cpp backend (median 0.05 and 0.09) than those are to each
  other (0.09); llama.cpp on Metal writes 33 tok/s from the same file. All six expert types of the file unpack
  to gguf-py's weights exactly. IQ3_XXS and IQ2_XS will not be run: there is no disk space for more model
  files. Their tensors all have a path through the loader and their block types pass the same tests.
- **The same answer with and without drafts at full speed.** True in float32; in float16 a window and a single
  token round differently and the answers can part at a near-tie. What that costs, judged by the model itself
  in float32 a token a pass (e29): float16 writes a token other than the model's first choice about once in
  250, with drafts or without, with the listed rows or without, each time one the model found nearly as
  probable (at the widest 44.6% against 38.0%); float32 with drafts and the list wrote the model's own answer
  in all eight answers. `--dtype float32` costs half the speed with the Q2_0 file (44-49 tok/s for 80-103, its
  table lookup being float16 only) and a tenth to a fifth with IQ3_S (53-73 for 69-80).
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
machine it repeats within 1%; other programs on the GPU move it by a half within minutes, so every change is
compared in one process, the two ways alternated pass by pass. Tokens a second are from `spec_decode.py`
(256 tokens, greedy, with thinking) on a quiet machine, with the draft head.

    step                                             one token   four tokens   prose / code / reasoning / edit
    start                                             22.3 ms      35.8 ms      61-63 / 69-70 / 77-82 / 77 tok/s
    hyper-connections read once for a window (e07)    22.1         34.0         63 / 73 / 82 / 82
    Q2_0 experts by table lookup (e12)                22.1         31.5         72 / 75 / 82 / 85
    n-gram rows on the CPU (e14), routers' own
      product, the experts' activation folded in      20.8         30.3
    one wait for the GPU a step; three guesses,
      then more while the head is sure (e13)                                    88 / 81 / 97 / 97

    a token a pass, no draft head: 43 tok/s at the start, 48 now.

What the measurements said, in the order they corrected each other:

- **A pass is not thousands of small kernels that cost by their number.** A token reads 5.5 GB of weights: 4.87
  of dense tensors (hyper-connections 1.27, delta-net projections 2.1, attention 0.6, output layer 0.5) and 0.66
  of experts. Summing those arrays once takes 13.9 ms; a one-token pass waits 20 ms for the GPU. An earlier note
  here put the bytes at 2 GB and the floor at 5 ms, and three rounds of merging kernels bought 15%.
- **Parts timed alone do not add up to a pass** (e01, e04, e05). MLX commits its command buffer after every
  50 MB of inputs, 23 us each time, so a kernel with a 236 MB stack of experts as input costs 25 us alone; in a
  pass the GPU is what is waited for and the limits (20 to 500 operations, any size) change nothing. Parts are
  measured by leaving them out of a real pass (e06): of 23.8 / 38.1 ms for one / four tokens at the start,
  routed experts 4.1 / 12.6, delta-net projections 6.0 / 8.7, hyper-connections 5.1 / 8.1, attention projections
  2.3 / 3.9, output layer 2.3 / 3.1, routers 2.0 / 2.7, shared experts 1.8 / 2.5.
- **One GPU thread is slow; a kernel takes as long as its longest thread** (e08, e09). A matrix of 324 by
  10,240 times four tokens: 38 us with 32 threads to a row, 21 with 1,024. A norm kernel of 128 threads looping
  160 times: 21 us for nothing. An array local to a thread: six times slower. From this the hyper-connections'
  three kernels and the routers' product (fused.py).
- **The expert kernel is bound by arithmetic, not by bytes or loads** (e02, e03, e10, e11): about eight
  operations a weight at all this GPU has, whatever the inner loop, the threads to a row, four values to a load
  or rows sharing their activations. A table of the 16 sums two weights make of two inputs is three operations
  a weight; a table of 256 sums for four weights does not stay in a cache and is four times slower (blocks.py,
  `lookup_matvec`; e12).
- **Waits for the GPU that nobody needed** cost 3.6 ms of a 41 ms step: the context tokens were kept as a GPU
  array (a slice and a concatenation waited for at every commit, and again by the n-gram lookup), and the states
  a pass leaves behind were waited for although only the next pass needs them.
- **How many guesses** (e13). The head's sureness of a run of guesses (the product of the probabilities it
  gave them) predicts whether they are kept, well for answers that repeat or reason step by step, poorly for
  prose. Three guesses in any case and more while it is sure: replayed, 6% over three every pass; measured, 2%
  on prose, 5% on code, 6% on reasoning, 10% rewriting a file. Deciding every guess by sureness lost on prose.

Tried, no gain:

- MLX's own switches (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`,
  `MLX_BFS_MAX_WIDTH`).
- The expert kernel loading four values at once (e10), or several rows to a thread (e11).
- A dense projection on the file's Q3_K blocks (e15): 74 / 80 / 91 us for 1 / 2 / 4 tokens against MLX's 77 /
  78 / 81 on its 8-bit form, 2.6 times the bytes. MLX's quantized product is flat up to four tokens and reads
  at 370-400 GB/s; the block kernel is held back by its million threads and by what each must do to undo the
  format. (With the layer's scales left in float32, MLX's product takes 134 us for four tokens: the first run
  of this experiment had it so, and the block kernel looked a third faster.)
- A copy of the output layer at fewer bits for the head's guesses (e16): 0.2 to 0.4 ms less a guess, and 3%,
  8%, 14% of the guesses change at 4, 3, 2 bits.
- A kernel for picking the largest logit: in a chain MLX's argmax of 248,320 takes 30 us (e17); the 0.4 ms it
  seemed to take alone was the wait for the GPU.

- Guesses copied from earlier text on top of the head's own (400 tokens, quiet machine): with three guesses a
  pass they add 7% rewriting a file (94 to 101 tok/s) and nothing on code; with the head guessing as many as it
  is sure of, nothing (101 to 100, 74 to 71). The server now turns them on by itself only without a draft head.

Not yet tried: the ten largest of a router's logits and their softmax as one kernel (about 18 us a layer
now); a block kernel for IQ4_XS, whose format is simpler than Q3_K's; the head's own layer, which is MLX
operations throughout. Each is under a millisecond a pass.

Changes weights, so not done, only measured (e18): the hyper-connections' 1.27 GB of 16-bit floats at 8 bits
would spare a pass 0.6 GB, about 1.6 ms. Rounded to 8 bits and compared with llama.cpp's answers: its token at
64 of 64 steps on the chat prompt as before, at 62 of 64 on the 7K-token one (64 before); at 6 bits 62 and 61,
at 4 bits 60 and 54.

### The IQ3_S file (e20 to e24)

Of the changes above, all but the table lookup apply to any file and were already in what IQ3_S ran with. What
that file has of its own is not in its experts but in its dense layers: 1.88 GB of them are Q6_K (the Q2_0
file has 84 MB), which scales every 16 weights, and MLX takes no scale to fewer than 32 (`quantize` refuses, the
product has no kernel). The loader unpacked Q6_K to floats: 4.58 GB of float16, so that a token read 7.16 GB of
dense arrays where the Q2_0 file's token reads 4.87, and the output layer's 1.27 GB again for every guess of the
draft head.

    step                                         one token   four tokens   a token reads   prose / code / reasoning / edit
    start (Q6_K as float16)                       26.6 ms      38.8 ms      7.16 GB         40 / 54 / 66 / 74 tok/s
    Q6_K as a byte a weight (dense.py, e23)       22.2         35.2         5.03            52 / 67 / 81 / 81

    a token a pass, no draft head: 28 / 33 / 36 / 37 tok/s before, 38 / 42 / 44 / 45 after.

Tokens a second: before and after a prompt at a time, in turns, a process each (the machine sped up through
that run, so a column compares, a row does not). llama.cpp's token on the chat prompt at 63 of 64 steps as
before. With the Q2_0 file, whose six Q6_K layers go the same way now: 64 of 64 on the chat prompt as before,
and on the 7K-token one 63 of 64 a token at a time (64 before), 64 of 64 four at a time. Its answers are as
fast as before or a little faster (in turns as above: 35-46 tok/s a token at a time against 35-45, 53-97 with
the head against 54-95).

- **On the file's blocks** (e20, e21): 6.56 bits a weight, and faster than MLX over float16 for one and two
  tokens (72 and 83 us against 130 for a 10,240 x 2,560 tensor), level at four, 1.7 to 3 times slower at six and
  eight: MLX's product stays at 130 us whatever the window, the block kernel undoes the format's interleaving
  for every weight and then multiplies for every token.
- **A byte a weight** (e22, e23): the code as a signed byte, the scales and d as the file has them, the same
  number at 8.56 bits; a kernel laid out as MLX's quantized product. In a pass, against float16: 4.0 ms less for
  one token, 3.5 for two, 3.2 for four, 1.4 for six, 1.7 more for eight. One form in memory, 2.45 GB for 4.58.
- **The draft head's guesses** (e24): three guesses cost a step 7.5 ms of 43, of which 4.1 are the output layer
  (2.9 of 6.2 with the Q2_0 file): every guess is all 248,320 rows for one token. See the next section.
- **The experts** were left: a four-token window reads 3.9 GB of them, which is most of the 11 ms they take.

Timings of answers on this machine move by up to a half for minutes at a time, and nothing but the GPU's own
speed shows it (bench/quiet.py). At least part of it is the machine: a 14-inch MacBook Pro whose GPU does a
4096-cube product at 14.5 T operations a second at rest does it at 12 after twenty seconds of full load, with
the thermal state then "fair"; in the middle of long benchmarks it was at 9 and below. So a benchmark's later
prompts run slower than its first, and before and after are compared a prompt at a time, in turns, a process
each. The first IQ3_S numbers of this round (25-27 tok/s a token at a time, 36-49 with the head) came from the
slow stretches and are void.

### Guesses over the rows that have been in sight (e25 to e28, rows.py)

A guess that comes out wrong costs speed, never the answer, so a guess need not look at every row of the output
layer if the row that wins is seldom one it skips. Recorded in four answers (1,190 guesses): of the guesses the
trunk kept, 72% were tokens of the conversation itself and 98% were that or among the 256 largest of some row
of logits the trunk had worked out by then, which are there for nothing. So those rows are listed, about
13,000 after an answer and 30,000 after four, a twentieth to an eighth of the layer, and a guess multiplies
only them. The list is kept between conversations; the largest of a pass's rows are picked out by the GPU
while the CPU makes the draft and are listed a step later.

In turns with guesses over all rows, one process, two rounds a prompt (e28):

    three guesses a pass, Q2_0            +5.4 +4.8 | +2.0 +7.6 | +3.8 +6.7 | +9.7 +7.9 %     tokens a pass as before
    three guesses a pass, IQ3_S           -0.6 +12.2 | +6.7 +8.7 | +3.6 +8.3 | +6.4 +9.1 %    as before but for the first
    as many as the head is sure of, Q2_0  -1.2 +7.4 | -3.6 +7.5 | +4.5 +13.0 | +14.0 -1.2 %   (5% on average)
    as many as the head is sure of, IQ3_S +8.1 +6.9 | +6.4 +11.0 | +4.1 +5.7 | +7.4 +10.9 %    (8% on average)

The first answer after a start has a short list and loses a few guesses to it (69% kept where 70% are, with
the Q2_0 file). With IQ3_S that first answer also parted from the other engine's at token 107, a near-tie in
float16 that the different windows tipped, and was no faster. With as many guesses as the head is sure of, a
run differs from the next by more than the list changes it:
the same engine settles 3.8 and then 4.4 tokens a pass on the same prompt, the policy having learned in
between. The head's probability of a guess is now taken among the listed rows; that makes it too sure by under
0.1% for nine guesses in ten.

Tried on the way: a sketch of the layer (its rows squeezed to the directions they vary most in) to pick the
rows instead of a list: the rows vary in all directions alike, and it takes a fifth of the layer to find 99% of
the winners (e25). Listing the rows within some distance of the largest instead of a fixed number: a row with
no clear winner lists tens of thousands (e27). Picking the largest out and waiting for them: 1.1 ms a step,
a third of what the list spares (e27).

### Against llama.cpp's server and LM Studio (bench/compare_servers.py)

One file, the same prompts as text with their chat markers, each engine loading the model alone and reporting
its own timings; 256 tokens an answer, the most likely token every time; "reading" is a prompt of 1,180 tokens.
Tokens a second, prose / code / reasoning / edit, then reading:

    Q2_0    llama-server 0.4.1                38-39 / 38 / 38 / 38                494-495
            LM Studio 0.4.25 (llama.cpp)      43 / 43 / 43 / 43                   426-446
            this engine, --spec 1             44-48 / 50-53 / 48-51 / 61-67       595-612   (320-385 before e30-e33)
            this engine as it serves          68-91 / 70-86 / 105-107 / 95-111    570-578   (370-386)
    IQ3_S   llama-server 0.4.1                35 / 35-36 / 35-36 / 35             488-489
            LM Studio 0.4.25 (llama.cpp)      40 / 40 / 40 / 40                   421-424
            this engine, --spec 1             43-45 / 49 / 43-45 / 53-56          504-515   (349-377)
            this engine as it serves          65-66 / 79-81 / 88-95 / 87-88       510-517   (334-349)

Answers come about twice as fast as from either (1.6 to 2.9 times). Prompts were read slower than by both
when this was first run; since the experts of a prompt go in tiles from the blocks (below) the 1,180 tokens are
read faster than by either, a prompt of 340 tokens about as fast as by llama.cpp with the Q2_0 file (419-451
against 427-433) and slower with IQ3_S (319-367), and prompts of 27 to 49 tokens slower (31-112 against 92-148).
Ollama 0.35.1 loads neither file: its GGUF reader does not know the Q2_0 tensors, which the IQ3_S file has too.
`--spec 1` is not a token a pass: with no draft head the server copies guesses from the context. The other
engines' rows are from one run of all engines in turn, twice in opposite orders; this engine's are from a run
of its own afterwards, on the code of 25cf76e.

That comparison found a fault: with Q6_K as a byte a weight, the IQ3_S file's prompts were read at 246 tok/s
(the Q2_0 file's at 390-400). A chunk of a prompt multiplied those layers as floats worked out anew by MLX
operations, half a second for every piece a prompt is read in. Now one kernel makes the floats (12 ms), at the
first prompt, and they are kept: 3.3 GB more held with the IQ3_S file, 0.2 GB with Q2_0. Made anew for each
prompt they cost a tenth of the reading speed (320 against 355-369 tok/s), the fresh memory and not the work.

### Why prompts are read slower: the ten chosen experts (e30)

Asked after the comparison, and not measured before: the loop above was all about answers. A chunk of a prompt
is few kernels, each milliseconds long, so its parts can be timed in place (each made to finish at its edges;
that costs a fifth more time with the Q2_0 file). A piece of 1,024 tokens, Q2_0, 2.4-2.5 s without the stops:

    the ten chosen experts                         1570-1630 ms   52-54%    4.83 T operations   3.0-3.1 T a second
    delta-net projections                           400-430       14%       4.27                10.0-10.6
    hyper-connections                               230-270        8%
    choosing the ten, weighting and adding them     210-230        7%
    the shared expert                               140-170        5%       0.48                2.9-3.5
    attention projections                           117-118        4%       1.22                10.4-10.5
    delta-net convolution and norms 70, its recurrence 53, attention itself 42-44, the routers' product 25-29,
    the n-gram table 18, glue 17-20, keeping the states 9, the output layer 5-6

The stops flatter the experts' share: without the stops, and with the ten chosen answering zeros, the piece
takes 850 ms where it takes 2,990 (both files, a warm machine). They are 71% of a prompt's time, at 2.3 T
operations a second, when the dense products of the same pass run at 10 and the GPU does 14.3 in one large
product. All else of a pass would read 1,200 tokens a second. So what an earlier note said, that reading is
near what this GPU can do and the gap to Strata is the hardware's, holds for a third of the pass and not for
the rest.

Why they are slow, as far as measured: 1,024 tokens are 10,240 pairs for 372 experts of a layer, 27.5 each,
so every product is small; in tiles of 16 tokens 26% of the slots are filling; a scale to every 64 weights
makes each tile forty products; and every chunk splits all 512 stacks of every tensor into codes and scales
whatever its length. With the IQ3_S file, which has no 2-bit operand, each stack goes through 1.7 GB of floats.

The two ways for Q2_0, forced at each length (ms a piece: 2-bit product in tiles / table of sums a pair):

    128 tokens   811 / 376      256   1066 / 667      512   1702 / 1492      1024   2610 / 3469      2048   4143 / 6332

The engine changes to tiles from 3,000 pairs (300 tokens), set before the table of sums existed; at 512 tokens
the table is still the faster by an eighth. Not changed yet: where they cross between 512 and 1,024 is not
measured. Not the cause: the cut before the last token (a prompt of 1,180 is read as 1,179 and 1), the draft
head reading along (3,140 ms against 3,113), the n-gram table, the recurrence.

What would help, it seemed, was a kernel that reads an expert's blocks once for all the tokens that chose it.
Measured before building it (e31, e32), the product was not what was slow:

    a layer's ten chosen, their inputs given (e31, ms, the mean of five layers of a real pass, 10,220 pairs):
    splitting the three stacks into codes and scales 4.1, the sums of the activations 2.3, the three products
    4.7 + 4.6 + 4.4, sorting the pairs on the CPU 0.5, the gathers 1.4, silu 0.5: 20.4 in all, 1.0 s a chunk

    one product (gate), T operations a second: the 2-bit product in tiles 7.1, MLX's gather_mm over the stack
    as float16 8.1, MLX's gather_qmm over the stack requantized at 2 bits 6.8, the table of sums 2.8

So the product was already where MLX's own grouped products are, and a chunk paid 1.7-2.1 s for 1.0 s of
experts. The rest was the wait: the pairs were sorted by expert on the CPU, so a pass stopped at every layer
until the GPU had the routers' choice, and what followed each stop ran slower. With the choices known
beforehand (taken from a pass before) the same chunk took 1.8 s instead of 2.7 (e32).

Now the GPU sorts them (blocks.tiles_of): the tiles are laid out by array operations, in as many tiles as the
pairs could need at most, and the kernel leaves out those nothing fills. In turns, one piece of 1,024 tokens:

    sorting on the CPU 2723 ms (376 tok/s)   the choices known beforehand 1821 (562)   sorting on the GPU 1832 (559)

The first token of the answer is the same each way; against llama.cpp's answers after prompts of 3,161 and
6,945 tokens the top token is equal at 64 of 64 and 63 of 64 steps a token at a time, 64 and 64 four a pass,
as before. A prompt of 1,180 tokens as the engine reads it: 2,298 ms (514 tok/s) where it took 2,885-3,113.
By length, tiles / table of sums, tok/s: 192 tokens 281 / 381, 256 330 / 386, 320 380 / 388, 384 410 / 390,
1,024 561 / 361, 2,048 627 / 344; the change of way stays at 3,000 pairs. Peak memory with chunks of 2,048:
4.6 GB over the weights. The ten chosen are now 54% of a chunk (980 ms of 1,810), at 4.9 T operations a
second.

The IQ3_S file's experts did not go this way (no 2-bit operand): each stack was unpacked to 1.7 GB of float16
(5.9 ms) for MLX's gather_mm (5.2 ms), three times a layer: 34 ms a layer step by step, 43 in a pass, 79 when
a layer's three products are made to finish together (three stacks of 1.7 GB at once).

Then the kernel (e33, blocks.tile_matmul), for every type: a threadgroup takes a tile of 16 tokens and 64 rows
of the tile's expert, unpacks 64 columns of those rows into threadgroup memory (`unpack32`, as the other
kernels) and Apple's matmul2d multiplies the tile's tokens by that operand; then the next 64 columns. Nothing
is split, summed or unpacked beforehand. On layers of a real pass, one product of 10,220 pairs:

    IQ2_S 5.3 ms   IQ3_S 5.1   IQ3_XXS 5.1   IQ4_NL 4.8   Q2_0 4.5-5.0        6.3 to 7.4 T operations a second
    a layer's three products: 16.3 ms; before, Q2_0 22.2 (split 4.1, sums 2.3, products 13.7), IQ3_S file 43-79
    tiles of 32 tokens, or 128 columns at a time: a fifth slower; 256 columns (all of a threadgroup's memory):
    wrong results on the real shapes. matmul2d sets its destination, it does not add to it.

So the 2-bit operand of Apple's primitive, the split and the sums are gone again (e30 to e32 are of that code).
A piece of a prompt, in tiles / a pair at a time (the table of sums for Q2_0, the matvec kernel else), tok/s:

    Q2_0     64 tokens 147 / 154   128 283 / 309   192 429 / 362   256 476 / 368   512 593 / 383   1,024 679 / 378   2,048 696 / 337
    IQ3_S    64 tokens  89 /  85   128 166 / 178   192 251 / 241   256 337 / 277   384 467 / 288   1,024 633 (358 unpacked)   2,048 671 (382)

Tiles from 1,600 pairs on. As the engine reads, with the draft head reading along, three prompts each:

    Q2_0     1,180 tokens 555-560 tok/s   2,620 652-653   8,668 420-507   peak 2.5 to 4.0 GB over what is held
    IQ3_S    1,180 tokens 488-513         2,620 633-639   8,668 455-614   peak 5.6 to 7.1 GB (3.3 of them the
             Q6_K layers' floats, kept from the first prompt on)

The comparison's prompt of 1,180 tokens was read at 370-386 (Q2_0) and 334-349 (IQ3_S) before; llama.cpp reads
it at 494 and 488, LM Studio at 426-446 and 421-424. The longest prompts slow down as they go: each read is
15 to 20 s of full load, and the machine slows under it (above).

Checks. Every type on random blocks against the unpacked weights, float16 and float32: 2e-7 of the largest
value. On real layers against a pair at a time: 1e-4 to 5e-4 a product, 1e-3 through all three, as the ways
before. Against llama.cpp's saved answers, top token equal a token at a time / four a pass: Q2_0 after 1,537
tokens 64 / 64, after 3,161 64 / 64, after 6,945 64 / 63 (before 63 / 64), in float32 after 3,161 64 / 64;
IQ3_S after 1,537 tokens 64 / 64 against both its Metal and its CPU answer (before 63). With part of the
experts read from the SSD (1.5 and 18.7 GB of them in memory) the 1,537 tokens are read at 273-287 tok/s
where they were at 136-159, and the answer starts with the same token.

The ten chosen are now half of a chunk's time (49-54%), at 4.9 to 6.1 T operations a second; all else of a
pass would read 1,200 tokens a second. What is left in them: a tile costs the same with one pair as with
sixteen, and 26% of the slots are filling at 1,024 tokens.

Open, seen on the way and not looked into: a short piece right after an idle stretch is slow. 64 tokens take
179 ms one after another and 430 to 580 ms after 2 to 8 s of nothing; one token of an answer is 21 ms either
way, 256 tokens show little of it. A question of a few dozen tokens therefore waits half a second for its
first token.

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
  per-expert ones do not: they run at 2.3 and are 71% of the time (e30, above), so the engine and not the GPU
  is what holds prompts at 320-440 tok/s.
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
