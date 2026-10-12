# Open issues

Problems that are known and not solved. What was tried and measured on the way to the current state is in
`ROADMAP.md`; this file holds only what is still wrong.

## 1. Short prompts are read slowly, and slower still after an idle stretch

[Issue 1 on GitHub](https://github.com/yibie/strata-mlx/issues/1)

**What is seen.** A question of a few dozen tokens waits about half a second for the first token of its
answer. In the comparison with llama.cpp and LM Studio (`bench/compare_servers.py`, M4 Max, tokens a second
at which each engine reads the prompt):

| prompt | llama.cpp | LM Studio | this engine, Q2_0 | this engine, IQ3_S |
| --- | ---: | ---: | ---: | ---: |
| 27 to 49 tokens | 92-148 | 36-70 | 44-112 | 31-59 |
| 340 tokens | 427-433 | 284-320 | 419-451 | 319-367 |
| 1,180 tokens | 488-495 | 421-446 | 570-612 | 504-517 |

So 27 tokens take about 0.6 s here and 0.3 s with llama.cpp; with the IQ3_S file 31 tokens took 0.8 to 1.0 s.
Long prompts are not affected.

**What was measured** (`bench/experiments/e34_after_idle.py`, Q2_0 file):

| | one after another | after 0.5 s of nothing | 2 s | 4 s | 8 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| a piece of 64 tokens | 178-180 ms | 205 | 459 | 530 | 506 |
| a piece of 256 tokens | 484-486 ms | 518 | 526 | 553 | 574 |
| one token of an answer | 20 ms | 22 | 22 | 22 | 21 |

(A run a few hours earlier gave the same for 64 tokens: 179-184 ms, then 190, 496, 430, 580.)

A piece of 64 tokens takes two to three times as long once the GPU has had nothing to do for two seconds; a
piece of 256 tokens 30 to 90 ms longer; a single token no longer. The
comparison rests 8 s between requests, and a person pauses between messages, so every short prompt pays this.
The pieces by length in `ROADMAP.md` (e30) were each timed after a rest of 4 s too: its rows for 64 and 128
tokens include it.

**Two things, perhaps one.** Even one after another, 64 tokens take 179 ms (358 tok/s) where a piece of 1,024
reads at 650: a piece's cost has a part that does not shrink with its length (a tile of experts costs the same
with one pair in it as with sixteen; the layers of a piece of more than eight tokens are not compiled). The
slowness after a rest comes on top.

**Not known.** Why a rest costs a piece of 64 tokens 280 to 350 ms, a piece of 256 tokens a quarter of that,
and a single token nothing.

**The next step**, two minutes: time one fixed matrix product (the 2048-cube of `bench/quiet.py`) after rests
of 0.5, 2, 4 and 8 s. If it slows down the same way, the GPU's clock comes down while idle and takes some
hundred milliseconds of work to come back up: that would fit a long piece hiding it (it is at speed for most
of its time) and a single token not showing it (its kernels are small and wait for memory, not for
arithmetic). If the product is not affected, the cause is in how a short piece is read: look at what MLX's
buffer cache holds before and after the rest (`mx.get_cache_memory()`), and at the time the CPU takes to build
the graph of a piece that is not compiled. No fix is proposed before one of the two is seen.

## 2. The comparison's reading speeds are from two sessions

[Issue 2 on GitHub](https://github.com/yibie/strata-mlx/issues/2)

`docs/OPTIMIZATION_RECORD.md` and `ROADMAP.md` say the prompt of 1,180 tokens is read faster by this engine
than by llama.cpp (570-612 against 494-495 tok/s with the Q2_0 file, 504-517 against 488-489 with IQ3_S).
llama.cpp's and LM Studio's rows are from the first run of the comparison, all engines in turn; this engine's
are from a run of its own twelve hours later, after its prompt path was rewritten. This machine's speed depends on
what it did in the minutes before (it slows under sustained load), so the two are not strictly comparable. The
gain on this engine's side is 50% and more, which that does not erase, but the 3 to 6% lead with the IQ3_S
file is within it.

**The next step**, about ten minutes a file: run llama.cpp and this engine in turns again, the reading prompt
only (`bench/compare_servers.py` reads it last, after four answers; a flag to run only that would do).

## 3. The memory rule does not count the floats kept for prompts

[Issue 3 on GitHub](https://github.com/yibie/strata-mlx/issues/3)

**What is wrong.** Whether a Mac holds a file whole, and how much of its memory the experts get if it does not,
is worked out in `strata_mlx/paged.py` (`all_fit`, `ExpertStore.fit`) from the experts, the dense weights, the
draft head, 8.5 GB left to macOS (4.5 until issue 6) and 4.0 GB of working memory (5.0 since prompts are read
a layer at a time: section 4). Since the Q6_K layers keep their weights as
floats from the first prompt on (`dense.py`, so that the IQ3_S file's prompts are not read a tenth slower),
the process holds more than that rule counts: 3.3 GB with the IQ3_S file, 0.2 GB with Q2_0. The rule was
written before, and its working memory was measured with Q2_0.

**Whom it concerns.** A Mac that reads IQ3_S experts from the SSD gives them 3.3 GB that the floats then take,
so macOS is left that much less than the rule means. Since issue 6 that is every Mac under 67.7 GB, the 64 GB
ones among them: they keep 93% of the experts in memory. (While the rule left macOS 4.5 GB, a 64 GB Mac held the
file whole and was 2.1 GB short by the rule's own sizes. A Mac that holds it whole by the rule and not with the
floats now has between 67.7 and 70.7 GB, and none has.) Seen since, in the
one time that file was run that way (`bench/paged.py --machine-memory 24`, one conversation of 60 tokens, to
see that it runs): the experts were given 10.5 GB, everything else was 7.0 GB when they were, and after the
first prompt the process held 21.1 GB of arrays, 3.6 more than the two.

`./strata-mlx --check` counts the floats and says where a Mac is short (`strata_mlx/check.py`), so what it
reports and what the engine does differ in this one case, on purpose.

**Not known.** What a 64 GB Mac does with that: nothing here has run on one. With 8.5 GB meant for macOS it
is left 5.2.

**The next step.** Keep the floats only where they fit, and count them in the rule where they are kept: a
Mac short of them makes them anew for each piece of a prompt, as before (prompts a tenth slower, 320 against
355-369 tok/s on the M4 Max), and its answers are as fast as ever. Then run IQ3_S as a 64 GB machine
(`--machine-memory 64`) and see what it holds.

## 4. Where experts are read from the SSD, a prompt waits for the disk two thirds of the time

[Issue 4 on GitHub](https://github.com/yibie/strata-mlx/issues/4)

**What is seen.** As a 16, 24 or 32 GB machine (Q2_0, simulated on the M4 Max, the file dropped from the page
cache) a prompt of 1,537 tokens is read at 160-182 tok/s, and one of 1,180 tokens through the server at
107-145: 8 to 11 s before the answer's first token, where the machine with every expert in memory takes 2 to
3 s.

**What was measured** (`bench/paged.py --machine-memory <GiB> --engines head --read <prompt>`, 1,537 tokens):

| as a machine of | experts in memory | read from the disk for the prompt | of the time reading | tok/s |
| --- | ---: | ---: | ---: | ---: |
| 16 GB | 3.3 GB | 27.7-30.3 GB | 67% | 165 |
| 32 GB | 20.5 GB | 25.9-26.6 GB | 67% | 159-163 |

(Those were the experts such machines kept then. Since issue 6 a 32 GB machine keeps 16 GB and a 16 GB one
0.5 GB; this has not been measured again.)

A chunk of a thousand tokens uses nearly every expert of a layer, so the layer's three tensors are read whole,
in file order, and none of it is kept (`ExpertStore.whole`). The pass waits for each layer's read, 6.2 of
9.3 s, and the disk waits while the GPU computes the layer, the other 3.1 s. And what is in memory is read
again: with 59% of the experts in memory the prompt still reads 26 GB.

**A figure that was wrong.** `ROADMAP.md` and `docs/OPTIMIZATION_RECORD.md` said such a machine read the
1,537 tokens at 273-287 tok/s. That was measured with the model's file in the page cache, and a read past the
cache is answered from it when the pages are there: the same script gives 255-277 tok/s with the file in the
cache and 149-162 with it dropped. Both records are corrected.

**What was found since** (e37 to e39; `ROADMAP.md`, "A prompt read a layer at a time").

- A prompt of several chunks reads each layer's experts once a chunk: 8,192 tokens read 99-100 GB, 70 of them
  for a second time or more, and take 37.6-45.3 s as a 32 GB machine.
- `whole` is not slow: alone it reads at 5.9-6.5 GB/s, which is the disk. In the engine it is 4.0-4.8, for a
  reason not found.
- Reading the next layer ahead with the order left as it is, the first step this section named, is 1.26-1.58
  times by estimate for 8,192 tokens: a chunk reads only 10 to 15 of its layers whole.

**What was done** (`strata_mlx/bylayer.py`; `ROADMAP.md`, "A prompt read a layer at a time"). A prompt of more
than one chunk goes a layer at a time in pieces of 8,192 tokens, each layer's experts read once, the next
layer's read while this one is computed. The state it leaves is the engine's bit for bit. As a 32 GB machine
(e40, medians of three, in turns): 8,192 tokens in 19.2 s where chunk by chunk took 32.4, and 34 GB read
where it read 70; 4,096 tokens 9.7 s for 15.1; 32,768 tokens, once, 116.8 s for 192.9. A prompt of one chunk
is read as before, 6.3 s for 1,537 tokens, and still waits for the disk half of that.

**Still open.** One chunk, and so every short prompt, waits for the disk as it did. With a large share of the
experts in memory only what is lacking need be read, as a 32 GB machine 18 of 34 GB a piece: not built, and it
matters where the disk is slower than this one. None of this has run on a small machine.

## 6. On a real 32 GB Mac the default leaves macOS too little memory, and it swaps

[Issue 6 on GitHub](https://github.com/yibie/strata-mlx/issues/6)

**What is seen.** On a MacBook Air M5 with 32 GB (macOS 27.0.1, MLX 0.32.3, commit `1d75ad4`, Q2_0 with the
draft head, about 2 GB of other programs open) the engine by default kept 20.5 GB of experts in memory, and the
machine swapped. With fewer experts in memory it was about twice as fast. Measured by @guoggg
([comment on issue 3](https://github.com/yibie/strata-mlx/issues/3#issuecomment-6110466402)) with
`bench/paged.py`, four prompts by two rounds of 200 tokens, and `--read` of a 1,180-token prompt:

| experts in memory | draft head tok/s (mean) | one token a pass (mean) | read 1,180 tokens, tok/s | arrays at peak | swap peak | swapped out during run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 GB (`--budget 14`) | 20.9-24.8 (23.6) | 13.6 | 152, 155 | 22.4 GB | not recorded | not recorded |
| 16 GB | 23.4-26.8 (25.3) | 13.8 | 95, 119 | 24.5 GB | 2.8 GB | 1.7 GB |
| 18 GB | 12.0-24.4 (15.4) | 9.2 | 89, 110 | 26.5 GB | 5.1 GB | 39 GB |
| 20.5 GB (the default then) | 5.8-19.2 (11.9) | 7.1 | 49, 136 | 29.1 GB | 6.9 GB | 106 GB |

Each row is one run; the prompt read is the reporter's own text, not the one used here.

The 16 GB row again with every other program closed
([comment](https://github.com/yibie/strata-mlx/issues/6#issuecomment-6115147674)): answers at 24.3-28.8
(27.1), one token a pass 14.9, the prompt at 108 and 157, arrays at the peak 24.5 GB, 0.39 GB swapped out during
the run. Before it `vm_stat` gave 23.65 GB free and macOS with its background items 6.1 GB (wired, active and
compressed), 3.6 GB more inactive.

**What it says.** The rule in `strata_mlx/paged.py` counts the process rightly: in all four rows the arrays
peak at the experts' budget plus 8.4 to 8.6 GB. What was too little is what it left to macOS and other
programs, `SYSTEM`, 4.5 GB: with the default the machine swapped 106 GB out during the run, and with about 8 GB
left it hardly swapped. Of those 8 GB about 6 are macOS itself and 2 were the other programs.

**Why it was not seen here.** The speeds given for 16, 24 and 32 GB machines are from the 128 GB M4 Max with
the experts' budget limited. Nothing else is short of memory there, so it never swaps. The README's 23-38 tok/s
"as 32 GB" is such a figure; the real machine's default gave 5.8-19.2.

**What was done.** `SYSTEM` is 8.5 GB, which gives a 32 GB machine's experts 16.2 GB: the row that was
fastest. By the same rule a 16 GB Mac is left the least, 0.5 GB of Q2_0's experts (it kept 9% of them; `--check`
calls that too small), a 48 GB Mac keeps
98% of Q2_0's experts where it held the file whole, and a 64 GB Mac 95% of IQ3_S's. (Since the working memory
is 5 GB, section 4: 15.2 GB for a 32 GB machine, and 95% and 93%.)

**Not known.** Whether a 16, 24, 36 or 48 GB Mac needs the same amount, or less, or the same share: this is one
machine, one run a row. A 16 GB Mac may well do with less than 8.5 GB and is cut to the least here on no
evidence of its own; `--expert-memory` sets the experts' share by hand. The speeds given for 16, 24 and 32 GB machines were
measured with 3.3, 11.9 and 20.5 GB of experts and not again since.

**The next step.** Any run on a real 16, 24 or 48 GB Mac, and the 32 GB machine with the new default.
