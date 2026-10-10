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
are from a run of its own a day later, after its prompt path was rewritten. This machine's speed depends on
what it did in the minutes before (it slows under sustained load), so the two are not strictly comparable. The
gain on this engine's side is 50% and more, which that does not erase, but the 3 to 6% lead with the IQ3_S
file is within it.

**The next step**, about ten minutes a file: run llama.cpp and this engine in turns again, the reading prompt
only (`bench/compare_servers.py` reads it last, after four answers; a flag to run only that would do).

## 3. The memory rule does not count the floats kept for prompts

**What is wrong.** Whether a Mac holds a file whole, and how much of its memory the experts get if it does not,
is worked out in `strata_mlx/paged.py` (`all_fit`, `ExpertStore.fit`) from the experts, the dense weights, the
draft head, 4.5 GB left to macOS and 4.0 GB of working memory. Since the Q6_K layers keep their weights as
floats from the first prompt on (`dense.py`, so that the IQ3_S file's prompts are not read a tenth slower),
the process holds more than that rule counts: 3.3 GB with the IQ3_S file, 0.2 GB with Q2_0. The rule was
written before, and its working memory was measured with Q2_0.

**Whom it concerns.** A Mac that holds the IQ3_S file by the rule and not with those floats: between 63.0 and
66.1 GB of memory, which is the 64 GB Macs. There the engine holds the file whole and, by the rule's own sizes,
is 2.1 GB short of what it means to leave to macOS. By what was measured on the 128 GB machine it is short by
less: IQ3_S with the draft head holds 57.8 GB, 61.1 after its first prompt, and peaks 3.8 GB above that while a
prompt is read. That is 64.9 GB of a 64 GB Mac's 68.7, which leaves macOS 3.8 GB where the rule means 4.5.
And a Mac that reads IQ3_S experts from the SSD gives them 3.3 GB that the floats then take; that file has
never been run that way.

`./strata-mlx --check` counts the floats and says where a Mac is short (`strata_mlx/check.py`), so what it
reports and what the engine does differ in this one case, on purpose.

**Not known.** What a 64 GB Mac does with that: nothing here has run on one.

**The next step.** Keep the floats only where they fit, and count them in the rule where they are kept: a
Mac short of them makes them anew for each piece of a prompt, as before (prompts a tenth slower, 320 against
355-369 tok/s on the M4 Max), and its answers are as fast as ever. Then run IQ3_S as a 64 GB machine
(`--machine-memory 64`) and see what it holds.
