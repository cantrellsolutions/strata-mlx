# Log

> 中文摘要:知识库的变更记录,最新的在前。每行写日期、jj change ID、改了哪些页、依据什么。

- 2026-10-16 · [kzynqwqp] · e49: --expert-memory available under wired memory, twice e48's speed and next to no swap;
  available-memory, swapping, simulated-machine, system-reserve.
- 2026-10-16 · [szyzmoou] · e48: as a 48 GB machine wired, a full desktop open: the engine compressed, 10-15 tok/s;
  simulated-machine, swapping, system-reserve, expert-store.
- 2026-10-16 · [zzzlsupr] · e47: an unwired hold of 96 GiB kept 8.9 resident, so the hold is wired since; simulated-machine,
  swapping, measurement-noise.
- 2026-10-16 · [tukzwlpx] · A hold is judged by its resident part, not its footprint, which counts swapped pages too;
  the run counts this process's own memory compressed or swapped out (bench/memory.py:own). e46's "hold whole"
  corrected; simulated-machine, swapping.
- 2026-10-16 · [vrnvzqql] · e46: as a 48 GB machine with 80 GiB held; simulated-machine, swapping, system-reserve,
  measurement-noise.
- 2026-10-15 · [wumtsnom] · simulated-machine, working-memory: memory held by another process and the kernel's
  footprint (bench/memory.py, bench/paged.py --hold); a first run as a 96 GB machine; MLX's arrays miss 1.55 GB
  of what the process holds.
- 2026-10-15 · [owsylkrr] · No concept changed. Updating this knowledge base made a fixed step: AGENTS.md lists it,
  and the checker fails a change to strata_mlx/ or bench/ without a line here, and an experiment from e43 on
  without its page.
- 2026-10-15 · [oonxmoxu] · measurement-noise: a control engine, the same setting as the baseline, says what a
  one-process comparison cannot tell apart; from the M5 Ultra's run in pull request 9's thread.
- 2026-10-15 · [uvunqsoq] · hyper-connections, gpu-generation: the fusion limit chosen by GPU generation (8 from 17,
  6 below), from the M5 Ultra's measurement in pull request 9's thread.
- 2026-10-14 · [nlykrpkv] · Every concept given an area, and graph.md drawn as an overview of the areas and one diagram
  for each. Three areas added: "Reading prompts" (prompt-chunk, expert-tiles, kept-floats, gpu-idle; ngram-table
  moved here), "Conversations" (recurrent-state, attention-cache, checkpoints, conversation-parking,
  session-files, stopping), "Precision" (float32-path, tf32-on-m5, reference-agreement; exactness moved here),
  from ROADMAP.md's "Done" and prompt sections, e29 to e33, issues 1, 3 and 5, pull request 7.
- 2026-10-14 · [wvuromwk] · "Memory and smaller Macs": thirteen concepts (memory-rule, system-reserve, working-memory,
  available-memory, swapping, simulated-machine, expert-store, ssd-reads, page-cache, dense-blocks,
  layer-major-prefill, layer-read-ahead, router-lookahead) from ROADMAP.md's sections on small Macs, e35 to e40,
  issues 3, 4, 6 and 8 and pull request 7; ngram-table given its relation to the page cache.
- 2026-10-14 · [skkpzrnm] · started as a pilot: "Writing answers", eleven concepts, three
  experiments (e43-e45) drawn from ROADMAP.md, pull requests 9 and 10, and [skxmzozu] [wykuvxvr] [tmyqkuqv]
  [zryukqou]; the checker and the graph.
