# Working in this repository

> 中文摘要:在本仓库工作的规矩。最重要的一条:每一轮实现或实验之后,都要把学到的东西写进知识库(docs/wiki),
> 和代码放在同一个提交里。测试会检查这一点:改了 strata_mlx/ 或 bench/ 而 docs/wiki/log.md 里没有这次改动的记录,测试就失败。

## After every round of implementing or measuring

What was learned goes into the knowledge base (`docs/wiki`, conventions in `docs/wiki/README.md`) in the same
change as the code. In order:

1. **The experiment's code and results stay.** A script in `bench/experiments/eNN_<name>.py` (the next number),
   its results in `bench/results/`. Not in a temporary folder or a worktree that will be removed.
2. **An experiment page**, `docs/wiki/experiments/eNN-<name>.md`: the question, the setup (machine, file, change,
   what else was running), the results, the conclusion, the concepts it touched.
3. **The concept pages it touched.** A new claim with its conditions and evidence; a relation with evidence; a claim
   that fell moved to `## Superseded` with what made it fall, not deleted. A new concept gets a page and an area.
4. **A line in `docs/wiki/log.md`** with this change's ID (`jj log -r @`): what changed of what is known. Where
   nothing known changed (a refactor, a comment), the line says so; it is still the record.
5. **Check and test**: `uv run python tools/wiki_check.py --write`, then `MTL_DEBUG_LAYER=0 uv run pytest -q`.
6. **Commit with jj**, code and knowledge together.

The tests enforce 1, 2 and 4 where they can: an experiment script from e43 on without a page fails, and so does a
change to `strata_mlx/` or `bench/` without a line for it in `log.md`. ROADMAP.md and ISSUES.md stay the record in
time order; a finding goes there too where it belongs.

## Measuring

- Compare settings in one process, an engine for each sharing the weights, passes shuffled, with a control engine
  of the baseline's setting (`docs/wiki/concepts/measurement-noise.md`). Separate processes in turn tell little on
  this machine.
- Say which machine, which file, which change, and what else was using the GPU.
- Prompts on a machine that reads experts from the SSD: drop both shards from the page cache first.
- A claim from someone else's machine is theirs: say so, with the issue or pull request.

## Version control

- The repository is git with jj alongside. Commit with jj (`jj describe`, `jj new`, `jj bookmark set main -r @-`,
  `jj git push --bookmark main`); cite change IDs, which stay the same through amends and rebases.
- No `Co-Authored-By` lines in commit messages.
- Push only when asked.

## Tests

`MTL_DEBUG_LAYER=0 uv run pytest -q`. With Xcode's Metal debug layer on, a test of attention aborts on an
assertion in Metal itself.
