# The knowledge base

> 中文摘要:这是本项目的知识库,按**概念**组织,而不是按时间。每个概念一页,写清它是什么、在引擎里的位置、
> 它和其他概念的关系(用固定的关系词),以及每条结论的依据(实验、提交、issue)。实验各有一页,和概念互相引用。
> 版本以 jj 的 change ID 为准,检查脚本 `tools/wiki_check.py` 保证引用不断、关系图不过期。

What this engine does is decided by a few dozen ideas that act on each other: how a pass spends its time, how
many tokens a window settles, what memory a Mac leaves the experts. ROADMAP.md and ISSUES.md hold what was found,
in the order it was found. This holds it by concept: one page each, with its relations to the others and the
evidence for every claim, so that a new result can be put where it belongs and an old one marked when it falls.

## Layout

- `index.md`: every concept, one line each, by area.
- `concepts/<id>.md`: one concept.
- `experiments/eNN-<name>.md`: one experiment, from e43 on. e01-e42 are in ROADMAP.md and `bench/experiments/`.
- `log.md`: what changed in this knowledge base, newest first, each line with its change ID.
- `graph.md`: the relations as a diagram, written by `tools/wiki_check.py --write`. Do not edit it by hand.

## A concept page

```
---
id: hyper-connections        # the file's name; what [[links]] use
kind: mechanism              # concept | mechanism | component | metric | hardware | constraint | method
status: current              # current | superseded
---
# Title

> 中文摘要:one or two lines.

## What it is
## In this engine            code as `path.py:Name`, which the checker looks for
## Relations                 one line each, see below
## What we know              claims, each with its conditions (machine, file, commit) and evidence
## Superseded                claims that fell, kept as they were, with what made them fall
## Open questions
```

A relation is one line under `## Relations`:

    - <relation> → [[<target>]]: what it means here, with numbers [evidence] [evidence]

Relations, from this page's concept to the target:

| relation | meaning |
| :--- | :--- |
| `part-of` | is a part of the target |
| `depends-on` | needs the target, or changes when it does |
| `costs` | takes time or memory from the target; give the figure |
| `limited-by` | is held back by the target |
| `trades-against` | more of one is less of the other |
| `tuned-by` | its best setting depends on the target |
| `invalidates` | made a claim about the target fall |

Every relation carries at least one piece of evidence in square brackets:

| evidence | means |
| :--- | :--- |
| `[e44]` | an experiment: its page here, or for e01-e42 its script in `bench/experiments/` |
| `[#9]` | an issue or pull request on GitHub |
| `[zryukqou]` | a jj change ID (eight letters k-z): the change that did or recorded it |
| `[ROADMAP: Faster answers]` | a heading of ROADMAP.md, by its start |

## Versions

The repository is a git repository with jj alongside (`jj git init --colocate`). A change ID stays the same when a
change is amended or rebased, which a commit hash does not; a commit made with plain git gets its change ID from
its hash, the same in every clone. So evidence cites change IDs. `jj log -r <id>` finds one; on GitHub the commit
hash is in `log.md` beside it.

Knowledge and code change together (the steps are in the repository's `AGENTS.md`): a change that alters a measured behaviour updates the concept pages it touches
in the same change, and adds a line to `log.md`. A claim that falls is moved to `## Superseded`, not deleted.

## The checker

    uv run python tools/wiki_check.py           # report; exit 1 on errors
    uv run python tools/wiki_check.py --write   # also rewrite graph.md

It fails on: a change to `strata_mlx/` or `bench/` with no line in `log.md` carrying its change ID; an
experiment script from e43 on without a page, or a page whose script is not there; a [[link]] to no page; a relation word outside the table; a relation without evidence; an experiment
or change ID or ROADMAP heading that does not exist; an experiment page naming a concept whose page does not cite
it, or the other way; `graph.md` out of date. It warns on code named under `## In this engine` that is not found.
`tests/test_wiki.py` runs it with the other tests.
