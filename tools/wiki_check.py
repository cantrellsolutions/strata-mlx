"""Check the knowledge base in docs/wiki and draw its relations (docs/wiki/README.md says what is checked).

    uv run python tools/wiki_check.py           # report; exit 1 on errors
    uv run python tools/wiki_check.py --write   # also rewrite docs/wiki/graph.md
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "docs" / "wiki"
RELATIONS = ("part-of", "depends-on", "costs", "limited-by", "trades-against", "tuned-by", "invalidates")
LINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
EVIDENCE = re.compile(r"(?<!\[)\[(e\d{2}|#\d+|[k-z]{8}|ROADMAP: [^\]]+)\](?!\])")
RELATION = re.compile(r"^- ([a-z-]+) → \[\[([a-z0-9-]+)\]\]: (.+)$")
CODE = re.compile(r"`([\w/]+\.py):(\w+)`")
LEGACY = 42  # experiments up to this one are in ROADMAP.md and bench/experiments, without a page here


@dataclass
class Page:
    path: Path
    id: str
    title: str
    kind: str
    text: str
    relations: list = field(default_factory=list)  # (relation, target, note, evidence)
    concepts: list = field(default_factory=list)  # an experiment's "## Concepts"


def front(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    head = text[4 : text.index("\n---", 4)]
    return {k.strip(): v.split("#")[0].strip() for k, v in (line.split(":", 1) for line in head.splitlines() if ":" in line)}


def section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def read(folder: str) -> dict:
    pages = {}
    for path in sorted((WIKI / folder).glob("*.md")):
        text = path.read_text()
        meta = front(text)
        title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), path.stem)
        page = Page(path, meta.get("id", path.stem), title, meta.get("kind", "experiment"), text)
        for line in section(text, "Relations").splitlines():
            m = RELATION.match(line)
            if m:
                page.relations.append((m.group(1), m.group(2), m.group(3), EVIDENCE.findall(m.group(3))))
            elif line.startswith("- "):
                page.relations.append((None, None, line, []))
        page.concepts = LINK.findall(section(text, "Concepts"))
        pages[page.id] = page
    return pages


def changes(ids: set) -> dict:
    """change ID -> commit, by jj; None where jj is not there."""
    if not shutil.which("jj"):
        return None
    found = {}
    for c in sorted(ids):
        r = subprocess.run(["jj", "log", "--ignore-working-copy", "-r", c, "--no-graph", "-T", 'commit_id.short(7) ++ "\\n"'],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() and set(r.stdout.strip()) != {"0"}:  # not the root
            found[c] = r.stdout.split()[0]
    return found


def check() -> tuple:
    concepts, experiments = read("concepts"), read("experiments")
    errors, warnings = [], []
    headings = [line.lstrip("#").strip() for line in (ROOT / "ROADMAP.md").read_text().splitlines() if line.startswith("#")]
    pages = list(concepts.values()) + list(experiments.values())
    others = [WIKI / "index.md", WIKI / "log.md"]
    cited_changes = set()

    def where(path):
        return path.relative_to(WIKI.parent.parent) if path.is_relative_to(WIKI.parent.parent) else path

    for page in pages:
        for target in LINK.findall(page.text):
            if target not in concepts:
                errors.append(f"{where(page.path)}: [[{target}]] has no page")
        for ev in EVIDENCE.findall(page.text):
            if re.fullmatch(r"e\d{2}", ev):
                n = int(ev[1:])
                if n > LEGACY and ev not in experiments:
                    errors.append(f"{where(page.path)}: [{ev}] has no page in experiments/")
                if n <= LEGACY and not list((ROOT / "bench" / "experiments").glob(f"{ev}_*.py")):
                    errors.append(f"{where(page.path)}: [{ev}] has no script in bench/experiments/")
            elif ev.startswith("ROADMAP: "):
                start = ev[len("ROADMAP: "):]
                if not any(h.startswith(start) for h in headings):
                    errors.append(f"{where(page.path)}: [{ev}] is no heading of ROADMAP.md")
            elif re.fullmatch(r"[k-z]{8}", ev):
                cited_changes.add(ev)
        for rel, target, note, evidence in page.relations:
            if rel is None:
                errors.append(f"{where(page.path)}: not a relation line: {note[:60]}")
                continue
            if rel not in RELATIONS:
                errors.append(f"{where(page.path)}: '{rel}' is not a relation ({', '.join(RELATIONS)})")
            if not evidence:
                errors.append(f"{where(page.path)}: {rel} → {target} has no evidence")
        if page.id in concepts:
            for file, name in CODE.findall(section(page.text, "In this engine")):
                src = ROOT / file
                if not src.exists() or not re.search(rf"\b(def|class) {name}\b|^{name}\b|\b{name}\s*=", src.read_text(), re.M):
                    warnings.append(f"{where(page.path)}: `{file}:{name}` not found")
    for path in others:
        text = path.read_text()
        cited_changes |= {ev for ev in EVIDENCE.findall(text) if re.fullmatch(r"[k-z]{8}", ev)}
        for target in LINK.findall(text):
            if target not in concepts:
                errors.append(f"{where(path)}: [[{target}]] has no page")

    # an experiment and the concepts it names cite each other
    for e in experiments.values():
        for c in e.concepts:
            if c in concepts and f"[{e.id}]" not in concepts[c].text:
                errors.append(f"{where(e.path)} names [[{c}]], whose page does not cite [{e.id}]")
    for c in concepts.values():
        for ev in set(EVIDENCE.findall(c.text)):
            if ev in experiments and c.id not in experiments[ev].concepts:
                errors.append(f"{where(c.path)} cites [{ev}], whose page does not name [[{c.id}]] under ## Concepts")

    found = changes(cited_changes)
    if found is None:
        warnings.append("jj is not installed: change IDs not checked")
    else:
        for c in sorted(cited_changes - set(found)):
            errors.append(f"change {c} is not in this repository")
    return concepts, experiments, errors, warnings, found or {}


def graph(concepts: dict, experiments: dict, found: dict) -> str:
    def node(c):
        return f'  {c.id.replace("-", "_")}["{c.title}"]'

    out = ["# The relations", "",
           "> 中文摘要:概念之间的关系图,由 `tools/wiki_check.py --write` 生成,不要手改。箭头从一个概念指向它影响或依赖的概念。",
           "", "Written by `tools/wiki_check.py --write` from the `## Relations` of every concept page. Do not edit.",
           "", "```mermaid", "flowchart LR"]
    out += [node(c) for c in concepts.values()]
    for c in concepts.values():
        for rel, target, _, _ in c.relations:
            if rel:
                out.append(f'  {c.id.replace("-", "_")} -- {rel} --> {target.replace("-", "_")}')
    kinds = sorted({c.kind for c in concepts.values()})
    colours = dict(zip(["component", "concept", "constraint", "hardware", "mechanism", "method", "metric"],
                       ["#dbeafe", "#f3f4f6", "#fee2e2", "#fef3c7", "#dcfce7", "#ede9fe", "#e0f2fe"]))
    for k in kinds:
        out.append(f"  classDef {k} fill:{colours.get(k, '#ffffff')},stroke:#555")
        ids = ",".join(c.id.replace("-", "_") for c in concepts.values() if c.kind == k)
        out.append(f"  class {ids} {k}")
    out += ["```", "", "Colours by kind: " + ", ".join(kinds) + ".", "",
            "| from | relation | to | evidence |", "| :--- | :--- | :--- | :--- |"]
    for c in concepts.values():
        for rel, target, _, evidence in c.relations:
            if rel:
                ev = ", ".join(f"{e} ({found[e]})" if e in found else e for e in evidence)
                out.append(f"| [{c.title}](concepts/{c.id}.md) | {rel} | [{concepts[target].title if target in concepts else target}]"
                           f"(concepts/{target}.md) | {ev} |")
    out += ["", "| experiment | concepts |", "| :--- | :--- |"]
    for e in experiments.values():
        out.append(f"| [{e.title}](experiments/{e.path.name}) | " + ", ".join(e.concepts) + " |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="rewrite docs/wiki/graph.md")
    args = p.parse_args()
    concepts, experiments, errors, warnings, found = check()
    drawn = graph(concepts, experiments, found)
    target = WIKI / "graph.md"
    if args.write:
        target.write_text(drawn)
    elif not target.exists() or target.read_text() != drawn:
        errors.append("docs/wiki/graph.md is out of date: run with --write")
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    edges = sum(1 for c in concepts.values() for r in c.relations if r[0])
    print(f"{len(concepts)} concepts, {edges} relations, {len(experiments)} experiments, "
          f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
