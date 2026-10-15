"""The knowledge base in docs/wiki holds together, and its checker finds what it says it finds."""

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("wiki_check", ROOT / "tools" / "wiki_check.py")
wiki_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wiki_check)


def test_the_knowledge_base_has_no_errors_and_its_graph_is_current():
    concepts, experiments, errors, _, found = wiki_check.check()
    assert errors == []
    assert (wiki_check.WIKI / "graph.md").read_text() == wiki_check.graph(concepts, experiments, found)


@pytest.fixture
def copy(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    shutil.copytree(wiki_check.WIKI, wiki)
    monkeypatch.setattr(wiki_check, "WIKI", wiki)
    return wiki


def errors_after(page: Path, old: str, new: str) -> list:
    text = page.read_text()
    assert old in text
    page.write_text(text.replace(old, new, 1))
    return wiki_check.check()[2]


@pytest.mark.parametrize("page, old, new, expected", [
    ("decode-pass", "[[hyper-connections]]", "[[no-such-concept]]", "has no page"),
    ("hyper-connections", "- part-of → [[decode-pass]]", "- slows → [[decode-pass]]", "is not a relation"),
    ("hyper-connections", "[e06]\n", "\n", "has no evidence"),
    ("hyper-connections", "[e18]", "[e99]", "has no page in experiments/"),
    ("kernel-fixed-cost", "[wykuvxvr]", "[kqkqkqkq]", "is not in this repository"),
    ("decode-pass", "[ROADMAP: Faster answers]", "[ROADMAP: No such heading]", "is no heading"),
    ("decode-pass", "area: answers", "area: somewhere", "is not one of"),
])
def test_the_checker_finds_each_kind_of_error(copy, page, old, new, expected):
    errors = errors_after(copy / "concepts" / f"{page}.md", old, new)
    assert any(expected in e for e in errors), errors


def test_an_experiment_and_its_concepts_must_cite_each_other(copy):
    errors = errors_after(copy / "experiments" / "e43-experts-as-qmm.md", "[[expert-lookup-kernel]]", "[[exactness]]")
    assert any("names [[exactness]], whose page does not cite [e43]" in e for e in errors)
    assert any("cites [e43], whose page does not name [[expert-lookup-kernel]]" in e for e in errors)


def test_an_experiment_page_names_a_script_that_is_there(copy):
    errors = errors_after(copy / "experiments" / "e43-experts-as-qmm.md", "script: bench/experiments/e43_experts_as_qmm.py",
                          "script: bench/experiments/e43_gone.py")
    assert any("its script bench/experiments/e43_gone.py is not there" in e for e in errors)
