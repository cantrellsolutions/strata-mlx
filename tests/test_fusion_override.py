import pytest
from strata_mlx import serve


def test_explicit_fusion_override_preserves_auto_and_rejects_invalid():
    assert serve.fusion_limit("auto", 6) == 6
    assert serve.fusion_limit("auto", 8) == 8
    assert serve.fusion_limit("6", 8) == 6
    assert serve.fusion_limit("8", 6) == 8
    for value in ("0", "7", "9", "", "AUTO", "6.0"):
        with pytest.raises(ValueError, match="--fusion-limit"):
            serve.fusion_limit(value, 8)


def test_fusion_override_is_recognized_without_changing_default():
    defaults, _ = serve._options([])
    assert defaults["--fusion-limit"] == "auto"
    selected, skipped = serve._options(["--fusion-limit", "6"])
    assert (selected["--fusion-limit"], skipped) == ("6", [])


@pytest.mark.parametrize(
    "arguments", [["--fusion-limit"], ["--fusion-limit", "--wired", "off"]]
)
def test_missing_fusion_value_is_rejected(arguments):
    with pytest.raises(ValueError, match="requires a value"):
        serve._options(arguments)


@pytest.mark.parametrize("value", ["auto", "6", "8"])
def test_main_forwards_selected_fusion_before_loading_an_engine(value):
    """A fresh process avoids serve.main's intentional protocol fd redirection.

    The loader is stubbed and Engine stops at construction: no weights are read.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = """
import importlib, json, sys
from types import SimpleNamespace
from strata_mlx import serve
loader = importlib.import_module("strata_mlx.load")
loader.load = lambda *a, **k: (SimpleNamespace(expert_store=None, keeps_floats=True), {})
class Captured(Exception): pass
def engine(*args, **kwargs):
    print("FUSION=" + json.dumps(kwargs["fuse_upto"]), file=sys.stderr)
    raise Captured()
serve.Engine = engine
sys.argv = ["strata-mlx", "--native", "unused.gguf", "--fusion-limit", sys.argv[1],
            "--wired", "off", "--keep-awake", "0"]
try:
    serve.main()
except Captured:
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", script, value],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    selected = json.loads(
        next(
            line.split("=", 1)[1]
            for line in result.stderr.splitlines()
            if line.startswith("FUSION=")
        )
    )
    assert selected == (serve.FUSE_UPTO if value == "auto" else int(value))
