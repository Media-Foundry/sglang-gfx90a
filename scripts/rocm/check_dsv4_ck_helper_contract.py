#!/usr/bin/env python3
"""Check the lazy CK helper's syntax/API without importing Torch or using a GPU.

Use --revision HEAD (or a commit) to validate committed rather than dirty source.
"""
import argparse
import ast
from pathlib import Path
import subprocess

HELPER = "python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py"
CALLER = "python/sglang/srt/layers/moe/moe_runner/aiter.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    def read(path):
        if args.revision:
            return subprocess.check_output(
                ["git", "show", f"{args.revision}:{path}"], cwd=root, text=True
            )
        return (root / path).read_text()

    tree = ast.parse(read(HELPER), filename=HELPER)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    function = functions["gfx90a_bf16_ck_moe"]
    assert functions["_logical_a16w4_scales"].end_lineno < function.lineno
    assert [a.arg for a in function.args.args] == [
        "hidden", "topk_ids", "topk_weights", "w13", "s13", "w2", "s2"
    ]
    keywords = {a.arg: default for a, default in zip(
        function.args.kwonlyargs, function.args.kw_defaults
    )}
    assert "scales_shuffled" in keywords, "missing keyword-only layout argument"
    default = keywords["scales_shuffled"]
    assert isinstance(default, ast.Constant) and default.value is False, (
        "scales_shuffled must default to boolean False, not (False,)"
    )
    # A stray body-level annotated assignment must not shadow the caller's flag.
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.target.id != "scales_shuffled", "layout flag shadowed in body"
    caller = ast.parse(read(CALLER), filename=CALLER)
    calls = [node for node in ast.walk(caller) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == function.name]
    assert calls, "production caller not found"
    allowed = set(keywords) | {a.arg for a in function.args.args}
    for call in calls:
        actual = {keyword.arg for keyword in call.keywords}
        assert "scales_shuffled" in actual, "production caller must specify scale layout"
        assert actual <= allowed, actual - allowed
    print(f"PASS: {args.revision or 'worktree'} syntax, helper placement, boolean "
          f"keyword-only layout flag, {len(calls)} production caller(s)")


if __name__ == "__main__":
    main()
