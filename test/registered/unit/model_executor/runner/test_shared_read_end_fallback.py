"""CPU-only evaluation of the actual graph runner's shared-read contract."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest


class ReadEnd(Enum):
    UNKNOWN = 0
    PRE_REPLAY = 1
    IN_REPLAY = 2
    POST_REPLAY = 3


class TestSharedReadEndFallback(unittest.TestCase):
    def test_missing_in_graph_event_never_publishes_early(self):
        path = Path(__file__).resolve().parents[5] / (
            "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
        )
        tree = ast.parse(path.read_text())
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_resolve_shared_read_ends"]
        self.assertEqual(len(methods), 1)
        method = methods[0]
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        namespace = {"SharedReadEnds": ReadEnd}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        resolve = namespace[method.name]
        for declared in ReadEnd:
            for event in (None, object()):
                actual = resolve(
                    SimpleNamespace(in_graph_metadata_prep_done=event),
                    SimpleNamespace(shared_read_ends=lambda mode: declared),
                    object(),
                )
                expected = (ReadEnd.POST_REPLAY if declared is ReadEnd.IN_REPLAY
                            and event is None else declared)
                self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
