#!/usr/bin/env python3
"""Compare the bundled V4.1 encoder to local checkpoint golden text fixtures.

No checkpoint Python is executed. Alias budgets are read as AST literals.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv41


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/media/PM983/deepseek-v4.1-flash", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"passed": False, "model_dir": str(args.model_dir), "fixtures": []}
    with args.output.open("x") as handle:
        try:
            profile = chat_encoding.resolve_dsv41_effort_profile(str(args.model_dir))
            report["effort_profile"] = profile
            fixtures = sorted((args.model_dir / "encoding/tests").glob("test_input_*.json"))
            assert fixtures, "No checkpoint golden fixtures found"
            for path in fixtures:
                case = json.loads(path.read_text())
                if isinstance(case, list):
                    case = {"messages": case}
                messages = copy.deepcopy(case["messages"])
                if case.get("tools"):
                    if messages[0]["role"] != "system":
                        messages.insert(0, {"role": "system", "content": ""})
                    messages[0]["tools"] = case["tools"]
                actual = encoding_dsv41.encode_messages(
                    # Checkpoint test_encoding.py calls encode_case(...,
                    # thinking_mode="chat"); explicit per-case mode overrides it.
                    messages, thinking_mode=case.get("thinking_mode", "chat"),
                    reasoning_effort=chat_encoding.dsv41_effort_budget(case.get("reasoning_effort"), profile),
                )
                expected_path = path.with_name(path.name.replace("input", "output")).with_suffix(".txt")
                expected = expected_path.read_text()
                report["fixtures"].append({"name": path.name, "exact": actual == expected,
                    "characters": len(actual),
                    "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
                    "actual_sha256": hashlib.sha256(actual.encode()).hexdigest()})
            report["passed"] = all(item["exact"] for item in report["fixtures"])
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
