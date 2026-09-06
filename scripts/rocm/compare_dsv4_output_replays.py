"""Compare saved fixed-input E2E replays, including completion logprobs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.captures]
    reference = records[0]
    failed = False
    for path, record in zip(args.captures[1:], records[1:]):
        for key in ("input_manifest_sha256", "request_count", "request_offset", "tokens"):
            assert reference[key] == record[key], (path, key)
        counts = {}
        differences = []
        for field in ("output_ids", "output_token_logprobs", "output_top_logprobs", "text"):
            counts[field] = sum(a[field] == b[field] for a, b in zip(reference["rows"], record["rows"]))
        for i, (a, b) in enumerate(zip(reference["rows"], record["rows"])):
            for field in ("output_ids", "output_token_logprobs", "output_top_logprobs"):
                if a[field] != b[field]:
                    first = next((j for j, (x, y) in enumerate(zip(a[field], b[field])) if x != y), None)
                    differences.append(dict(request=i, field=field, first_output_index=first))
        failed |= bool(differences)
        print(json.dumps(dict(capture=str(path), exact_requests=counts,
                              cached_tokens=[r["cached_tokens"] for r in record["rows"]],
                              differences=differences)), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
