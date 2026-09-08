#!/usr/bin/env python3
"""Summarize complete eight-rank realtime samples, not E2E throughput.

Use separate line ranges for C1/C32; logger replay IDs do not encode batch size.
Stage rank-max medians must not be added as a critical-path estimate.
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--start-line', type=int, default=1)
    p.add_argument('--stop-line', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-detail', action='store_true',
                   help='Require slots28..31 for output-stage breakdown')
    a = p.parse_args()
    pattern = re.compile(r'gfx90a realtime layer trace: rank=(\d+) ticks=(\[.*?\]).* replay=(\d+)')
    samples = {}
    invalid = 0
    for line in a.log.read_text(errors='replace').splitlines()[a.start_line-1:a.stop_line]:
        if 'Invalid gfx90a realtime' in line:
            invalid += 1
            continue
        match = pattern.search(line)
        if match:
            rank, ticks, replay = match.groups()
            samples.setdefault(int(replay), {})[int(rank)] = json.loads(ticks)
    spans = {'whole_layer':(0,7), 'attention_mhc_norm':(0,1),
             'attention_entry_gap':(1,2), 'attention_prepare':(2,3),
             'attention_core':(3,4), 'attention_output_collective':(4,5),
             'ffn_mhc_norm':(5,6), 'moe_collective':(6,7),
             'moe_16_17':(16,17), 'moe_17_18':(17,18),
             'moe_18_19':(18,19), 'moe_23_24':(23,24)}
    if a.output_detail:
        spans.update({'inverse_rope':(4,29), 'wo_a':(29,30),
                      'wo_b_matmul':(30,31), 'wo_b_collective':(31,28),
                      'output_tail':(28,5)})
    complete = {r:v for r,v in samples.items() if set(v)==set(range(8))}
    if a.output_detail:
        assert complete, 'no complete eight-rank samples'
        order = [4,29,30,31,28,5]
        assert all(all(t[i] > 0 for i in order) and
                   all(t[hi] >= t[lo] for lo,hi in zip(order,order[1:]))
                   for ranks in complete.values() for t in ranks.values()), (
            'missing or nonmonotonic output-stage markers')
    stats = {}
    for name,(lo,hi) in spans.items():
        values = [max((t[hi]-t[lo])*.04 for t in ranks.values())
                  for ranks in complete.values()
                  if all(t[lo]>0 and t[hi]>=t[lo] for t in ranks.values())]
        stats[name] = {'count':len(values), 'rankmax_median_us':statistics.median(values) if values else None}
    result = {'log':str(a.log),'start_line':a.start_line,'stop_line':a.stop_line,
              'complete_samples':len(complete),'incomplete_samples':len(samples)-len(complete),
              'invalid_log_lines':invalid,'tick_us':.04,'spans':stats,'samples':complete}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':
    main()
