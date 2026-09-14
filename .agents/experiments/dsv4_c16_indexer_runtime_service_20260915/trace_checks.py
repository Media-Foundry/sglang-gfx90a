"""Validate rank-local compile reuse only for matching non-M metadata."""
from collections import defaultdict
import re

PATTERN=re.compile(r'\[TP(\d+)\] DSV4 indexer compile-shape rows=(\d+) width=(\d+) '
    r'pages=(\d+) page_stride=(\d+) group=(\d+) runtime_m=([01]) '
    r'align=([0-9,]+) artifact=([0-9a-f]+) object=(\d+)')

def check_shapes(text, runtime_m):
    groups=defaultdict(list)
    for match in PATTERN.finditer(text):
        rank,m,width,pages,stride,group,mode=map(int,match.groups()[:7])
        assert mode==runtime_m and group==16
        alignment=match[8]
        groups[(rank,width,pages,stride,group,alignment)].append(
            dict(m=m,artifact=match[9],object=int(match[10])))
    assert groups,'No compile-shape records'
    result=[];proven=set()
    for key,records in sorted(groups.items()):
        rows=sorted({r['m'] for r in records})
        hashes=sorted({r['artifact'] for r in records})
        objects=sorted({r['object'] for r in records})
        if runtime_m:
            assert len(hashes)==len(objects)==1,(key,records)
        if len(rows)>1:
            if not runtime_m:assert len(hashes)>=2,(key,records)
            proven.add(key[0])
        result.append(dict(rank=key[0],width=key[1],pages=key[2],page_stride=key[3],
                           group=key[4],alignment=key[5],rows=rows,artifacts=hashes,objects=objects))
    assert proven==set(range(8)),('Missing multiple-M proof for ranks',set(range(8))-proven)
    return result
