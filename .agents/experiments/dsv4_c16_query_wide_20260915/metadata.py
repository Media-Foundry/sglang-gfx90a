"""CPU-only initial synthetic geometry accounting; not measured HBM traffic."""
def canonical_tie_ids(length,k=512):
    n=min(max(length,0),k)
    return list(range(n-1,-1,-1))+[-1]*(k-n)

def tile_stats(lengths,owners,width,bq=16,bs=16):
    assert len(lengths)==len(owners) and lengths and width>0
    assert all(0<=n<=width for n in lengths)
    launched=((len(lengths)+bq-1)//bq)*((width+bs-1)//bs)
    nonempty=shared=loads=baseline=0
    for start in range(0,len(lengths),bq):
        counts={};row_tiles=[]
        for n,owner in zip(lengths[start:start+bq],owners[start:start+bq]):
            tiles=(n+bs-1)//bs if n>512 else 0
            row_tiles.append(tiles)
            if tiles:counts[owner]=max(counts.get(owner,0),tiles)
        values=sorted(counts.values(),reverse=True)
        active=values[0] if values else 0
        contested=values[1] if len(values)>1 else 0
        nonempty+=active;shared+=active-contested
        loads+=active-contested+sum(min(n,contested) for n in row_tiles)
        baseline+=sum(row_tiles)
    return dict(launched_group_tiles=launched,nonempty_group_tiles=nonempty,
        initial_shared_page_group_tiles=shared,
        nonempty_group_tile_fraction=nonempty/launched,
        initial_shared_page_fraction=shared/nonempty if nonempty else 0,
        baseline_logical_k_tile_loads=baseline,candidate_logical_k_tile_loads=loads,
        logical_k_load_reuse=baseline/loads if loads else 0,
        scope='Initial independent-owner physical pages only; excludes caches and transaction amplification, not HBM byte measurements')
