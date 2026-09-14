"""Compare repeated service outputs only up to their first shared-prefix split."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def ids(row):
    count=row['meta_info']['completion_tokens']
    result=row['output_ids'][-count:]
    assert len(result)==count and count>0
    return result


def compare(a,b):
    rows=[]
    for index,(x,y) in enumerate(zip(a,b,strict=True)):
        p,q=ids(x),ids(y)
        split=next((j for j,(u,v) in enumerate(zip(p,q)) if u!=v),None)
        assert len(p)==len(q)
        item=dict(request=index,exact=p==q,first_split=split)
        until=len(p) if split is None else split
        left=x['meta_info']['output_token_logprobs'][:until]
        right=y['meta_info']['output_token_logprobs'][:until]
        assert len(left)==len(right)==until
        item['shared_prefix_logprobs_exact']=left==right
        item['shared_prefix_max_abs_logprob_delta']=max(
            (abs(u[0]-v[0]) for u,v in zip(left,right,strict=True)),default=0.0)
        if split is not None:
            for label,row in [('left',x),('right',y)]:
                top=row['meta_info']['output_top_logprobs'][split]
                item[label+'_top5']=top
                values=sorted((v[0] for v in top),reverse=True)
                item[label+'_margin']=values[0]-values[1]
        rows.append(item)
    return dict(exact=sum(x['exact'] for x in rows),requests=len(rows),
                shared_prefix_logprobs_exact=sum(x['shared_prefix_logprobs_exact'] for x in rows),rows=rows)


def main():
    result={}
    waves={}
    for mode in ('atomic','fixed'):
        root=ROOT/mode
        assert (root/'complete.json').exists()
        waves[mode]=[json.loads((root/f'batch-{j}.json').read_text()) for j in (1,2,3)]
        paired={f'{i+1}-{j+1}':compare(waves[mode][i],waves[mode][j])
                for i,j in itertools.combinations(range(3),2)}
        concurrent=json.loads((root/'concurrent-quality.json').read_text())['rounds']
        result[mode]=dict(controlled_batch_repeats=paired,
            concurrent_repeat_exact=sum(a==b for a,b in zip(concurrent[0]['completion_ids'],concurrent[1]['completion_ids'],strict=True)),
            prefill_rates=[r['aggregate_input_tok_s'] for r in json.loads((root/'prefill.json').read_text())['rounds']],
            france=json.loads((root/'France.json').read_text())['text'])
        result[mode]['controlled_forward_entry_groups']=[
            [[i for i,row in enumerate(wave) if row['meta_info']['forward_entry_time']==stamp]
             for stamp in sorted({row['meta_info']['forward_entry_time'] for row in wave})]
            for wave in waves[mode]]
        groups=[]
        for wave in concurrent:
            order=sorted(range(16),key=lambda i:wave['request_ttft_s'][i])
            groups.append([sorted(order[i:i+4]) for i in range(0,16,4)])
        result[mode]['concurrent_groups_inferred_from_ttft']=groups
    result['cross_path_first_wave']=compare(waves['atomic'][0],waves['fixed'][0])
    a=statistics.median(result['atomic']['prefill_rates'])
    b=statistics.median(result['fixed']['prefill_rates'])
    result['performance']=dict(atomic_median=a,fixed_median=b,change_pct=100*(b/a-1),
        scope='Diagnostic A/B, not a performance-acceptance ABBA.')
    matched=ROOT/'atomic-matched'
    assert (matched/'complete.json').exists(), 'Matched logprob control must complete first'
    batched=[json.loads((matched/f'batch-{j}.json').read_text()) for j in (1,2,3)]
    split=[json.loads((matched/f'split-lp-{j}.json').read_text()) for j in (0,1)]
    def groups(wave):
        return [[i for i,row in enumerate(wave) if row['meta_info']['forward_entry_time']==stamp]
                for stamp in sorted({row['meta_info']['forward_entry_time'] for row in wave})]
    result['matched_logprob_control']=dict(
        batched_pairs={f'{i+1}-{j+1}':compare(batched[i],batched[j]) for i,j in itertools.combinations(range(3),2)},
        split_pair=compare(split[0],split[1]),
        batched_groups=[groups(w) for w in batched],split_groups=[groups(w) for w in split],
        scope='Both submission protocols request output logprobs/top5 with identical sampling settings.')
    manifest={}
    for root in (ROOT,ROOT/'atomic',ROOT/'fixed',matched):
        for path in sorted(root.iterdir()):
            if path.is_file() and path.name not in ('summary.json','measurement-evidence.tar.gz','evidence-index.json'):
                manifest[str(path.relative_to(ROOT))]=dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result['artifact_manifest']=manifest
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode in ('atomic','fixed'):
        print(mode, 'controlled pairs', {k:v['exact'] for k,v in result[mode]['controlled_batch_repeats'].items()},
              'concurrent',result[mode]['concurrent_repeat_exact'])
    print(result['performance'])
    print('matched_logprob_control',
          {k:v['exact'] for k,v in result['matched_logprob_control']['batched_pairs'].items()},
          result['matched_logprob_control']['split_pair']['exact'])


if __name__=='__main__':main()
