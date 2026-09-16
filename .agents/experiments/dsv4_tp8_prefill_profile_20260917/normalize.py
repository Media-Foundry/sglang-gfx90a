"""Normalize the literal measured launcher without executing the model server.

Reject shell expansions/control flow. Verify terminal environment with inert
/usr/bin/env substitution in isolated shells; never invoke the real launcher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'dsv4_ck_route_producer_service_20260917/validated-launcher.sh'
EXEC='exec bash scripts/rocm_dsv4_flash.sh serve'


def normalize(text):
    actions={};directory=None;found_exec=False
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if found_exec:raise ValueError('unexpected code after exec')
        if line=='set -euo pipefail':continue
        if any(c in line for c in ('$','`',';','&','|','<','>','\\')):
            raise ValueError('nonliteral shell input')
        words=shlex.split(line)
        if words[0]=='cd':
            assert len(words)==2 and directory is None
            directory=words[1]
        elif words[0]=='export':
            for assignment in words[1:]:
                key,value=assignment.split('=',1)
                assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key)
                actions[key]=value
        elif words[0]=='unset':
            for key in words[1:]:
                assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key)
                actions[key]=None
        elif line==EXEC:found_exec=True
        else:raise ValueError('unsupported statement: '+line)
    assert directory and found_exec
    lines=['#!/usr/bin/env bash',
        '# Literal normalized copy of the accepted TP8 native-AR prefill profile.',
        '# CPU environment-equivalent; fresh-process service replay still required.',
        'set -euo pipefail','cd '+shlex.quote(directory)]
    for key,value in sorted(actions.items()):
        lines.append('unset '+key if value is None else 'export '+key+'='+shlex.quote(value))
    return '\n'.join(lines+[EXEC,'']),actions


def resolve(text,seed):
    assert text.count(EXEC)==1
    inert=text.replace(EXEC,'exec /usr/bin/env -0')
    subprocess.run(['/bin/bash','-n'],input=inert.encode(),check=True)
    result=subprocess.check_output(['/bin/bash','--noprofile','--norc','-c',inert],env=seed)
    return dict(item.split(b'=',1) for item in result.split(b'\0') if item)


def verify(source,candidate):
    normalized,actions=normalize(source)
    assert candidate==normalized
    seeds=[{'PATH':'/usr/bin:/bin'},
        {'PATH':'/usr/bin:/bin','SGLANG_PROFILE_UNTOUCHED_SENTINEL':'preserve',
         **{key:'stale-inherited-value' for key in actions}}]
    checks=[]
    for index,seed in enumerate(seeds):
        before,after=resolve(source,seed),resolve(candidate,seed)
        assert before==after,(before.keys()^after.keys(),
                              [k for k in before.keys()&after.keys() if before[k]!=after[k]])
        if index:assert after[b'SGLANG_PROFILE_UNTOUCHED_SENTINEL']==b'preserve'
        checks.append(dict(seed=index,environment_exact=True,key_count=len(after)))
    return dict(status='cpu_environment_equivalent',service_validated=False,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        source_lines=len(source.splitlines()),candidate_lines=len(candidate.splitlines()),
        checks=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',type=Path)
    args=parser.parse_args()
    original=SOURCE.read_text()
    if args.check:print(json.dumps(verify(original,args.check.read_text()),indent=2))
    else:print(normalize(original)[0],end='')
