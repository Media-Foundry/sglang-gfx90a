"""Package completed trial data, never checkpoint or tensor cache files."""
import argparse
import hashlib
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase',choices=('pilot','abba'),required=True)
args=parser.parse_args()
names=['pilot'] if args.phase=='pilot' else ['A1','B','A2']
target=root/(args.phase+'-evidence.tar.gz')
assert not target.exists()
files=[]
for name in names:
    assert (root/name/'complete.json').exists()
    files += [p for p in sorted((root/name).iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
if args.phase=='pilot':files += [root/'component.json',root/'component.log']
with tarfile.open(target,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
print(dict(files=len(files),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
