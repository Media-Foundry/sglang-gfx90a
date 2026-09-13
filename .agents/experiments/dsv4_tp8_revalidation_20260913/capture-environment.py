"""Read-only package/build/model metadata; no device initialization or HTTP."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def command(args):
    proc = subprocess.run(args, text=True, capture_output=True, timeout=30)
    return dict(argv=args, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def main():
    root = Path(__file__).resolve().parent
    versions = {}
    for name in ('torch', 'triton', 'transformers', 'tokenizers', 'aiter', 'sglang-kernel'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'distribution metadata unavailable'
    model = Path('/home/pc/models/modelscope')
    files = {}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'generation_config.json', 'model.safetensors.index.json'):
        path = model/name
        if path.is_file():
            content = path.read_bytes()
            files[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    builds = []
    for path in ('/home/pc/Code/sglang', '/home/pc/pytorch/third_party/aiter',
                 '/home/pc/Code/composable_kernel'):
        builds.append(command(['git', '-C', path, 'rev-parse', 'HEAD']))
        builds.append(command(['git', '-C', path, 'status', '--short', '--untracked-files=no']))
    builds.append(command(['/opt/rocm/bin/hipcc', '--version']))
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=versions, model_path=str(model), model_metadata=files, builds=builds,
                  caveat='Captured during the matrix; service execution source is documented separately. Metadata hashes are not full weight-file hashes. /opt/rocm/bin/hipcc reports the installed compiler, not the provenance of every cached binary; dirty tracked files are listed separately.')
    (root/'environment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
