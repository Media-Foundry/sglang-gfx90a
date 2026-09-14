"""Archive bounded diagnostic evidence, excluding checkpoint/activation tensors."""
import hashlib
import json
from pathlib import Path
import tarfile


root = Path(__file__).resolve().parent
target = root / "prepare-evidence.tar.gz"
assert not target.exists(), "Do not overwrite a published evidence archive"
files = []
inventory = {}
for name in ("layer1-prepare", "layer0-changed-rows", "layer0-stable-qkv"):
    run = root / name
    assert (run / "complete.json").exists()
    assert (run / "prepare-summary.json").exists()
    for trace in sorted(run.glob("trace-*")):
        for path in sorted(trace.glob("*.pt")):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            inventory[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    files.extend(p for p in sorted(run.iterdir()) if p.is_file()
                 and p.suffix in (".json", ".log", ".sh", ".patch"))
manifest = root / "prepare-evidence-manifest.json"
assert not manifest.exists()
manifest.write_text(json.dumps(inventory, indent=2) + "\n")
files.append(manifest)
for name in ('qkv-order-oracle.json', 'qkv-service-oracle.json',
             'qkv-tie-analysis.json', 'qkv-shape-row-axes.json'):
    files.append(root / name)
with tarfile.open(target, "w:gz") as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(root)))
print(json.dumps({"archive": str(target), "files": len(files),
                  "bytes": target.stat().st_size,
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
