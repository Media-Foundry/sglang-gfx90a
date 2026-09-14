"""Archive diagnostic evidence, never the checkpoint weight/activation dumps."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence.tar.gz"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p.startswith("trace-") or p == "__pycache__"
                                     for p in relative.parts):
            continue
        if path.suffix not in (".json", ".log", ".sh", ".patch"):
            continue
        files.append(path)
    inventory = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    with tarfile.open(OUT, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    (ROOT / "evidence-manifest.json").write_text(json.dumps({
        "files": inventory, "archive_bytes": OUT.stat().st_size,
        "archive_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
