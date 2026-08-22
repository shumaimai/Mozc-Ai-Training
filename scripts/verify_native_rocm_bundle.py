from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        path = args.bundle / record["name"]
        if not path.is_file():
            failures.append(f"missing: {record['name']}")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {record['name']}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"hash mismatch: {record['name']}")
    if not manifest.get("source_binary_consistent"):
        failures.append("binary name does not match source build identity")
    report = {"passed": not failures, "bundle": str(args.bundle), "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
