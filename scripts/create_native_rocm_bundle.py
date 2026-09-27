from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import torch

from tools.train.plamo_ssd_hip import ROOT, SOURCES, extension_info, load_extension


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    load_extension()
    info = extension_info()
    identity = info["build_identity"]
    bundle = ROOT / "artifacts" / "native-rocm-bundle" / identity
    bundle.mkdir(parents=True, exist_ok=True)

    files = [*SOURCES, ROOT / "tools" / "train" / "plamo_ssd_hip.py"]
    binary = Path(info["binary"])
    files.append(binary)
    records = []
    for source in files:
        destination = bundle / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    manifest = {
        "format": 1,
        "build_identity": identity,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "rocm": getattr(torch.version, "rocm", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": properties.name,
        "architecture": properties.gcnArchName,
        "vram_bytes": properties.total_memory,
        "source_binary_consistent": identity in binary.name,
        "files": records,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(bundle), **manifest}, indent=2))


if __name__ == "__main__":
    main()
