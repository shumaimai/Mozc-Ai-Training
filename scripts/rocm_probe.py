#!/usr/bin/env python3
import shutil
import subprocess
import sys

print("python", sys.version)
print("executable", sys.executable)

try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print("matmul_ok", float(y.mean()))
except Exception as exc:
    print("torch_error", type(exc).__name__, exc)

try:
    import bitsandbytes as bnb
    print("bitsandbytes", getattr(bnb, "__version__", "imported"))
except Exception as exc:
    print("bitsandbytes_error", type(exc).__name__, exc)

for cmd in (["rocm-smi"], ["rocminfo"]):
    exe = shutil.which(cmd[0])
    print("which", cmd[0], exe)
    if not exe:
        continue
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        print(out[:1500])
    except Exception as exc:
        print(cmd[0], "failed", exc)

# write test
from pathlib import Path
Path("/tmp/rocm_probe_ok").write_text("ok\n", encoding="utf-8")
print("wrote /tmp/rocm_probe_ok")
