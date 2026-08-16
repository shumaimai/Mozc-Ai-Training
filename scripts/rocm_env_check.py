#!/usr/bin/env python3
from pathlib import Path
import subprocess

Path("/tmp/agent_write_test").write_text("agent_can_write\n", encoding="utf-8")
print(Path("/tmp/agent_write_test").read_text(encoding="utf-8").strip())

for pkg in ("transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
    try:
        mod = __import__(pkg if pkg != "bitsandbytes" else "bitsandbytes")
        print(pkg, "version", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(pkg, "missing", type(exc).__name__, exc)

print("cwd_listing")
for p in (Path("/workspace"), Path("/home"), Path("/opt/venv")):
    print(p, "exists", p.exists())
    if p.exists() and p.is_dir():
        try:
            print(" ", [x.name for x in list(p.iterdir())[:12]])
        except Exception as exc:
            print(" ", exc)

out = subprocess.check_output(["df", "-h", "/"], text=True)
print(out)
