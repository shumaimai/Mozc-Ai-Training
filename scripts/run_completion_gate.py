from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ARTIFACTS = ROOT / "artifacts"


def run(name: str, command: list[str], timeout: int = 600) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return {
            "name": name,
            "passed": result.returncode == 0,
            "seconds": round(time.perf_counter() - started, 3),
            "returncode": result.returncode,
            "output_tail": output[-8000:],
        }
    except subprocess.TimeoutExpired as error:
        output = ((error.stdout or "") + "\n" + (error.stderr or "")).strip()
        return {
            "name": name,
            "passed": False,
            "seconds": round(time.perf_counter() - started, 3),
            "returncode": None,
            "timeout": timeout,
            "output_tail": output[-8000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    checks = [
        ("rocm_health", [python, "-c", "import torch; assert torch.version.hip and torch.cuda.is_available(); x=torch.ones(1024,device='cuda'); torch.cuda.synchronize(); print(torch.__version__,torch.version.hip,torch.cuda.get_device_name(0),x.sum().item())"]),
        ("nf4", [python, str(SCRIPTS / "probe_bnb_rocm.py")]),
        ("paged_optimizer_resume", [python, str(SCRIPTS / "probe_paged_optimizer_rocm.py")]),
        ("fp32_kernel_parity", [python, str(SCRIPTS / "test_plamo_fallback_math.py")]),
        ("fp16_bf16_oracle", [python, str(SCRIPTS / "test_plamo_low_precision_oracle.py")]),
        ("qlora_memory", [python, str(SCRIPTS / "stress_qlora_memory.py"), "--hidden", "4096", "--tokens", "512", "--repeats", "8", "--activation-offload"]),
    ]
    results = [run(name, command) for name, command in checks]

    from tools.train.plamo_ssd_hip import extension_info, load_extension

    load_extension()
    info = extension_info()
    bundle_dir = ARTIFACTS / "native-rocm-bundle" / info["build_identity"]
    results.append(run("bundle_create", [python, str(SCRIPTS / "create_native_rocm_bundle.py")]))
    results.append(
        run("bundle_verify", [python, str(SCRIPTS / "verify_native_rocm_bundle.py"), str(bundle_dir)])
    )
    with tempfile.TemporaryDirectory(prefix="rocm-bundle-negative-") as temporary:
        corrupt = Path(temporary) / "bundle"
        shutil.copytree(bundle_dir, corrupt)
        target = next(path for path in corrupt.iterdir() if path.name != "manifest.json")
        with target.open("ab") as handle:
            handle.write(b"intentional-negative-test")
        negative = run(
            "bundle_negative_internal",
            [python, str(SCRIPTS / "verify_native_rocm_bundle.py"), str(corrupt)],
        )
        results.append(
            {
                "name": "bundle_corruption_detected",
                "passed": not negative["passed"],
                "verifier_returncode": negative["returncode"],
            }
        )

    if args.include_model:
        out = ARTIFACTS / "completion-gate" / "current"
        if out.exists():
            archive = ARTIFACTS / "_trash" / f"completion-gate-previous-{int(time.time())}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(archive))
        out.mkdir(parents=True, exist_ok=True)
        fixture = ROOT / "tests" / "fixtures" / "lora_smoke.jsonl"
        base = [
            python, "-m", "tools.train.lora_sft",
            "--data", str(fixture),
            "--model", "pfnet/plamo-2-1b",
            "--trust-remote-code",
            "--revision", "92c75fd6eea9018bcb9c33ee8921589febe071fa",
            "--out", str(out),
            "--epochs", "1",
            "--batch-size", "1",
            "--grad-accum", "1",
            "--max-length", "64",
            "--max-steps", "4",
            "--save-steps", "1",
            "--logging-steps", "1",
            "--seed", "42",
            "--deterministic",
            "--limit", "8",
            "--load-in-4bit",
            "--activation-offload",
            "--optimizer", "adamw_torch",
            "--use-hip-kernels",
        ]
        results.append(run("plamo_three_steps", [*base, "--stop-after-steps", "3"], timeout=1800))
        meta_three_path = out / "train_meta_step_3.json"
        results.append(run("plamo_resume", base, timeout=1800))
        adapter = out / "adapter"
        checkpoints = list(out.glob("checkpoint-*"))
        meta_four_path = out / "train_meta_step_4.json"
        metadata_failures = []
        if not meta_three_path.is_file() or not meta_four_path.is_file():
            metadata_failures.append("per-run metadata missing")
        else:
            meta_three = json.loads(meta_three_path.read_text(encoding="utf-8"))
            meta_four = json.loads(meta_four_path.read_text(encoding="utf-8"))
            if not meta_three.get("trainable_parameters_changed"):
                metadata_failures.append("three-step trainable parameters did not change")
            if not meta_four.get("trainable_parameters_changed"):
                metadata_failures.append("resume-step trainable parameters did not change")
            if meta_three.get("final_trainable_fingerprint") != meta_four.get("initial_trainable_fingerprint"):
                metadata_failures.append("checkpoint parameter continuity failed")
            losses = [row["loss"] for row in [*meta_three.get("loss_history", []), *meta_four.get("loss_history", [])]]
            if not losses or not all(isinstance(value, (int, float)) and value == value for value in losses):
                metadata_failures.append("finite loss history missing")
            grad_norms = [
                row["grad_norm"]
                for row in [*meta_three.get("grad_norm_history", []), *meta_four.get("grad_norm_history", [])]
            ]
            if not grad_norms or not all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf") for value in grad_norms):
                metadata_failures.append("finite gradient norm history missing")
            if meta_three.get("global_step") != 3 or meta_four.get("global_step") != 4:
                metadata_failures.append("global step continuity failed")
        results.append(
            {
                "name": "plamo_outputs",
                "passed": adapter.is_dir() and bool(checkpoints) and not metadata_failures,
                "adapter": str(adapter),
                "checkpoints": [str(path) for path in checkpoints],
                "metadata_failures": metadata_failures,
            }
        )

    required_exclusions = {
        "rccl_multi_gpu": "Only one GPU is installed and native Windows RCCL is unavailable.",
        "triton_inductor": "PyTorch 2.12 native Windows ROCm has no supported Triton package.",
        "driver_private_runtime": "AMD Windows PAL/HIP runtime internals are not public source.",
    }
    report = {
        "scope": "RX 7800 XT single-GPU native Windows ROCm acceptance",
        "passed": all(result["passed"] for result in results),
        "checks": results,
        "external_exclusions": required_exclusions,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "native-rocm-completion-report.json"
    payload = json.dumps(report, indent=2)
    output.write_text(payload, encoding="utf-8")
    history = ARTIFACTS / "completion-reports"
    history.mkdir(parents=True, exist_ok=True)
    (history / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(
        payload, encoding="utf-8"
    )
    print(json.dumps({"report": str(output), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
