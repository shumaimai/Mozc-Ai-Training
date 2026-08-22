from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate whether QLoRA base weights fit VRAM")
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument("--parameters", type=float, help="override parameter count")
    parser.add_argument("--reserve-gb", type=float, default=3.0)
    args = parser.parse_args()

    parameter_count = args.parameters
    if parameter_count is None:
        config = AutoConfig.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=True
        )
        parameter_count = getattr(config, "num_parameters", None)
        if callable(parameter_count):
            parameter_count = parameter_count()
    if parameter_count is None:
        raise SystemExit(
            "The model config does not expose a parameter count; pass --parameters."
        )

    parameter_count = float(parameter_count)
    # NF4 data is 0.5 bytes/parameter. Scales, double-quant metadata, and
    # non-quantized modules vary by architecture, so use a conservative 0.65.
    quantized_gb = parameter_count * 0.65 / 1024**3
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    budget_gb = max(0.0, vram_gb - args.reserve_gb)
    report = {
        "model": args.model,
        "parameters": int(parameter_count),
        "estimated_nf4_base_gb": round(quantized_gb, 2),
        "vram_gb": round(vram_gb, 2),
        "reserved_for_runtime_gb": args.reserve_gb,
        "base_weight_budget_gb": round(budget_gb, 2),
        "fits_supported_qlora_path": quantized_gb <= budget_gb,
    }
    print(json.dumps(report, indent=2))
    if quantized_gb > budget_gb:
        raise SystemExit(
            "The quantized base itself exceeds the safe GPU budget. Native Windows "
            "ROCm has no supported training-aware CPU/disk weight streaming backend; "
            "use a smaller model, a larger GPU, or Linux DeepSpeed ZeRO-3."
        )


if __name__ == "__main__":
    main()
