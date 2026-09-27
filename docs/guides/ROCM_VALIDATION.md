# Native Radeon validation

This guide is for reproducing the native Windows ROCm training path on Radeon
hardware and reporting results without publishing models, checkpoints, API
keys, or private training data.

## Current validated system

- Windows 11 build 26200
- Radeon RX 7800 XT (`gfx1101`, 16 GiB)
- PyTorch `2.12.0+rocm7.14.0`
- HIP `7.14.60850`
- Python 3.12
- bitsandbytes 0.50.1

The stable PLaMo path uses the HIP causal-conv kernel and the Torch SSD
fallback. The experimental HIP SSD passes synthetic parity tests but produces
NaN gradients in multi-step PLaMo QLoRA, so it is not part of the acceptance
gate.

## Setup

Install the AMD ROCm PyTorch environment first. Point `ROCM_PYTHON` at that
environment if it is not in the workspace default location.

```powershell
$env:ROCM_PYTHON = "C:\path\to\rocm-venv\Scripts\python.exe"
.\scripts\setup_native_rocm.ps1
```

Do not install PyPI `torch` over the AMD wheel. Do not enable
`expandable_segments` on ROCm.

## GPU-only validation

This command uses synthetic tensors and the committed public fixture only.

```powershell
.\scripts\run_native_rocm_tests.ps1
```

It validates:

- ROCm allocation
- NF4 forward/backward
- PagedAdamW8bit in-memory checkpoint resume
- causal-conv and SSD FP32 parity
- independent FP16/BF16 closed-form oracle
- hidden-size 4096 activation-offload stress

## Completion gate

```powershell
.\scripts\run_completion_gate.ps1 -IncludeModel
```

The model gate uses the committed synthetic fixture, writes only to
`artifacts/`, and checks finite losses and gradient norms, LoRA parameter
changes, checkpoint continuity, bundle hashes, and corruption detection.

## Kernel selection

```powershell
# Validated configuration: HIP causal-conv + Torch SSD
.\scripts\run_native_rocm_plamo.ps1 -Data .\path\to\input.jsonl -QLoRA -UseHipKernels

# Research only: known multi-step NaN gradients
.\scripts\run_native_rocm_plamo.ps1 -Data .\path\to\input.jsonl -QLoRA `
  -UseHipKernels -UseExperimentalHipSsd
```

Emergency kill switches:

```powershell
$env:MOZC_DISABLE_HIP_CONV = "1"
$env:MOZC_DISABLE_HIP_SSD = "1"
```

## Reporting results

Open the Radeon validation Issue form. Include the sanitized command output and
`train_meta.json` fields, but do not attach:

- API keys or tokens
- Hugging Face cache
- model weights, adapters, or checkpoints
- private or personal training data
- raw `artifacts/` directories

The source/binary build identity is derived from the extension sources,
PyTorch, HIP, and GPU architecture. Report that identity so results can be
compared across GPUs.
