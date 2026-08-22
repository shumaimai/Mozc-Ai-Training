"""Minimal LoRA SFT for IME data (CPU / CUDA-ROCm / DirectML)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_TARGETS = {
    "gpt2": ["c_attn"],
    "plamo": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "default": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def patch_plamo_tied_weights_compat() -> None:
    """PLaMo uses legacy list `_tied_weights_keys`; newer transformers expects a mapping."""
    from transformers import PreTrainedModel

    original = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original is None:
        return

    def safe_get_expanded_tied_weights_keys(self, all_submodels: bool = False):  # type: ignore[no-untyped-def]
        tied = getattr(self.__class__, "_tied_weights_keys", None)
        if isinstance(tied, list):
            return {}
        return original(self, all_submodels=all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = safe_get_expanded_tied_weights_keys  # type: ignore[method-assign]


def _install_causal_conv1d_torch_stub() -> None:
    """Pure-torch stand-in when CUDA causal_conv1d extension is unavailable (ROCm)."""
    import sys
    import types

    import torch.nn.functional as F

    try:
        import causal_conv1d.causal_conv1d_interface as _ccd  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError, OSError):
        pass

    def causal_conv1d_ref(
        x,
        weight,
        bias=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
        **kwargs,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        dtype_in = x.dtype
        x = x.to(weight.dtype)
        seqlen = x.shape[-1]
        dim, width = weight.shape
        if initial_states is None:
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
        else:
            x = torch.cat([initial_states, x], dim=-1)
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
        out = out[..., :seqlen]
        if return_final_states:
            final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
            if final_states_out is not None:
                final_states_out.copy_(final_states)
            else:
                final_states_out = final_states
        out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
        return out if not return_final_states else (out, final_states_out)

    def causal_conv1d_fn(
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
        **kwargs,
    ):
        if seq_idx is not None:
            raise NotImplementedError("torch causal_conv1d stub does not support seq_idx")
        return causal_conv1d_ref(
            x=x,
            weight=weight,
            bias=bias,
            initial_states=initial_states,
            return_final_states=return_final_states,
            final_states_out=final_states_out,
            activation=activation,
        )

    def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, **kwargs):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        dtype_in = x.dtype
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)
        batch, dim, seqlen = x.shape
        width = weight.shape[1]
        state_len = conv_state.shape[-1]
        if cache_seqlens is None:
            x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
            conv_state.copy_(x_new[:, :, -state_len:])
        else:
            width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
            width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
            x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
            copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
            copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
            conv_state.scatter_(2, copy_idx, x)
        out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
        if unsqueeze:
            out = out.squeeze(-1)
        return (out if activation is None else F.silu(out)).to(dtype=dtype_in)

    def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, **kwargs):
        return causal_conv1d_update_ref(
            x, conv_state, weight, bias=bias, activation=activation, cache_seqlens=cache_seqlens
        )

    pkg = types.ModuleType("causal_conv1d")
    iface = types.ModuleType("causal_conv1d.causal_conv1d_interface")
    iface.causal_conv1d_fn = causal_conv1d_fn
    iface.causal_conv1d_ref = causal_conv1d_ref
    iface.causal_conv1d_update = causal_conv1d_update
    iface.causal_conv1d_update_ref = causal_conv1d_update_ref
    pkg.causal_conv1d_interface = iface
    sys.modules["causal_conv1d"] = pkg
    sys.modules["causal_conv1d.causal_conv1d_interface"] = iface
    print("installed pure-torch causal_conv1d stub", flush=True)


def _selective_state_update_torch(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """Pure-torch selective_state_update_ref (no einops / mamba_ssm).

    Returns (out, new_state). new_state is a fresh tensor so training autograd works;
    do not inplace-mutate the input state buffer across a scan loop.
    """
    import torch.nn.functional as F

    squeeze_heads = state.dim() == 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    ngroups = B.shape[1]
    nheads = state.shape[1]
    if dt_bias is not None:
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(dt.unsqueeze(-1) * A)
    if ngroups != nheads:
        repeat = nheads // ngroups
        B = B.repeat_interleave(repeat, dim=1)
        C = C.repeat_interleave(repeat, dim=1)
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    new_state = state * dA + dB * x.unsqueeze(-1)
    out = torch.einsum("bhdn,bhn->bhd", new_state.to(C.dtype), C)
    if D is not None:
        out = out + (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if squeeze_heads:
        out = out.squeeze(1)
        new_state = new_state.squeeze(1)
    return out, new_state


def patch_plamo_rocm_kernel_fallbacks(
    target_module=None,
    *,
    use_hip_kernels: bool = False,
    use_experimental_hip_ssd: bool = False,
) -> dict:
    """Bind missing CUDA kernels to torch fallbacks inside loaded PLaMo modeling."""
    import sys

    _install_causal_conv1d_torch_stub()

    mod = target_module
    if mod is None:
        for name, module in reversed(list(sys.modules.items())):
            if name.endswith("modeling_plamo") or name.endswith(".modeling_plamo"):
                mod = module
                break
    if mod is None:
        return {"ssd": "unpatched", "causal_conv": "unpatched", "reason": "model module not found"}

    import causal_conv1d.causal_conv1d_interface as causal_conv1d

    mod.causal_conv1d = causal_conv1d
    hip_conv = None
    status = {"ssd": "torch", "causal_conv": "torch", "reason": "HIP kernels not requested"}
    if use_hip_kernels and torch.version.hip is not None and os.environ.get("MOZC_DISABLE_HIP_CONV") != "1":
        try:
            from tools.train.plamo_ssd_hip import extension_info, make_conv_autograd_function

            hip_conv = make_conv_autograd_function()
            status["causal_conv"] = "hip"
            status.update(extension_info())
            status["reason"] = "requested and loaded"
            print("loaded fused PLaMo causal-conv HIP forward/backward", flush=True)
        except Exception as exc:
            status["reason"] = f"causal-conv HIP load failed: {type(exc).__name__}: {exc}"
            print(f"PLaMo causal-conv HIP extension unavailable: {exc}", flush=True)

    def _causal_conv1d_update(conv_state, weight, xBC):
        dtype = conv_state.dtype
        xBC = xBC.to(dtype)
        weight = weight.to(dtype)
        x = causal_conv1d.causal_conv1d_update_ref(
            x=xBC,
            conv_state=conv_state,
            weight=weight[:, 0, :],
            activation="silu",
        )
        return x, conv_state

    def _causal_conv1d(conv_state, weight, x, seq_idx):
        dtype = x.dtype
        if conv_state is not None:
            dtype = conv_state.dtype
        weight = weight.to(dtype)
        x = x.to(dtype)
        return_final_states = conv_state is not None
        if seq_idx is None:
            x, conv_state = causal_conv1d.causal_conv1d_ref(
                x=x,
                initial_states=conv_state,
                return_final_states=True,
                weight=weight[:, 0, :],
                activation="silu",
            )
        else:
            if conv_state is None:
                bsize = x.shape[0]
                dim = weight.shape[0]
                d_conv = weight.shape[-1]
                conv_state = torch.zeros(bsize, dim, d_conv - 1, dtype=x.dtype, device=x.device)
            length = x.shape[-1]
            # Width is four for PLaMo. Build four shifted views instead of
            # launching one tiny grouped convolution per token.
            conv_input = x
            width = weight.shape[-1]
            if hip_conv is not None and x.is_cuda and width <= 8:
                x = hip_conv(x, weight, seq_idx)
            else:
                out = torch.zeros_like(x)
                for kernel_index in range(width):
                    lag = width - 1 - kernel_index
                    shifted = torch.nn.functional.pad(x, (lag, 0))[..., :length]
                    if lag:
                        same_sequence = torch.zeros_like(seq_idx, dtype=torch.bool)
                        valid = torch.ones_like(seq_idx[:, lag:], dtype=torch.bool)
                        for boundary_offset in range(lag):
                            valid &= (
                                seq_idx[:, lag - boundary_offset : length - boundary_offset]
                                == seq_idx[:, lag - boundary_offset - 1 : length - boundary_offset - 1]
                            )
                        same_sequence[:, lag:] = valid
                        shifted = shifted * same_sequence[:, None, :]
                    out = out + shifted * weight[:, 0, kernel_index][None, :, None]
                x = torch.nn.functional.silu(out)
            conv_state = x.new_zeros(x.shape[0], x.shape[1], width - 1)
            tail = min(length, width - 1)
            for tail_index in range(tail):
                source_index = length - tail + tail_index
                same_sequence = seq_idx[:, source_index] == seq_idx[:, -1]
                conv_state[:, :, width - 1 - tail + tail_index] = torch.where(
                    same_sequence[:, None],
                    conv_input[:, :, source_index],
                    torch.zeros_like(conv_input[:, :, source_index]),
                )
        if return_final_states:
            return x, conv_state
        return x, None

    def ssd_update_state(ssm_state, x, dt, A, B, C, D, z, dt_bias, dt_softplus):
        assert ssm_state.dtype == torch.float32
        hidden_size_per_head = x.shape[-1]
        d_state = B.shape[-1]
        A = A[:, None, None].expand(-1, hidden_size_per_head, d_state).float()
        dt = dt[..., None].expand(-1, -1, hidden_size_per_head)
        dt_bias_e = dt_bias[:, None].expand(-1, hidden_size_per_head)
        D = D[:, None].expand(-1, hidden_size_per_head)
        out, new_state = _selective_state_update_torch(
            ssm_state,
            x,
            dt,
            A.float(),
            B,
            C,
            D.float(),
            z,
            dt_bias_e.float(),
            dt_softplus=dt_softplus,
        )
        # Inference cache path: keep buffer identity; training scan uses reassignment below.
        if ssm_state.data_ptr() == new_state.data_ptr():
            pass
        else:
            ssm_state.copy_(new_state.detach())
            ssm_state.grad = None
        return out[:, None]

    def _ssd_naive_train(
        x, dt, A, B, C, D, z, dt_bias, dt_softplus, seq_idx, ssm_state
    ):
        import torch.nn.functional as F

        length = x.shape[1]
        ys = []
        for i in range(length):
            if i != 0 and seq_idx is not None:
                ssm_state = torch.where(
                    (seq_idx[:, i - 1] != seq_idx[:, i])[:, None, None, None],
                    torch.zeros_like(ssm_state),
                    ssm_state,
                )
            dt_i = dt[:, i].float() + dt_bias.float()
            if dt_softplus:
                dt_i = F.softplus(dt_i)
            decay = torch.exp(dt_i * A.float())
            x_i = x[:, i]
            ssm_state = (
                ssm_state * decay[..., None, None]
                + dt_i[..., None, None]
                * x_i.float()[..., None]
                * B[:, i].float()[..., None, :]
            )
            C_i = C[:, i]
            out_i = torch.einsum(
                "bhdn,bhn->bhd", ssm_state.to(C_i.dtype), C_i
            )
            out_i = out_i + (x_i * D[None, :, None]).to(out_i.dtype)
            out_i = (out_i * F.silu(z[:, i])).to(x_i.dtype)
            ys.append(out_i[:, None])
        return torch.cat(ys, dim=1), ssm_state

    orig_ssd = mod.ssd_chunk_scan_combined
    hip_ssd = None
    if use_experimental_hip_ssd and torch.version.hip is not None and os.environ.get("MOZC_DISABLE_HIP_SSD") != "1":
        try:
            from tools.train.plamo_ssd_hip import extension_info, make_autograd_function

            hip_ssd = make_autograd_function()
            status["ssd"] = "hip"
            status.update(extension_info())
            status["reason"] = "requested and loaded"
            print("loaded fused PLaMo SSD HIP forward", flush=True)
        except Exception as exc:
            status["reason"] = f"SSD HIP load failed: {type(exc).__name__}: {exc}"
            print(f"PLaMo SSD HIP extension unavailable: {exc}", flush=True)
    elif use_hip_kernels:
        status["reason"] = "causal-conv HIP enabled; SSD kept on Torch for multi-step stability"

    def ssd_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        dt_softplus,
        return_final_states,
        seq_idx,
        ssm_state,
    ):
        if seq_idx is not None:
            assert seq_idx.dtype == torch.int32
            assert ssm_state is None
            assert not return_final_states
        # The optimized Mamba kernel needs chunk alignment. The sequential
        # fallback does not; padding length 64 to 256 quadruples its work.
        zero_initial_state = ssm_state is None
        if zero_initial_state:
            bsize, _, num_heads, channel = x.shape
            state = B.shape[-1]
            ssm_state = torch.zeros(
                bsize, num_heads, channel, state, dtype=torch.float32, device=x.device
            )
        if (
            hip_ssd is not None
            and x.is_cuda
            and dt_softplus
            and not return_final_states
            and zero_initial_state
        ):
            native_seq = (
                seq_idx
                if seq_idx is not None
                else torch.empty(0, device=x.device, dtype=torch.int32)
            )
            return hip_ssd(x, dt, A, B, C, D, z, dt_bias, native_seq)
        out, ssm_state = _ssd_naive_train(
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            seq_idx,
            ssm_state,
        )
        if return_final_states:
            return out, ssm_state
        return out

    mod._causal_conv1d_update = _causal_conv1d_update
    mod._causal_conv1d = _causal_conv1d
    mod.ssd_update_state = ssd_update_state
    mod.ssd_chunk_scan_combined = ssd_chunk_scan_combined
    mod._orig_ssd_chunk_scan_combined = orig_ssd
    print(f"patched PLaMo ROCm fallbacks in {getattr(mod, '__name__', mod)}", flush=True)
    print(f"plamo_kernel_status={json.dumps(status, sort_keys=True)}", flush=True)
    return status


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml  # type: ignore

        return str(torch_directml.device())
    except Exception:
        return "cpu"


def infer_target_modules(model_name: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    lower = model_name.lower()
    if "gpt2" in lower:
        return DEFAULT_TARGETS["gpt2"]
    if "plamo" in lower:
        return DEFAULT_TARGETS["plamo"]
    return DEFAULT_TARGETS["default"]


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LoRA SFT for Mozc-AI IME data")
    parser.add_argument("--data", required=True, help="train_mixed.jsonl")
    parser.add_argument("--model", default="rinna/japanese-gpt2-medium")
    parser.add_argument("--out", default="artifacts/lora_poc")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="optional row cap for smoke runs")
    parser.add_argument("--max-steps", type=int, default=-1, help="optional Trainer max_steps (-1 = epoch-driven)")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=0,
        help="stop this process after N steps while preserving the max_steps scheduler",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-hip-kernels",
        action="store_true",
        help="enable the validated HIP causal-conv kernel (SSD remains Torch)",
    )
    parser.add_argument(
        "--use-experimental-hip-ssd",
        action="store_true",
        help="enable experimental HIP SSD; known to produce NaN gradients in multi-step PLaMo QLoRA",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="optional model/repository revision for reproducible remote code",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="use bitsandbytes NF4 QLoRA (requires a matching native GPU backend)",
    )
    parser.add_argument(
        "--activation-offload",
        action="store_true",
        help="offload autograd-saved activations to system RAM during training",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw_torch", "paged_adamw_8bit"],
        default="adamw_torch",
        help="paged optimizer is useful for large trainable parameter sets",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[],
        help="LoRA target module names (auto for gpt2/plamo if omitted)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device()
    print(f"device={device}", flush=True)
    if args.trust_remote_code or "plamo" in args.model.lower():
        patch_plamo_tied_weights_compat()
        _install_causal_conv1d_torch_stub()
    rows = load_rows(Path(args.data), limit=args.limit)
    if len(rows) < 4:
        raise SystemExit(f"need at least 4 examples, got {len(rows)}")
    random.shuffle(rows)
    n_eval = max(1, int(len(rows) * args.eval_ratio))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    targets = infer_target_modules(args.model, args.target_modules or None)
    print(
        f"train={len(train_rows)} eval={len(eval_rows)} model={args.model} "
        f"targets={targets} trust_remote_code={args.trust_remote_code}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        texts = []
        for instruction, inp, output in zip(batch["instruction"], batch["input"], batch["output"]):
            prompt = inp if inp else instruction
            eos = tokenizer.eos_token or ""
            texts.append(f"{prompt}{output}{eos}")
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_ds = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=list(train_rows[0].keys())
    )
    eval_ds = Dataset.from_list(eval_rows).map(
        tokenize, batched=True, remove_columns=list(eval_rows[0].keys())
    )

    dtype = torch.float32
    model_kwargs: dict = {
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
    }
    if device == "cuda":
        # ROCm often prefers float16; bf16 support varies by arch.
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        model_kwargs["torch_dtype"] = dtype
    if args.load_in_4bit:
        if device != "cuda":
            raise SystemExit("--load-in-4bit requires a CUDA or ROCm GPU")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    kernel_status = {"ssd": "not-applicable", "causal_conv": "not-applicable"}
    if args.trust_remote_code or "plamo" in args.model.lower():
        kernel_status = patch_plamo_rocm_kernel_fallbacks(
            use_hip_kernels=args.use_hip_kernels,
            use_experimental_hip_ssd=args.use_experimental_hip_ssd,
        )
    if device == "cuda" and not args.load_in_4bit:
        model = model.to("cuda")

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()

    def parameter_fingerprint() -> str:
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_fingerprint = parameter_fingerprint()

    class FingerprintCallback(TrainerCallback):
        def __init__(self) -> None:
            self.train_begin_fingerprint: str | None = None

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            self.train_begin_fingerprint = parameter_fingerprint()

    fingerprint_callback = FingerprintCallback()

    class StopAfterStepCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if state.global_step >= args.stop_after_steps:
                control.should_training_stop = True
            return control

    callbacks = [fingerprint_callback]
    if args.stop_after_steps:
        callbacks.append(StopAfterStepCallback())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_fp16 = device == "cuda" and dtype == torch.float16
    use_bf16 = device == "cuda" and dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=50 if args.max_steps > 0 else 100,
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=device == "cuda" and os.name != "nt",
        optim=args.optimizer,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    last_ckpt = None
    try:
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = get_last_checkpoint(str(out))
    except Exception:
        last_ckpt = None
    if last_ckpt:
        print(f"resuming from {last_ckpt}", flush=True)
    offload_context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False)
        if args.activation_offload
        else contextlib.nullcontext()
    )
    if args.activation_offload:
        print("activation_offload=CPU", flush=True)
    with offload_context:
        trainer.train(resume_from_checkpoint=last_ckpt)
    final_fingerprint = parameter_fingerprint()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    loss_history = [
        {"step": row.get("step"), "loss": row["loss"]}
        for row in trainer.state.log_history
        if "loss" in row
    ]
    grad_norm_history = [
        {"step": row.get("step"), "grad_norm": row["grad_norm"]}
        for row in trainer.state.log_history
        if "grad_norm" in row
    ]
    effective_initial_fingerprint = (
        fingerprint_callback.train_begin_fingerprint or initial_fingerprint
    )
    metadata = {
                "base_model": args.model,
                "revision": args.revision,
                "device": device,
                "dtype": str(dtype),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "data": str(args.data),
                "target_modules": targets,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "load_in_4bit": args.load_in_4bit,
                "activation_offload": args.activation_offload,
                "optimizer": args.optimizer,
                "kernel_status": kernel_status,
                "peak_vram_mb": (
                    round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                    if device == "cuda"
                    else 0.0
                ),
                "seed": args.seed,
                "deterministic": args.deterministic,
                "pre_resume_trainable_fingerprint": initial_fingerprint,
                "initial_trainable_fingerprint": (
                    effective_initial_fingerprint
                ),
                "final_trainable_fingerprint": final_fingerprint,
                "trainable_parameters_changed": effective_initial_fingerprint != final_fingerprint,
                "loss_history": loss_history,
                "grad_norm_history": grad_norm_history,
                "global_step": trainer.state.global_step,
    }
    payload = json.dumps(metadata, ensure_ascii=False, indent=2)
    (out / "train_meta.json").write_text(payload, encoding="utf-8")
    (out / f"train_meta_step_{trainer.state.global_step}.json").write_text(
        payload, encoding="utf-8"
    )
    print(f"saved adapter to {out / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
