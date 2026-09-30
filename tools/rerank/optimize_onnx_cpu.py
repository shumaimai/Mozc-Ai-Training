"""Generate CPU experiments without changing the frozen source graph."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    root=Path(args.out_dir)
    root.mkdir(parents=True,exist_ok=True)
    source=onnx.load(args.model)
    print("source_ops",dict(__import__('collections').Counter(n.op_type for n in source.graph.node)),flush=True)
    # ModernBERT shares standard LayerNorm/GELU subgraphs with BERT. Disable
    # BERT attention fusion: its positional semantics are not ModernBERT RoPE.
    from onnxruntime.transformers.fusion_options import FusionOptions
    options=FusionOptions("bert")
    options.enable_attention=False
    options.enable_embed_layer_norm=False
    options.enable_skip_layer_norm=True
    optimized=optimize_model(args.model,model_type="bert",num_heads=4,hidden_size=256,
        optimization_options=options,opt_level=1,use_gpu=False)
    path=root/"cross_encoder_optimized_fp32.onnx"
    optimized.save_model_to_file(str(path))
    print("optimized_ops",dict(__import__('collections').Counter(n.op_type for n in optimized.model.graph.node)),flush=True)
    for name,model in (("raw",args.model),("optimized",str(path))):
        quantize_dynamic(model,str(root/f"cross_encoder_{name}_int8.onnx"),
            weight_type=QuantType.QInt8,per_channel=True,
            op_types_to_quantize=["MatMul"],extra_options={"MatMulConstBOnly":True,
                "DefaultTensorType":onnx.TensorProto.FLOAT})
    print(json.dumps({p.name:p.stat().st_size for p in root.glob('*.onnx')},indent=2),flush=True)


if __name__=="__main__":
    main()
