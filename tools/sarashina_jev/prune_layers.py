"""Create a smaller Sarashina-JEV artifact by selecting retained layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.sarashina_jev.model import SarashinaJevScorer, _layers
from tools.sarashina_jev.util import select_even_layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-layers", type=int, required=True)
    args = parser.parse_args()

    src = Path(args.artifact)
    out = Path(args.out)
    model, tokenizer, meta = SarashinaJevScorer.load_artifact(
        src, torch_dtype=torch.float32
    )
    layers = _layers(model.backbone)
    positions = select_even_layers(len(layers), args.keep_layers)
    source_indices = list(meta.get("kept_layer_indices", range(len(layers))))
    if len(source_indices) != len(layers):
        raise ValueError(
            f"metadata layer count {len(source_indices)} != model layer count {len(layers)}"
        )
    selected_original = [int(source_indices[i]) for i in positions]
    model.backbone.layers = torch.nn.ModuleList([layers[i] for i in positions])
    model.backbone.config.num_hidden_layers = len(positions)
    model.backbone.config.use_cache = False
    model.save_artifact(
        out,
        tokenizer,
        source_model=str(meta.get("source_model", "sarashina2.2-0.5b")),
        kept_layer_indices=selected_original,
        page_size=int(meta.get("page_size", 5)),
    )
    out_meta = json.loads((out / "jev_config.json").read_text(encoding="utf-8"))
    out_meta.update(
        {
            "layer_pruning": {
                "source_artifact": str(src),
                "source_kept_layer_indices": source_indices,
                "selected_positions": positions,
                "selected_original_layer_indices": selected_original,
                "selection": "even_positions_with_first_and_last",
            },
            "parameter_report": model.parameter_report(),
        }
    )
    (out / "jev_config.json").write_text(
        json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "source_artifact": str(src),
        "out": str(out),
        "source_layer_indices": source_indices,
        "selected_positions": positions,
        "selected_original_layer_indices": selected_original,
        "parameter_report": model.parameter_report(),
    }
    (out / "layer_prune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
