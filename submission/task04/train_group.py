"""Train one task 04 group from the shared YOLO11n checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "yolo11n.pt"
DATA = ROOT / "data.yaml"
RUNS = ROOT / "runs" / "task04"


def audit_parameters(model, freeze):
    prefixes = [f"model.{index}." for index in range(freeze)] + [".dfl"]
    total = frozen = 0
    for name, parameter in model.model.named_parameters():
        total += parameter.numel()
        if any(prefix in name for prefix in prefixes):
            frozen += parameter.numel()
    return {
        "freeze": freeze,
        "frozen_layer_prefixes": [f"model.{index}." for index in range(freeze)],
        "always_frozen": [".dfl"],
        "total_parameters": total,
        "frozen_parameters": frozen,
        "trainable_parameters": total - frozen,
        "frozen_fraction": round(frozen / total, 6),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=int, required=True, choices=(0, 10))
    parser.add_argument("--name", required=True, choices=("full", "freeze10"))
    args = parser.parse_args()

    torch.set_num_threads(3)
    output = RUNS / args.name
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    config = {
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "data": str(DATA.relative_to(ROOT)),
        "freeze": args.freeze,
        "epochs": 5,
        "imgsz": 416,
        "batch": 8,
        "device": "cpu",
        "workers": 0,
        "seed": 42,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "mosaic": 0.0,
        "mixup": 0.0,
        "scale": 0.20,
        "plots": False,
        "project": str(RUNS.relative_to(ROOT)),
        "name": args.name,
        "exist_ok": True,
    }
    (output / "requested-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model.train(
        data=str(DATA), device="cpu", imgsz=416, epochs=5, batch=8,
        workers=0, seed=42, deterministic=True, optimizer="AdamW", lr0=.001,
        mosaic=0.0, mixup=0.0, scale=.20, plots=False, project=str(RUNS),
        name=args.name, exist_ok=True, verbose=True, freeze=args.freeze,
    )
    trained = YOLO(str(output / "weights" / "best.pt"))
    audit = audit_parameters(trained, args.freeze)
    audit["source"] = str((output / "weights" / "best.pt").relative_to(ROOT))
    (output / "freeze-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
