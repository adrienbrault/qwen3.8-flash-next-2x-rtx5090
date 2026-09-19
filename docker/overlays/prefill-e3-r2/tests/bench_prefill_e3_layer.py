#!/usr/bin/env python3
"""Real-weight per-layer OFF/E3 benchmark, one layer per K per CUDA card.

For every visible card, K=2 and K=3 are required. K=4 is benchmarked when a
layer containing that projection width is placed on the card. Every case uses
the same real checkpoint layer, routing and input for OFF and ON, at 512 and
2048 rows, with CUDA-event samples and finite/error statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("EXL3_MOE_PREFILL_E3", "1")

import torch
from exllamav3 import Config, Model
from exllamav3.modules import BlockSparseMLP
import exllamav3.modules.block_sparse_mlp as bsm


def timed(fn, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid CUDA-event timing: {elapsed}")
        values.append(elapsed)
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def errors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    d = (reference - candidate).float()
    ref = reference.float()
    rmse = d.square().mean().sqrt()
    return {
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "rmse": rmse.item(),
        "nrmse": rmse.div(ref.square().mean().sqrt().clamp_min(1e-12)).item(),
        "equal": torch.equal(reference, candidate),
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "candidate_finite": bool(torch.isfinite(candidate).all().item()),
    }


def k_signature(layer: BlockSparseMLP) -> tuple[int, int, int]:
    return (layer.multi_gate.K, layer.multi_up.K, layer.multi_down.K)


def eligible(layer: BlockSparseMLP) -> bool:
    return (
        layer.device is not None and layer.device.type == "cuda"
        and layer.num_experts == 512 and layer.num_local_experts == 512
        and layer.num_experts_per_tok == 10 and layer.expert_size == 2560
        and layer.intermediate_size_padded == 640 and layer.support_fused
        and layer.multi_gate is not None
        and all(k in (2, 3, 4) for k in k_signature(layer))
        and layer.multi_gate.mul1 and layer.multi_up.mul1 and layer.multi_down.mul1
    )


@torch.inference_mode()
def bench_layer(layer, rows: int, warmup: int, iterations: int, seed: int) -> dict:
    device = layer.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn((rows, 2560), dtype=torch.half, device=device, generator=gen)

    selected, _ = layer.routing_fn(rows, layer.routing_cfg, x, {})
    counts = torch.bincount(selected.reshape(-1), minlength=512)
    routing = {
        "assignments": int(selected.numel()),
        "mean_rows": counts.float().mean().item(),
        "max_rows": int(counts.max().item()),
        "experts_gt_32": int((counts > 32).sum().item()),
        "experts_gt_256": int((counts > 256).sum().item()),
    }

    saved_e3 = bsm.MOE_PREFILL_E3
    saved_shared = layer.shared_experts
    saved_gate = layer.shared_gate
    layer.shared_experts = None
    layer.shared_gate = None
    try:
        def run(active: bool):
            bsm.MOE_PREFILL_E3 = active
            return layer.forward(x, {})

        torch.cuda.synchronize(device)
        off = run(False)
        torch.cuda.synchronize(device)
        on = run(True)
        torch.cuda.synchronize(device)
        on2 = run(True)
        parity = {
            "on_vs_off": errors(off, on),
            "on_repeat": errors(on, on2),
        }
        if not parity["on_vs_off"]["reference_finite"] or not parity["on_vs_off"]["candidate_finite"]:
            raise RuntimeError("non-finite OFF/ON output")
        if not parity["on_repeat"]["candidate_finite"]:
            raise RuntimeError("non-finite repeated ON output")
        off_t = timed(lambda: run(False), warmup, iterations)
        on_t = timed(lambda: run(True), warmup, iterations)
    finally:
        bsm.MOE_PREFILL_E3 = saved_e3
        layer.shared_experts = saved_shared
        layer.shared_gate = saved_gate
    return {
        "rows": rows,
        "routing": routing,
        "parity": parity,
        "off": off_t,
        "on": on_t,
        "speedup": off_t["median_ms"] / on_t["median_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--use-per-device", default="30,30")
    ap.add_argument("--rows", default="512,2048")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    row_counts = [int(value) for value in args.rows.split(",")]
    if row_counts != [512, 2048]:
        raise SystemExit("round-2 gate requires --rows 512,2048")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    use = [float(value) for value in args.use_per_device.split(",")]
    if use != [30.0, 30.0]:
        raise SystemExit("round-2 gate requires --use-per-device 30,30")
    t0 = time.time()
    model.load(
        use_per_device=use,
        max_chunk_size=max(row_counts),
        progressbar=True,
    )
    layers = [module for module in model if isinstance(module, BlockSparseMLP) and eligible(module)]
    per_device: dict[int, list[BlockSparseMLP]] = {}
    for layer in layers:
        per_device.setdefault(layer.device.index, []).append(layer)
    expected_devices = set(range(torch.cuda.device_count()))
    if set(per_device) != expected_devices:
        raise SystemExit(
            f"eligible per-device layers missing: have={sorted(per_device)}, "
            f"expected={sorted(expected_devices)}"
        )

    record = {
        "kind": "real-weight routed-MoE per-layer per-K microbenchmark",
        "model": str(Path(args.model).resolve()),
        "load_seconds": time.time() - t0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "thin_rows": bsm.MOE_PREFILL_E3_THIN_ROWS,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "layout": {
            "signature_counts": {
                "/".join(map(str, signature)): count
                for signature, count in sorted(Counter(k_signature(layer) for layer in layers).items())
            },
            "mixed_projection_layers": [
                layer.key for layer in layers if len(set(k_signature(layer))) != 1
            ],
        },
        "devices": [],
    }
    for device_idx, device_layers in sorted(per_device.items()):
        available_k = sorted({k for layer in device_layers for k in k_signature(layer)})
        missing = {2, 3} - set(available_k)
        if missing:
            raise SystemExit(f"device {device_idx} lacks required K={sorted(missing)} layer")
        target_k = [2, 3] + ([4] if 4 in available_k else [])
        dev = {
            "device": device_idx,
            "name": torch.cuda.get_device_name(device_idx),
            "capability": list(torch.cuda.get_device_capability(device_idx)),
            "available_k": available_k,
            "layers": [],
        }
        if dev["capability"] != [12, 0]:
            raise SystemExit(f"device {device_idx} is not sm_120: {dev['capability']}")
        record["devices"].append(dev)
        with torch.cuda.device(device_idx):
            for bits in target_k:
                layer = next(layer for layer in device_layers if bits in k_signature(layer))
                layer_record = {
                    "target_k": bits,
                    "layer": layer.key,
                    "projection_k": list(k_signature(layer)),
                    "cases": [],
                }
                for rows in row_counts:
                    result = bench_layer(
                        layer,
                        rows,
                        args.warmup,
                        args.iterations,
                        seed=1701 + rows + device_idx * 17 + bits,
                    )
                    layer_record["cases"].append(result)
                    print(json.dumps({
                        "device": device_idx,
                        "target_k": bits,
                        "layer": layer.key,
                        "projection_k": list(k_signature(layer)),
                        **result,
                    }), flush=True)
                    output.write_text(json.dumps(record, indent=2) + "\n")
                dev["layers"].append(layer_record)
                output.write_text(json.dumps(record, indent=2) + "\n")
        output.write_text(json.dumps(record, indent=2) + "\n")
    model.unload()


if __name__ == "__main__":
    main()
