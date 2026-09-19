#!/usr/bin/env python3
"""Checkpoint-backed GPU parity test and one-layer MoE microbenchmark.

The parent runs fresh OFF and ON child processes because the native opt-in is
latched while BC_BlockSparseMLP objects are constructed. Each child loads the
served checkpoint across both GPUs, selects one eligible MoE layer per card,
and evaluates identical deterministic inputs for every row count 1..16.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


PATCHED_SOURCE_SHA256 = "6aad01d70d20e82971f986adecbac1f6b53fe132c4daf276afe4da77b5d1327d"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-split", default="30,30")
    parser.add_argument("--expect-devices", type=int, default=2)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("off", "on"), help=argparse.SUPPRESS)
    parser.add_argument("--tensor-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.expect_devices < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("device count, warmup and repeats must be positive")
    if args.worker and (args.mode is None or args.tensor_out is None or args.report_out is None):
        parser.error("worker mode needs --mode, --tensor-out and --report-out")
    return args


def deterministic_input(torch, device, rows, hidden_size):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5EED0000 + 1000 * device.index + rows)
    return torch.randn((1, rows, hidden_size), generator=generator).half().to(device)


def time_module(torch, module, x, warmup, repeats):
    for _ in range(warmup):
        module.forward(x, {})
    torch.cuda.synchronize(x.device)
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        module.forward(x, {})
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def worker(args):
    expected_flag = "1" if args.mode == "on" else "0"
    if os.environ.get("EXL3_SHARED_EXPERT_OVERLAP") != expected_flag:
        raise RuntimeError("worker environment does not match --mode")

    import hashlib
    import torch
    import exllamav3
    from exllamav3 import Config, Model
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != args.expect_devices:
        raise RuntimeError(
            f"expected {args.expect_devices} visible CUDA devices, got {torch.cuda.device_count()}"
        )
    package_root = Path(exllamav3.__file__).resolve().parent
    patched_source = package_root / "exllamav3_ext/libtorch/blocksparse_mlp.cpp"
    source_hash = hashlib.sha256(patched_source.read_bytes()).hexdigest()
    if source_hash != PATCHED_SOURCE_SHA256:
        raise RuntimeError(f"unexpected installed patched source hash: {source_hash}")

    split = [float(value) for value in args.gpu_split.split(",")]
    if len(split) != args.expect_devices:
        raise ValueError("--gpu-split must have one value per expected device")
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(
        use_per_device=split,
        progressbar=True,
        verbose=True,
        max_batch_size=16,
        max_chunk_size=16,
        autosplit_no_forward=True,
    )

    chosen = {}
    for module in model:
        if not isinstance(module, BlockSparseMLP) or module.device is None:
            continue
        device = torch.device(module.device)
        if device.type != "cuda" or device.index in chosen:
            continue
        if module.bc is not None and module.bc_sh_exp and module.shared_experts is not None:
            chosen[device.index] = module
    missing = sorted(set(range(args.expect_devices)) - set(chosen))
    if missing:
        raise RuntimeError(f"no eligible shared-expert MoE layer found on cuda devices {missing}")

    outputs = {}
    report = {
        "mode": args.mode,
        "flag": expected_flag,
        "source_sha256": source_hash,
        "layers": {},
        "benchmarks": [],
    }
    for device_index, module in sorted(chosen.items()):
        device = torch.device("cuda", device_index)
        report["layers"][str(device_index)] = module.key
        with torch.cuda.device(device), torch.inference_mode():
            for rows in range(1, 17):
                x = deterministic_input(torch, device, rows, module.hidden_size)
                # Eager prime, graph capture and replay are all outside the retained output.
                module.forward(x, {})
                module.forward(x, {})
                y = module.forward(x, {}).detach().cpu().clone()
                torch.cuda.synchronize(device)
                outputs[f"cuda:{device_index}/rows:{rows}"] = y
            if args.bench:
                for rows in (1, 4, 16):
                    x = deterministic_input(torch, device, rows, module.hidden_size)
                    median_us = time_module(torch, module, x, args.warmup, args.repeats)
                    report["benchmarks"].append(
                        {
                            "device": device_index,
                            "layer": module.key,
                            "rows": rows,
                            "median_us": median_us,
                        }
                    )

    torch.save(outputs, args.tensor_out)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def run_child(args, mode, root):
    tensor_out = root / f"{mode}.pt"
    report_out = root / f"{mode}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode", mode,
        "--model", args.model,
        "--gpu-split", args.gpu_split,
        "--expect-devices", str(args.expect_devices),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--tensor-out", str(tensor_out),
        "--report-out", str(report_out),
    ]
    if args.bench:
        command.append("--bench")
    env = os.environ.copy()
    env["EXL3_SHARED_EXPERT_OVERLAP"] = "1" if mode == "on" else "0"
    env["EXL3_MOE_COOP_V2"] = "1"
    subprocess.run(command, env=env, check=True)
    return tensor_out, json.loads(report_out.read_text())


def parent(args):
    import torch

    with tempfile.TemporaryDirectory(prefix="shared-expert-overlap-") as temp:
        root = Path(temp)
        off_path, off_report = run_child(args, "off", root)
        on_path, on_report = run_child(args, "on", root)
        if off_report["layers"] != on_report["layers"]:
            raise AssertionError(
                f"OFF/ON selected different layers: {off_report['layers']} vs {on_report['layers']}"
            )
        off = torch.load(off_path, map_location="cpu", weights_only=True)
        on = torch.load(on_path, map_location="cpu", weights_only=True)
        if off.keys() != on.keys():
            raise AssertionError("OFF/ON output key sets differ")
        for key in off:
            if not torch.equal(off[key], on[key]):
                unequal = torch.count_nonzero(off[key] != on[key]).item()
                max_abs = (off[key].float() - on[key].float()).abs().max().item()
                raise AssertionError(f"{key}: {unequal} unequal values, max_abs={max_abs}")

        result = {
            "status": "pass",
            "torch_equal": True,
            "devices": args.expect_devices,
            "rows": list(range(1, 17)),
            "off_layers": off_report["layers"],
            "on_layers": on_report["layers"],
        }
        print(json.dumps(result, sort_keys=True))
        if args.bench:
            off_by_key = {
                (row["device"], row["rows"]): row for row in off_report["benchmarks"]
            }
            comparisons = []
            for row in on_report["benchmarks"]:
                baseline = off_by_key[(row["device"], row["rows"])]
                comparisons.append(
                    {
                        "device": row["device"],
                        "rows": row["rows"],
                        "layer_off": baseline["layer"],
                        "layer_on": row["layer"],
                        "off_us": baseline["median_us"],
                        "on_us": row["median_us"],
                        "speedup_pct": 100.0 * (baseline["median_us"] / row["median_us"] - 1.0),
                    }
                )
            print(json.dumps({"benchmark": comparisons}, indent=2, sort_keys=True))


def main():
    args = arguments()
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
