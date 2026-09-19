#!/usr/bin/env python3
"""Read safetensors headers and report expert projection K by MoE layer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_TENSORS = {2: 38_400, 3: 35_328, 4: 1_536}


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        return json.loads(stream.read(length))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-served-pack", action="store_true")
    args = ap.parse_args()
    model = Path(args.model)
    files = sorted(model.rglob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors below {model}")

    tensor_counts: Counter[int] = Counter()
    layout: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        for name, metadata in read_header(path).items():
            if name == "__metadata__" or ".experts." not in name or not name.endswith(".trellis"):
                continue
            projection = next((p for p in PROJECTIONS if f".{p}.trellis" in name), None)
            if projection is None:
                continue
            shape = metadata.get("shape", [])
            if not shape or shape[-1] % 16:
                raise SystemExit(f"invalid trellis shape for {name}: {shape}")
            bits = shape[-1] // 16
            expert_text = name.split(".experts.", 1)[1].split(".", 1)[0]
            layer = name.split(".experts.", 1)[0]
            layout[layer][projection].append((int(expert_text), bits))
            tensor_counts[bits] += 1

    layers = []
    irregular = []
    mixed = []
    for layer, projections in sorted(layout.items()):
        signature = {}
        layer_problems = []
        for projection in PROJECTIONS:
            entries = projections.get(projection, [])
            ks = sorted({bits for _, bits in entries})
            signature[projection] = ks[0] if len(ks) == 1 else ks
            if len(entries) != 512:
                layer_problems.append(f"{projection}: {len(entries)} experts")
            expert_ids = {expert for expert, _ in entries}
            if expert_ids != set(range(512)):
                layer_problems.append(
                    f"{projection}: expert IDs differ from 0..511"
                )
            if len(ks) != 1:
                layer_problems.append(f"{projection}: K values {ks}")
        record = {"layer": layer, "projection_k": signature}
        layers.append(record)
        values = [signature[p] for p in PROJECTIONS]
        if all(isinstance(value, int) for value in values) and len(set(values)) != 1:
            mixed.append(record)
        if layer_problems:
            irregular.append({**record, "problems": layer_problems})

    result = {
        "schema_version": 1,
        "model": str(model.resolve()),
        "safetensors_files": len(files),
        "expert_trellis_tensor_counts": {
            str(bits): count for bits, count in sorted(tensor_counts.items())
        },
        "moe_layers": len(layers),
        "mixed_projection_layers": mixed,
        "irregular_layers": irregular,
        "layers": layers,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if args.strict_served_pack:
        if dict(tensor_counts) != EXPECTED_TENSORS:
            raise SystemExit(
                f"served-pack tensor counts differ: {dict(tensor_counts)} != {EXPECTED_TENSORS}"
            )
        if irregular:
            raise SystemExit(f"irregular expert layout in {len(irregular)} layer(s)")
        if any(k not in EXPECTED_TENSORS for k in tensor_counts):
            raise SystemExit(f"unsupported K values: {sorted(tensor_counts)}")


if __name__ == "__main__":
    main()
