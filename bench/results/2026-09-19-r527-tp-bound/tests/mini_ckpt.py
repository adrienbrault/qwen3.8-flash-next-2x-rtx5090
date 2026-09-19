"""Synthetic qwen4_exp checkpoint for the CPU dry run: the served text_config (ref/model-config-scalars.txt) with fewer
layers / experts / vocab, and every tensor the real loaders request at its REAL per-tensor shape (EXL3 trellis
(in/16, out/16, 16*K) int16 + suh/svh, fp16 weights for the unquantized linears, norms, GDN, hyper-connections).
Tensor data is zero (the dry run checks calls, not numbers). Keys are derived from the model the real architecture
code builds, so a key the loader asks for and this file lacks fails the dry run loudly.
"""
from __future__ import annotations

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCALARS = os.path.join(HERE, "..", "..", "..", "ref", "model-config-scalars.txt")

MINI = {"num_hidden_layers": 8, "num_experts": 16, "vocab_size": 8192}
# routed expert K per layer (the 2.50 bpw pack mixes K = 2/3/4 per layer)
EXPERT_K = {0: 2, 1: 3, 2: 4, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3, "mtp": 4}


def write_config(d: str, overrides: dict | None = None) -> dict:
    with open(SCALARS) as f:
        scal = ast.literal_eval(f.read())
    scal.update(MINI)
    scal.update(overrides or {})
    os.makedirs(d, exist_ok=True)
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": scal, "tie_word_embeddings": False}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return scal


def _expert_k(key: str) -> int | None:
    if ".experts." not in key:
        return None
    if key.startswith("mtp."):
        return EXPERT_K["mtp"]
    li = int(key.split(".layers.")[1].split(".")[0])
    return EXPERT_K[li]


def tensor_specs(models) -> dict:
    """key -> (shape, dtype-name) for everything the real loaders read."""
    import torch  # noqa: F401
    from exllamav3.modules import Linear, RMSNorm, GatedDeltaNet, GatedResidual, Embedding
    from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
    from exllamav3.modules.arch_specific.qwen4_exp_mtp import Qwen4ExpMTPInputLayer
    specs = {}
    for model in models:
        for m in model:
            if isinstance(m, Linear):
                if m.qmap is None:
                    specs[f"{m.key}.weight"] = ((m.out_features, m.in_features), "F16")
                else:
                    K = _expert_k(m.key) or 4
                    assert m.in_features % 16 == 0 and m.out_features % 16 == 0, m.key
                    specs[f"{m.key}.trellis"] = ((m.in_features // 16, m.out_features // 16, 16 * K), "I16")
                    specs[f"{m.key}.suh"] = ((m.in_features,), "F16")
                    specs[f"{m.key}.svh"] = ((m.out_features,), "F16")
            elif isinstance(m, RMSNorm):
                dim = {"q_norm": 256, "k_norm": 256, "q_layernorm": 128, "k_layernorm": 128}.get(
                    m.key.split(".")[-1], 2560)
                specs[m.tensor_key] = ((dim,), "F16")
            elif isinstance(m, GatedRMSNorm):
                specs[f"{m.key}.weight"] = ((128,), "F16")
            elif isinstance(m, GatedDeltaNet):
                nv = m.num_v_heads
                specs[f"{m.key}.A_log"] = ((nv,), "F32")
                specs[f"{m.key}.dt_bias"] = ((nv,), "BF16")
                specs[f"{m.key}.conv1d.weight"] = ((m.fdim_qkv, 1, m.conv_kernel_size), "BF16")
            elif isinstance(m, GatedResidual):
                H, D, r = m.hc_mult, m.hidden_size, 320
                specs[f"{m.key}.hc_norm.weight"] = ((H * D,), "F16")
                specs[f"{m.key}.input_mix_weight_down.weight"] = ((r, H * D), "F16")
                specs[f"{m.key}.input_mix_weight_up.weight"] = ((H * D, r), "F16")
                if m.use_combine:
                    specs[f"{m.key}.block_inject_weight.weight"] = ((H, H * D), "F16")
            elif isinstance(m, Embedding):
                specs[f"{m.key}.weight"] = ((m.vocab_size, m.hidden_size), "F16")
            elif isinstance(m, Qwen4ExpMTPInputLayer):
                specs[f"{m.key}.pre_fc_norm_hidden.weight"] = ((m.hidden_size,), "F16")
    return specs


def write_tensors(d: str, specs: dict):
    """One safetensors file, zero data, written directly (header + zero bytes) to avoid materializing tensors."""
    esize = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4}
    header, off = {}, 0
    for k in sorted(specs):
        shape, dt = specs[k]
        n = 1
        for s in shape:
            n *= s
        header[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n * esize[dt]]}
        off += n * esize[dt]
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    path = os.path.join(d, "model.safetensors")
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        f.truncate(8 + len(hb) + off)      # sparse zero data
    return path, off


def build(d: str) -> dict:
    """Write config + tensors; returns {'specs', 'bytes'}. Requires dryrun_env.install() first."""
    write_config(d)
    from exllamav3 import Config, Model
    cfg = Config.from_directory(d)
    models = [Model.from_config(cfg), Model.from_config(cfg, component="mtp")]
    specs = tensor_specs(models)
    _, nbytes = write_tensors(d, specs)
    return {"specs": specs, "bytes": nbytes}
