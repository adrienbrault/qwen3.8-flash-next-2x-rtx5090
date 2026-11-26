"""rebase167 import smoke, run from /app inside the built image with no GPU (CUDA_VISIBLE_DEVICES=).

Checks that the rebuilt extension loads and exports the served entry points + upstream's new ones, that the
integrated package imports (every module the daily's Flash-Next path touches), the version string, and that
TabbyAPI's ExLlamaV3 backend imports against it. Exit status 1 on any failure.
"""
import importlib
import os
import sys
import traceback

failures = []


def check(label, fn):
    try:
        out = fn()
        print(f"PASS {label}" + (f": {out}" if out not in (None, True) else ""))
    except Exception:
        failures.append(label)
        print(f"FAIL {label}")
        traceback.print_exc()


import torch  # noqa: E402

print(f"torch {torch.__version__}, cuda {torch.version.cuda}, devices {torch.cuda.device_count()}")


def ext_names():
    import exllamav3_ext as e
    served = [
        # served perf stack (every flag family in the daily launcher)
        "gr_mix_v2", "gr_mix_v2_int8", "gr_mix_v2_int8_statein", "gr_mix_v2_int8_v3", "gr_mix_v2_fused",
        "dense_v2_modes", "dense_v2_set_mode", "dense_v2_k6", "dense_v2_set_k6", "dense_v2_launch_counts",
        "exl3_moe_coop_rows_cap", "exl3_moe_prefill_e3", "exl3_moe_prefill_e3_det", "exl3_moe_prefill_e3_det_reduce",
        "exl3_moe_prefill_e3_tile_rows", "exl3_moe_coop", "exl3_moe_coop_ev",
        "batched_conv_rewind", "batched_scan_replay", "gdn_ba_gemv",
        # upstream 1.6.7 additions the integrated Python calls
        "hc_apply", "gr_mix", "gr_mix_tiled",
    ]
    missing = [n for n in served if not hasattr(e, n)]
    assert not missing, f"missing extension entry points: {missing}"
    attrs = {a: getattr(e, a, None) for a in ("dense_v2_revision", "hc_mix_v2_revision", "hc_mix_v3_revision",
                                              "moe_coop_v3_revision", "moe_rows32_revision", "latchain_revision")}
    assert all(v is not None for v in attrs.values()), attrs
    return attrs


def ext_defaults():
    import exllamav3_ext as e
    # The image ENV (inherited from tune268) pins these off; the launcher's EXTRA_ENV turns them on at run time
    return {"dense_v2_k6": e.dense_v2_k6(), "dense_v2_modes": e.dense_v2_modes()}


def pkg_version():
    import exllamav3
    from exllamav3.version import __version__
    assert __version__ == "1.6.7", __version__
    from importlib.metadata import version
    assert version("exllamav3").split("+", 1)[0] == __version__
    return f"exllamav3 {__version__} (metadata {version('exllamav3')}) at {os.path.dirname(exllamav3.__file__)}"


MODULES = [
    "exllamav3", "exllamav3.ext", "exllamav3.util.cuda_flags",
    "exllamav3.modules.gated_delta_net", "exllamav3.modules.hyperconnections", "exllamav3.modules.block_sparse_mlp",
    "exllamav3.modules.moe_batch_recon", "exllamav3.modules.mlp", "exllamav3.modules.embedding",
    "exllamav3.modules.embedding_pruned", "exllamav3.modules.ngram_embedding", "exllamav3.modules.ple",
    "exllamav3.modules.row_table", "exllamav3.modules.sparse_head", "exllamav3.modules.qsa_indexer",
    "exllamav3.modules.attention_fn.bc_attn", "exllamav3.modules.attention_fn.qsa_triton",
    "exllamav3.modules.arch_specific.qwen4_exp_mtp",
    "exllamav3.generator", "exllamav3.generator.fast_accept", "exllamav3.generator.disk_cache", "exllamav3.cache",
]


def tabby_backend():
    sys.path.insert(0, "/app")
    importlib.import_module("common.config_models")
    importlib.import_module("backends.exllamav3.model")
    importlib.import_module("endpoints.OAI.router")
    importlib.import_module("endpoints.core.router")
    return "backends.exllamav3.model"


check("extension entry points", ext_names)
check("extension default modes", ext_defaults)
check("package version", pkg_version)
for m in MODULES:
    check(f"import {m}", lambda m = m: importlib.import_module(m) and None)
check("tabby backend import", tabby_backend)

print(f"== smoke: {len(failures)} failure(s) ==")
sys.exit(1 if failures else 0)
