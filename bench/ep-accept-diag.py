#!/usr/bin/env python3
"""Isolated QSA/PLE GPU qualification. `selftest` imports only the standard library.

Exit codes: 0 all required checks PASS; 1 at least one FAIL; 2 incomplete/NOT-RUN.
No Model.load(), TP capability override, EP routing, network, or native build.
See ep-accept-status.md for the protocol, artifact costs and scope of the verdict.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest

VERSION = 1
DEFAULT_MODEL = "/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw"
STATUSES = {"PASS", "FAIL", "NOT-RUN"}
TOL = {"rtol": 1e-3, "atol": 1e-3}
EXACT = {"rtol": 0.0, "atol": 0.0}
HOST_MONITOR = None
CHECKS = {
    "completion": ("Every requested module, cache mode, reconstruction setting and transport plan finished", "An interrupted or partial matrix cannot close an EP prerequisite and must be resumed in a fresh run."),
    "preconditions": ("All required files, modules, methods, bindings and two CUDA devices exist", "Missing prerequisites block qualification; they are not evidence of numerical correctness."),
    "provenance": ("Hash the actual checkpoint, configuration, source, extension and inputs", "Unidentified weights or runtime settings require a reproducible rerun before scheduling EP."),
    "transport": ("Real SHM imports preserve full weights, geometry, settings, cache IDs and device ownership", "A transport failure requires fixing module replication before EP integration can begin."),
    "repeatability": ("Repeat the untouched CUDA:0 reference on identical saved input bytes", "Reference instability prevents attributing numerical differences to transport and needs investigation first."),
    "numerics": ("Every replica output and intermediate matches the untouched reference", "Numerical divergence refutes this prerequisite and adds kernel or transport debugging to the EP schedule."),
    "memory": ("Record measured peaks and compare them with replicated planner estimates", "An underestimate or OOM requires memory budgeting or placement work before EP can fit."),
    "qsa.indices": ("Selected indices match exactly and every incomplete causal tail is included", "Selection divergence changes attended tokens and blocks EP attention qualification."),
    "qsa.cache": ("Unequal histories, both cache IDs, chunk/decode boundaries and side planes agree", "Cache divergence requires fixing state ownership or updates before concurrent EP requests are safe."),
    "qsa.reduction": ("QSA calls no reduction, ordinary sharded attention does, combined reduction is rejected", "A duplicate or missing reduction changes hidden states and blocks the EP output policy."),
    "qsa.graph": ("Production BC warmup, capture and replay agree with eager on both cache IDs", "A graph-only failure requires a graph fix or a measured eager serving policy before EP."),
    "qsa.copy": ("Copy a partial page then continue without changing the source page", "Page-copy divergence blocks prefix sharing and speculative continuation under EP."),
    "qsa.companding": ("Preserve a=0.25 and compare sparse output with explicitly decompanded KV", "Failure exposes a companded-reader defect; zero-companding serving remains a separately reported gate."),
    "ple.state": ("Negative IDs, two caches, CPU token windows, GPU convolution and logical slots survive", "Recurrent ownership or history errors block multi-request EP correctness."),
    "ple.checkpoints": ("Real recurrent dispatch clears, rewinds, stashes, restores and deletes independently", "Checkpoint corruption blocks speculative decoding and prefix reuse under EP."),
    "ple.tables": ("Raw fp16/bf16 and K=1..8 trellis, disk/RAM, bias and shard layouts agree", "A table-format failure requires fixing embedding transport before accepting that storage policy."),
    "ple.lifetime": ("Reads survive parent unload, collection close, SHM reuse and consumer close", "Lifetime failure requires redesign or repair of ownership before long-running EP workers are viable."),
    "ple.prefetch": ("Workers start fresh and exercise hits, misses, retirement and pending-work unload", "Prefetch races or hangs require synchronization work before EP workers are reliable."),
    "ple.graph": ("Keep CPU hashing/history eager around CUDA graph warmup and changing-input replay", "Frozen CPU decisions require an enforced eager boundary before graph-backed EP integration."),
    "ple.resources": ("Repeated import/unload releases table descriptors and persistent references", "Resource growth threatens long-running workers and adds lifecycle debugging to EP."),
}


class NotRun(RuntimeError):
    """Missing capability/input/precondition, never a success or silent skip."""


def require(ok, reason):
    if not ok:
        raise NotRun(reason)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def families(kind):
    return [k for k in CHECKS if "." not in k or k.startswith(kind + ".")]


def verdict(rows):
    if not rows or any(r["status"] == "NOT-RUN" for r in rows):
        result = "NOT-RUN"
    else:
        result = "PASS"
    return "FAIL" if any(r["status"] == "FAIL" for r in rows) else result


def validate_report(doc):
    if doc.get("schema_version") != VERSION or doc.get("kind") not in ("qsa", "ple"):
        raise ValueError("invalid report identity")
    if not isinstance(doc.get("environment"), dict) or not isinstance(doc.get("checks"), list):
        raise ValueError("environment/checks missing")
    seen = set()
    for row in doc["checks"]:
        for field in ("id", "family", "status", "assertion", "consequence", "reason", "tolerance", "metrics"):
            if field not in row:
                raise ValueError(f"missing check field: {field}")
        if row["id"] in seen or row["family"] not in families(doc["kind"]):
            raise ValueError("duplicate ID or invalid family")
        seen.add(row["id"])
        if row["status"] not in STATUSES or not row["reason"] or not row["consequence"]:
            raise ValueError("invalid verdict or absent explanation")
        if not isinstance(row["metrics"], dict):
            raise ValueError("metrics must be an object")
        for key in ("rtol", "atol"):
            v = row["tolerance"].get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("invalid tolerance")
    if set(r["family"] for r in doc["checks"]) != set(families(doc["kind"])):
        raise ValueError("missing required check family")
    if doc.get("verdict") != verdict(doc["checks"]):
        raise ValueError("incorrect aggregate verdict")
    json.dumps(doc, allow_nan=False)


class Report:
    def __init__(self, kind, directory):
        self.kind, self.directory = kind, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.ids = set()
        self.journal = self.directory / "checks.jsonl"
        self.journal.write_text("")
        self.last_flush = time.monotonic()
        self.environment = {"python": sys.version, "platform": platform.platform(),
                            "seed": 1234, "inference_mode": True, "ep_enabled": False,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.flush()

    def add(self, family, name, status, reason, tolerance=EXACT, **metrics):
        assert status in STATUSES and family in families(self.kind)
        row = dict(id=f"{family}/{name}", family=family, status=status, reason=str(reason),
                   assertion=CHECKS[family][0], consequence=CHECKS[family][1],
                   tolerance=dict(tolerance), metrics=metrics)
        assert row["id"] not in self.ids, row["id"]
        self.ids.add(row["id"])
        self.rows.append(row)
        # Append every outcome durably to a closed file. Rewriting a many-thousand
        # row JSON document after every tensor checkpoint would make the all-layer
        # protocol quadratic in report I/O. Atomic snapshots are periodic instead.
        with self.journal.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"{status}: {row['id']}: {reason}", flush=True)
        if status != "PASS" or len(self.rows) % 128 == 0 or time.monotonic() - self.last_flush >= 15:
            self.flush()
        return row

    def check(self, family, name, fn, tolerance=EXACT):
        try:
            metrics = fn() or {}
            return self.add(family, name, "PASS", "assertions satisfied", tolerance, **metrics)
        except NotRun as e:
            return self.add(family, name, "NOT-RUN", str(e), tolerance)
        except Exception as e:
            status = "NOT-RUN" if family in ("preconditions", "provenance") else "FAIL"
            return self.add(family, name, status, f"{type(e).__name__}: {e}", tolerance,
                            traceback=traceback.format_exc())

    def document(self):
        rows = list(self.rows)
        for family in families(self.kind):
            if not any(r["family"] == family for r in rows):
                rows.append(dict(id=f"{family}/pending", family=family, status="NOT-RUN",
                                 reason="Required phase has not completed; see prior failures/preconditions.",
                                 assertion=CHECKS[family][0], consequence=CHECKS[family][1],
                                 tolerance=dict(EXACT), metrics={}))
        return dict(schema_version=VERSION, kind=self.kind, environment=self.environment,
                    checks=rows, verdict=verdict(rows))

    def flush(self):
        doc = self.document()
        validate_report(doc)
        atomic_json(self.directory / "report.json", doc)
        lines = [f"{self.kind.upper()}: {doc['verdict']}", ""]
        for r in doc["checks"]:
            lines += [f"{r['status']} {r['id']} (rtol={r['tolerance']['rtol']}, atol={r['tolerance']['atol']}): {r['reason']}",
                      f"  Consequence if failing: {r['consequence']}"]
        (self.directory / "summary.txt").write_text("\n".join(lines) + "\n")
        self.last_flush = time.monotonic()

    def exit_code(self):
        return {"PASS": 0, "FAIL": 1, "NOT-RUN": 2}[self.document()["verdict"]]


def checkpoint_files(root):
    root = Path(root)
    require(root.is_dir(), f"checkpoint directory missing: {root}")
    require((root / "config.json").is_file(), f"checkpoint config missing: {root / 'config.json'}")
    json.loads((root / "config.json").read_text())
    files = sorted(root.glob("*.safetensors"))
    require(bool(files), f"checkpoint has no .safetensors files: {root}")
    for index in root.glob("*.safetensors.index.json"):
        data = json.loads(index.read_text())
        require(isinstance(data.get("weight_map"), dict), f"invalid weight_map: {index}")
        for filename in set(data["weight_map"].values()):
            p = root / filename
            require(p.resolve().is_relative_to(root.resolve()) and p.is_file(), f"indexed checkpoint shard missing: {p}")
    return sorted(set(files + list(root.glob("*.json"))))


def source_preconditions(package, kind):
    package = Path(package)
    expected = {"modules/qsa_indexer.py": ("QSAIndexer", ["tp_export", "tp_import"]),
                "modules/attn.py": ("Attention", ["tp_export", "tp_import", "bc_attn_step"]),
                "model/model_tp_shared.py": ("SMProducer", ["send", "export", "clear", "close"])}
    if kind == "ple":
        expected.update({"modules/ple.py": ("PLELayer", ["tp_export", "tp_import"]),
                         "modules/ngram_embedding.py": ("NGramEmbedding", ["tp_export", "tp_import"])})
    errors = []
    for rel, (cls, methods) in expected.items():
        p = package / rel
        if not p.is_file():
            errors.append(f"missing module: {p}")
            continue
        tree = ast.parse(p.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        names = {n.name for c in nodes for n in c.body if isinstance(n, ast.FunctionDef)}
        errors += [f"missing symbol: {rel}:{cls}.{method}" for method in methods if method not in names]
    for rel in ("cache/qsa.py", "model/model_tp_alloc.py", "cache/recurrent.py", "architecture/qwen4_exp.py"):
        if not (package / rel).is_file():
            errors.append(f"missing module: {package / rel}")
    require(not errors, "\n".join(errors))


def apply_exact_patch(package, patch):
    """Unified patch with exact complete context, permitting offsets but never fuzz.

    Idempotency is per complete file: reject partial application or ambiguous matches.
    Only package-relative existing Python files may be changed.
    """
    sections = re.split(r"(?m)^diff --git ", Path(patch).read_text())[1:]
    require(bool(sections), f"empty/non-unified patch: {patch}")
    staged = {}
    for section in sections:
        lines = section.splitlines(keepends=True)
        match = re.search(r"(?m)^\+\+\+ b/(.+)$", section)
        require(match is not None, "patch has no destination")
        rel = Path(match[1])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.suffix == ".py", f"unsafe patch path: {rel}")
        path = Path(package) / rel
        require(path.is_file(), f"missing module required by patch: {path}")
        original = path.read_text()
        hunks, old, new = [], [], []
        active = False
        for line in lines:
            if line.startswith("@@ "):
                if active:
                    hunks.append(("".join(old), "".join(new)))
                old, new, active = [], [], True
            elif active:
                if line.startswith((" ", "-")):
                    old.append(line[1:])
                if line.startswith((" ", "+")):
                    new.append(line[1:])
        if active:
            hunks.append(("".join(old), "".join(new)))
        require(bool(hunks), f"no hunks: {rel}")
        if all(original.count(new) == 1 for _, new in hunks):
            continue
        result = original
        for old, new in hunks:
            require(bool(old) and result.count(old) == 1, f"patch context absent/ambiguous: {rel}; incompatible or partly patched image")
            result = result.replace(old, new, 1)
        ast.parse(result)
        staged[path] = result
    for path, result in staged.items():
        path.write_text(result)


def stage_package(args):
    spec = importlib.util.find_spec("exllamav3")
    require(spec is not None and spec.origin, "missing Python module: exllamav3")
    source = Path(spec.origin).parent
    target = Path(args.destination) / "exllamav3"
    require(not target.exists(), f"overlay already exists: {target}")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    gates = ["architecture/qwen4_exp.py", "model/model.py"]
    before = {p: sha256(target / p) for p in gates}
    patch_kinds = ("qsa",) if args.kind == "qsa" else ("qsa", "ple")
    for name in patch_kinds:
        apply_exact_patch(target, Path(args.patches) / f"ep-{name}-applied.patch")
    assert before == {p: sha256(target / p) for p in gates}, "loader/capability gate changed"
    for kind in patch_kinds:
        source_preconditions(target, kind)
    atomic_json(Path(args.destination) / "stage.json", {"source": str(source), "gate_hashes": before,
                "patches": {f"ep-{k}-applied.patch": sha256(Path(args.patches) / f"ep-{k}-applied.patch") for k in patch_kinds}})


def runtime_preconditions(args, report):
    checkpoint_files(args.model)
    require(Path(args.serving_config).is_file(), f"serving config missing: {args.serving_config}")
    missing = [n for n in ("torch", "numpy", "triton", "safetensors", "yaml", "exllamav3", "exllamav3_ext")
               if importlib.util.find_spec(n) is None]
    require(not missing, "missing Python module(s): " + ", ".join(missing))
    spec = importlib.util.find_spec("exllamav3_ext")
    require(any(spec.origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES),
            "exllamav3_ext is not a precompiled extension; JIT/native builds are forbidden")
    package = Path(importlib.util.find_spec("exllamav3").origin).parent
    source_preconditions(package, args.command)
    import torch
    report.environment.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                              extension=spec.origin, native_compilation_performed=False)
    import exllamav3_ext as ext
    symbols = ["quant_cache_paged", "dequant_cache_paged", "dsa_topk", "dsa_topk_tile", "dsa_topk_merge_tiles", "BC_Attention"]
    if args.command == "ple":
        symbols += ["ple_forward_streams", "ple_gate", "ngram_hash_cpu", "ngram_gather_cpu", "ngram_dequant"]
    absent = [s for s in symbols if not hasattr(ext, s)]
    require(not absent, "missing extension symbol(s): " + ", ".join(absent))
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA GPUs required (cuda:0 and cuda:1)")
    # Import the real package/dispatch dependency tree NOW, before serving is stopped.
    # Finding a spec alone would miss absent transitive Python dependencies.
    from exllamav3 import Config, Model
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.ple import PLELayer
    from exllamav3.model.model_tp_shared import SMConsumer
    from exllamav3.model.model_tp_fn import mp_model_append
    from exllamav3.cache.recurrent import mp_cache_recurrent_stash, mp_cache_recurrent_del
    for cls, names in ((SMConsumer, ("recv", "close")), (Attention, ("forward", "project_qkv", "project_o")),
                       (PLELayer, ("prefetch", "forward_streams_reference"))):
        require(all(callable(getattr(cls, name, None)) for name in names), f"missing method in {cls.__name__}: {names}")
    config = Config.from_directory(args.model)
    try:
        model = Model.from_config(config)
        require(model.caps.get("supports_tp") is False, "qwen4_exp loader gate must remain closed")
        candidates = ([m.key for m in model if isinstance(m, Attention) and m.qsa_indexer is not None]
                      if args.command == "qsa" else [m.key for m in model if isinstance(m, PLELayer)])
        require(bool(candidates), f"checkpoint contains no {args.command.upper()} module")
        require(not args.module or args.module in candidates, f"requested module missing: {args.module}; available: {candidates}")
    finally:
        config.stc.close()
    import yaml
    cfg = yaml.safe_load(Path(args.serving_config).read_text())
    model_cfg = cfg.get("model", {})
    require(str(model_cfg.get("cache_mode")).replace(" ", "") == f"{args.k_bits},{args.v_bits}",
            f"served cache_mode {model_cfg.get('cache_mode')!r} differs from declared {args.k_bits},{args.v_bits}")
    require(model_cfg.get("tensor_parallel") is False, "served config must explicitly keep tensor_parallel: false")
    require(Path(model_cfg.get("model_name", "")).name == Path(args.model).name,
            "served model_name does not identify the requested checkpoint")
    for key in ("cache_compand_a", "compand_a"):
        if key in model_cfg:
            require(float(model_cfg[key]) == args.compand_a, f"{key} differs from --compand-a")
    report.environment.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                              cudnn=torch.backends.cudnn.version(),
                              torch_build_config=torch.__config__.show(), native_compilation_performed=False,
                              driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader"], text=True).strip().splitlines(),
                              gpu=[dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i))) for i in range(2)],
                              kernel_flags={k: v for k, v in os.environ.items() if k.startswith(("EXL", "CUDA", "NCCL", "PYTORCH", "TORCH", "CUBLAS", "TRITON", "OMP", "MKL"))},
                              kv=dict(k_bits=args.k_bits, v_bits=args.v_bits, compand_a=args.compand_a, probe_compand_a=0.25),
                              package=str(package), extension=spec.origin,
                              reconstruction={"served_no_reconstruct": args.no_reconstruct, "default_no_reconstruct": False})
    report.environment["effective_infer_params"] = {k: v for k, v in vars(config.infer_params).items()
                                                    if isinstance(v, (bool, int, float, str)) or v is None}
    report.environment["torch_numeric_flags"] = dict(matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)
    return {"package": str(package), "precompiled_extension": spec.origin}


def provenance(args, report):
    root = Path(args.model)
    files = checkpoint_files(root)
    entries = []
    for p in files:
        print(f"HASH {p}", flush=True)
        before = p.stat()
        digest = sha256(p)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"checkpoint changed while hashing: {p}")
        entries.append(dict(path=str(p.relative_to(root)), bytes=after.st_size, sha256=digest, mtime_ns=after.st_mtime_ns))
    package = Path(report.environment["package"])
    report.environment.update(checkpoint_path=str(root), checkpoint_files=entries,
        checkpoint_manifest_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        config_sha256=sha256(root / "config.json"), serving_config_sha256=sha256(args.serving_config),
        extension_sha256=sha256(report.environment["extension"]), harness_sha256=sha256(__file__),
        source_sha256={str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))})
    stage = package.parent / "stage.json"
    if stage.is_file():
        report.environment["overlay"] = json.loads(stage.read_text())
    return {"checkpoint_files": len(entries), "hash_method": "full file SHA-256; no size/mtime-only fingerprint"}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="standard-library-only tests; no GPU/torch/model")
    s = sub.add_parser("stage", help="copy installed package and apply exact-context Python transport patches")
    s.add_argument("--destination", required=True)
    s.add_argument("--kind", choices=("qsa", "ple", "all"), default="all")
    s.add_argument("--patches", default=str(Path(__file__).with_name("ep-accept-patches")))
    for kind in ("qsa", "ple"):
        q = sub.add_parser(kind, allow_abbrev=False)
        q.add_argument("--model", default=DEFAULT_MODEL)
        q.add_argument("--serving-config", default="/srv/qwen5090/flashnext-config.yml")
        q.add_argument("--output", required=True)
        q.add_argument("--module", help="exact module key; default runs every matching module, one at a time")
        q.add_argument("--k-bits", type=int, choices=range(2, 9), default=8)
        q.add_argument("--v-bits", type=int, choices=range(2, 9), default=8)
        q.add_argument("--compand-a", type=float, default=0.0, help="explicit served companding value, recorded even when zero")
        q.add_argument("--no-reconstruct", action="store_true", help="served InferParams.no_reconstruct; also test default if different")
        q.add_argument("--timeout", type=int, default=1800, help="per-worker handshake/phase deadline in seconds")
        q.add_argument("--preflight", action="store_true", help="imports and prerequisites only; never loads weights")
        q.add_argument("--not-run-reason", help=argparse.SUPPRESS)
    return p


def cpu_tensor(t):
    return t.detach().to("cpu", copy=True).contiguous()


def tensor_digest(t):
    import torch
    c = cpu_tensor(t)
    return dict(shape=list(c.shape), dtype=str(c.dtype),
                sha256=hashlib.sha256(c.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def compare_tensors(actual, expected, tolerance):
    import torch
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert bool(torch.isfinite(actual).all()), "nonfinite actual tensor"
    assert bool(torch.isfinite(expected).all()), "nonfinite reference tensor"
    diff = (actual.double() - expected.double()).abs()
    flat = int(diff.argmax()) if diff.numel() else 0
    maximum = float(diff.max()) if diff.numel() else 0.0
    relative = float((diff / expected.double().abs().clamp_min(1e-12)).max()) if diff.numel() else 0.0
    metrics = dict(max_abs=maximum, max_relative=relative, max_abs_flat_index=flat,
                   bitwise_equal=torch.equal(actual, expected), shape=list(actual.shape))
    try:
        torch.testing.assert_close(actual, expected, **tolerance, equal_nan=False)
    except AssertionError as e:
        raise AssertionError(f"{metrics}\n{e}") from e
    return metrics


class Recorder:
    """Streaming artifacts: references never come from an imported module.

    Each event stores its own CPU tensors; comparison consumes one event at a time.
    Every mismatch is reported with errors/locations; it does not hide later events.
    """
    def __init__(self, report, root, phase, write=False):
        self.report, self.root, self.phase, self.write = report, Path(root), phase, write
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = []

    def emit(self, family, name, tensors, tolerance=TOL):
        import torch
        self.events.append(name)
        path = self.root / (hashlib.sha256(name.encode()).hexdigest() + ".pt")
        values = {k: cpu_tensor(t) for k, t in tensors.items()}
        def test():
            assert values, "empty numerical event"
            for k, t in values.items():
                assert bool(torch.isfinite(t).all()), f"nonfinite {k}"
            if self.write:
                torch.save(values, path)
                return {"artifact": str(path), "tensors": {k: list(t.shape) for k, t in values.items()}}
            require(path.is_file(), f"reference event missing: {name} ({path})")
            expected = torch.load(path, map_location="cpu", weights_only=True)
            assert values.keys() == expected.keys(), "reference tensor set differs"
            metrics = {k: compare_tensors(v, expected[k], EXACT if not v.is_floating_point() else tolerance)
                       for k, v in values.items()}
            if self.report.kind == "qsa" and "output" in values:
                reference = expected["output"]
                if bool(reference.count_nonzero()):
                    assert not torch.equal(values["output"], 2 * reference), "replica output is twice the reference"
                    metrics["output"]["equals_twice_reference"] = False
            return metrics
        target = "repeatability" if self.phase == "repeat" else family
        self.report.check(target, f"{self.phase}/{self.root.parent.name}/{self.root.name}/{name}", test, tolerance)

    def finish(self):
        path = self.root / "events.json"
        if self.write:
            atomic_json(path, self.events)
        else:
            require(path.is_file(), f"reference event manifest missing: {path}")
            assert self.events == json.loads(path.read_text()), "replica omitted/reordered reference events"


class NoReduction:
    def all_reduce(self, *args, **kwargs):
        raise AssertionError("QSA/PLE replica attempted backend.all_reduce")


def module_fingerprint(module):
    """Hash loaded bytes (including EXL3 trellis/scales/bias), not regenerated weights."""
    attrs = ("hidden_size", "num_q_heads", "num_kv_heads", "head_dim", "layer_idx", "out_dtype",
             "n_heads", "kv_heads", "token_budget", "compress_ratio", "block_topk", "scale",
             "in_features", "out_features", "in_features_unpadded", "out_features_unpadded", "quant_type",
             "rms_norm_eps", "constant_bias", "constant_scale", "span_heads", "groups", "unweighted",
             "ngram_size", "heads_per_ngram", "ple_embed_dim", "eos_token_id", "mm_token_id", "hc_mult",
             "conv_kernel_size", "conv_state_len", "gate_scale", "mode", "K", "rows_per_shard", "num_rows", "_row_dtype")
    result = {}
    for m in module:
        d = {k: str(getattr(m, k)) for k in attrs if hasattr(m, k)}
        d["class"] = type(m).__name__
        d["tensors"] = {k: tensor_digest(v) for k, v in m.get_tensors().items() if v is not None}
        for k in ("head_offsets", "head_vocab_sizes", "layer_multipliers", "head_bias"):
            v = getattr(m, k, None)
            if v is not None:
                d[k] = tensor_digest(v)
        result[m.key] = d
        d["no_reconstruct"] = m.config.infer_params.no_reconstruct if hasattr(m.config, "infer_params") else False
        # Capture constructor settings beyond the common geometry list, but exclude
        # child objects and checkpoint-selection inputs (their loaded bytes are above).
        import inspect
        import dataclasses
        for name in inspect.signature(type(m).__init__).parameters:
            if name in ("self", "config", "qmap", "qbits_key", "stream_from_disk") or not hasattr(m, name):
                continue
            value = getattr(m, name)
            if dataclasses.is_dataclass(value):
                d[name] = str(dataclasses.asdict(value))
            elif value is None or isinstance(value, (str, int, float, bool, tuple, list)) or type(value).__name__ == "dtype":
                d[name] = str(value)
    if hasattr(module, "cache_layers"):
        result["__caches__"] = {str(cl.cache_id): dict(cls=type(cl).__name__, shape=list(cl.shape),
            raw_shape=list(cl.raw_k_shape), pool_shape=list(cl.pooled_shape), capacity=cl.max_num_tokens,
            k_bits=getattr(cl, "k_bits", 16), v_bits=getattr(cl, "v_bits", 16), compand_a=getattr(cl, "compand_a", 0.0))
            for cl in module.cache_layers}
    return result


def memory_sample(device):
    import torch
    import resource
    proc = {}
    for path in ("/proc/self/smaps_rollup", "/proc/self/status"):
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.split(":")[0] in ("Rss", "Pss", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty", "VmHWM", "VmRSS"):
                    proc[line.split(":")[0]] = line.split(":", 1)[1].strip()
    shm = shutil.disk_usage("/dev/shm")
    return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device),
                max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, proc=proc,
                shm_used=shm.used, fds=len(list(Path("/proc/self/fd").iterdir())),
                sampled_host_peaks=dict(HOST_MONITOR.peaks) if HOST_MONITOR is not None else {})


class HostMonitor:
    """Sample process PSS/RSS, fd count and host SHM at 250 ms; RSS HWM is also reported."""
    def __init__(self):
        self.peaks = {"period_seconds": 0.25}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                path = Path("/proc/self/smaps_rollup")
                for line in path.read_text().splitlines():
                    name, _, value = line.partition(":")
                    if name in ("Rss", "Pss", "Private_Dirty", "Shared_Dirty"):
                        key = name + "_KiB"
                        self.peaks[key] = max(self.peaks.get(key, 0), int(value.split()[0]))
                self.peaks["shm_used_bytes"] = max(self.peaks.get("shm_used_bytes", 0), shutil.disk_usage("/dev/shm").used)
                self.peaks["fds"] = max(self.peaks.get("fds", 0), len(list(Path("/proc/self/fd").iterdir())))
            except OSError as exc:
                self.peaks["sampling_error"] = str(exc)
            self.stop_event.wait(0.25)

    def close(self):
        self.stop_event.set()
        self.thread.join(2)


def qsa_geometry(idx, page):
    t = idx.sparse_threshold()
    length = max(t + page + 5, (idx.SEL_TILE + 2) * idx.compress_ratio + 5)
    per_row = math.ceil((length + page) / page) * page
    return t, length, per_row


def qsa_schedule(length, threshold, page, tile_boundary, start=0):
    # End chunks immediately before each interesting position, then use scalar decode.
    special = {p for b in (threshold, page, tile_boundary) for p in range(b - 2, b + 4)}
    special |= set(range(length - 8, length))
    pos, result = start, []
    while pos < length:
        if pos in special:
            n = 1
        elif pos == threshold + 4:
            n = 4
        else:
            nxt = min((p for p in special if p > pos), default=length)
            n = min(257, nxt - pos, length - pos)
        result.append((pos, n))
        pos += n
    return result


def reset_qsa(module):
    for cl in module.cache_layers:
        for t in cl.get_tensors():
            t.zero_()


def qsa_params(module, cl, lengths, block_table):
    import torch
    from exllamav3.modules.attn import prepare_for_attn
    params = dict(attn_mode="flash_attn", cache=cl, block_table=block_table,
                  cache_seqlens=torch.tensor(lengths, dtype=torch.int32), backend=NoReduction())
    prepare_for_attn(torch.zeros((len(lengths), 1), dtype=torch.long), params)
    if module.has_split_cache:
        params["cache"] = cl.cache_id
    return params


def qsa_planes(cl, bt, lengths):
    result = {}
    for b, length in enumerate(lengths):
        pages = bt[b].to(cl.device).long()
        result[f"raw/{b}"] = cl.raw_k[pages].flatten(0, 1)[:length]
        result[f"pool/{b}"] = cl.pooled[pages].flatten(0, 1)[:length // cl.compress_ratio]
    return result


def tail_assert(indices, lengths, seq, cr):
    import torch
    for b, pos in enumerate(lengths):
        for s in range(seq):
            end = pos + s
            row = indices[b * seq + s]
            assert bool(((row == -1) | ((row >= 0) & (row <= end))).all()), "future/out-of-range selection"
            for token in range(((end + 1) // cr) * cr, end + 1):
                assert bool((row == token).any()), f"incomplete tail token {token} omitted"
            valid = row[row >= 0]
            assert torch.unique(valid).numel() == valid.numel(), "duplicate selected token"


def qsa_cached(module, inputs, rec, graph=False):
    import torch
    import exllamav3.modules.attn as attn_impl
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.util.tensor import g_tensor_cache
    idx = module.qsa_indexer
    threshold, length, per_row = qsa_geometry(idx, PAGE_SIZE)
    reset_qsa(module)
    old_enable = attn_impl._bc_attn_enable
    attn_impl._bc_attn_enable = graph
    bc_hits = {}
    original_bc = module.bc_attn_step
    def observe(x, cl, params, bt, sl, host_seqlens=None):
        y = original_bc(x, cl, params, bt, sl, host_seqlens=host_seqlens)
        if y is not None:
            regime = int(int(host_seqlens.max()) + x.shape[1] > threshold)
            key = (cl.cache_id, x.shape[0], x.shape[1], regime)
            bc_hits[key] = bc_hits.get(key, 0) + 1
        return y
    module.bc_attn_step = observe
    try:
        # Graph sparse slots are single-row in baseline; independently test bsz=2 eager.
        bsz = 1 if graph or rec.root.name == "single" else 2
        bt = torch.arange(2 * per_row // PAGE_SIZE, dtype=torch.int32).view(2, -1)[:bsz].clone()
        for cache_no, cl in enumerate(module.cache_layers):
            bt_run = bt.clone()
            offset = 7 if bsz == 2 else 0
            if offset:
                p = qsa_params(module, cl, [0], bt_run[:1])
                module.forward(inputs[:1, :offset].to(module.device), p)
            for pos, n in qsa_schedule(length, threshold, PAGE_SIZE, idx.SEL_TILE * idx.compress_ratio, offset):
                lengths = [pos] if bsz == 1 else [pos, pos - offset]
                x = torch.stack([inputs[b, start:start+n] for b, start in enumerate(lengths)]).to(module.device).contiguous()
                p = qsa_params(module, cl, lengths, bt_run)
                y = module.forward(x, p)
                end = [v + n for v in lengths]
                name = f"cache{cache_no}/p{pos}/n{n}"
                rec.emit("qsa.cache", name, {"output": y, **qsa_planes(cl, bt_run, end)})
                # Project queries separately without writing the raw/pool planes under test.
                q = torch.cat([idx.project(x[b:b+1], module.rope, {}, position=start)[0]
                               for b, start in enumerate(lengths)])
                selected = idx.select_indices_paged(cl, q, bt_run.to(module.device), torch.tensor(lengths, dtype=torch.int32))
                rec.report.check("qsa.indices", f"{rec.phase}/{rec.root.parent.name}/{rec.root.name}/{name}/tail",
                                 lambda: tail_assert(selected, lengths, n, idx.compress_ratio))
                rec.emit("qsa.indices", name + "/indices", {"indices": selected}, EXACT)
                # Read the actual BC selection buffer before another BC call can overwrite it.
                if graph and bsz == 1 and n <= attn_impl._bc_max_qlen and end[0] > threshold:
                    key = (cl.cache_id, bsz, n, 1)
                    if bc_hits.get(key, 0):
                        actual = g_tensor_cache.get_bucketed(module.device, n * idx.k_pad(), torch.int32, "bca_qsa_indices").view(n, idx.k_pad()).clone()
                        rec.report.check("qsa.graph", f"{rec.phase}/{rec.root.parent.name}/{name}/bc-indices",
                                         lambda: compare_tensors(cpu_tensor(actual), cpu_tensor(selected), EXACT))
            # Copy-on-write: redirect the last partially filled page to an unused page,
            # continue, and prove all bytes of the shared source page remain unchanged.
            pos = length
            src_col = (pos - 1) // PAGE_SIZE
            count = (pos - 1) % PAGE_SIZE + 1
            src, dst = int(bt_run[0, src_col]), int(bt_run[0, -1])
            assert src != dst
            before = [t[src].clone() for t in cl.get_tensors()]
            cl.copy_page(cl, src, dst, count)
            for t, saved in zip(cl.get_tensors(), before):
                count_t = math.ceil(count / idx.compress_ratio) if t is cl.pooled else count
                torch.testing.assert_close(t[dst, :count_t], saved[:count_t], **EXACT)
            bt_run[0, src_col] = dst
            p = qsa_params(module, cl, [pos], bt_run[:1])
            y = module.forward(inputs[:1, :4].to(module.device).contiguous(), p)
            for t, saved in zip(cl.get_tensors(), before):
                torch.testing.assert_close(t[src], saved, **EXACT)
            rec.emit("qsa.copy", f"cache{cache_no}/continuation", {"output": y, **qsa_planes(cl, bt_run[:1], [pos + 4])})
        if graph:
            def verify_bc():
                for cl in module.cache_layers:
                    for regime in (0, 1):
                        require(bc_hits.get((cl.cache_id, 1, 1, regime), 0) >= 3,
                                f"BC declined or did not warm up/capture/replay: cache={cl.cache_id}, regime={regime}, hits={bc_hits}")
                return {"bc_successful_calls": {str(k): v for k, v in bc_hits.items()},
                        "capture_contract": "BC_Attention.run: first eager, second capture, subsequent replay"}
            rec.report.check("qsa.graph", f"{rec.phase}/{rec.root.parent.name}/activation", verify_bc)
    finally:
        module.bc_attn_step = original_bc
        attn_impl._bc_attn_enable = old_enable
    rec.finish()


def qsa_nc(module, inputs, rec):
    from exllamav3.constants import PAGE_SIZE
    idx = module.qsa_indexer
    threshold, length, _ = qsa_geometry(idx, PAGE_SIZE)
    for n in sorted(set([1, idx.compress_ratio-1, idx.compress_ratio, idx.compress_ratio+1, threshold, threshold+1, length])):
        require(n > 0, "compress_ratio must exceed 1 for the prescribed prefix test")
        x = inputs[:, :n].to(module.device).contiguous()
        p = dict(attn_mode="flash_attn_nc", backend=NoReduction())
        y, stages = qsa_nc_trace(module, x, p)
        q, raw = idx.project(x, module.rope, p)
        pool = idx.pool_keys(raw, module.rope, p)
        rec.emit("numerics", f"prefix{n}", {"output": y, "index_q": q, "raw": raw, "pool": pool, **stages})
        selected = idx.select_indices(q, pool, 0, n)
        rec.emit("qsa.indices", f"prefix{n}/indices", {"indices": selected}, EXACT)
        if n == length:
            rec.report.add("memory", f"{rec.phase}/{rec.root.parent.name}/observed-at-L", "PASS",
                           "allocator counters sampled with the full L tensors live; replica estimate comparison is a separate check",
                           L=length, **memory_sample(module.device))
    rec.finish()


def qsa_nc_trace(module, x, params):
    """Record projection -> normalized/roped Q -> attention -> O diagnostic stages.

    Wrap real calls without replacing their math; snapshots are CPU copies so a
    later in-place norm/RoPE cannot rewrite an earlier diagnostic checkpoint.
    """
    import exllamav3.modules.attn as ai
    trace = {}
    idx = module.qsa_indexer
    old_qkv, old_o = module.project_qkv, module.project_o
    old_dispatch, old_sparse = ai.attn_dispatch, idx.sparse_attend_nc
    def project(*args, **kwargs):
        result = old_qkv(*args, **kwargs)
        for name, value in zip(("projected_q", "projected_k", "projected_v", "gate"), result):
            if value is not None:
                trace[name] = cpu_tensor(value)
        return result
    def project_o(o, *args, **kwargs):
        trace["o_projection_input"] = cpu_tensor(o)
        return old_o(o, *args, **kwargs)
    def dispatch(*args, **kwargs):
        trace["normalized_roped_q"] = cpu_tensor(kwargs["q"])
        result = old_dispatch(*args, **kwargs)
        trace["attention_output"] = cpu_tensor(result)
        return result
    def sparse(attn, q, k, v, *args, **kwargs):
        trace["normalized_roped_q"] = cpu_tensor(q)
        result = old_sparse(attn, q, k, v, *args, **kwargs)
        trace["attention_output"] = cpu_tensor(result)
        return result
    module.project_qkv, module.project_o = project, project_o
    ai.attn_dispatch, idx.sparse_attend_nc = dispatch, sparse
    try:
        y = module.forward(x, params)
        return y, trace
    finally:
        module.project_qkv, module.project_o = old_qkv, old_o
        ai.attn_dispatch, idx.sparse_attend_nc = old_dispatch, old_sparse


def decompanded_sparse(layer, attn, q, q_idx, bt, lengths):
    """Independent float32 gathered attention on ext.dequant_cache_paged's decompanded KV.

    Includes only selected causal tokens. This does not reuse the packed sparse reader.
    """
    import torch
    from exllamav3.constants import PAGE_SIZE
    require(attn.sinks is None and not attn.logit_softcapping and attn.sliding_window in (-1, None),
            "decompanded oracle requires unsunk, unsoftcapped full attention")
    bsz, seq, heads, dim = q.shape
    end = (lengths + seq).to(attn.device)
    k, v = layer.get_kv(end, bt)
    sel = attn.qsa_indexer.select_indices_paged(layer, q_idx, bt, lengths)
    out = torch.empty_like(q)
    group = heads // attn.num_kv_heads
    for b in range(bsz):
        for s in range(seq):
            tok = sel[b * seq + s]
            tok = tok[tok >= 0].long()
            pages = bt[b, tok // PAGE_SIZE].long()
            kk = k[pages, tok % PAGE_SIZE].repeat_interleave(group, dim=1).float()
            vv = v[pages, tok % PAGE_SIZE].repeat_interleave(group, dim=1).float()
            scores = torch.einsum("hd,khd->hk", q[b, s].float(), kk) * attn.sm_scale
            out[b, s] = torch.einsum("hk,khd->hd", scores.softmax(-1), vv).half()
    return out


def qsa_companding(module, inputs, report, tag):
    import torch
    import exllamav3.modules.attn as ai
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.cache.qsa import CacheLayer_qsa_quant
    cl = module.cache_layers[0]
    if not isinstance(cl, CacheLayer_qsa_quant):
        return
    old_a, enabled = cl.compand_a, ai._bc_attn_enable
    idx = module.qsa_indexer
    orig = idx.sparse_attend
    try:
        ai._bc_attn_enable = False
        # Exercise both the served setting and the mandatory nonzero stress setting.
        for value in sorted(set([old_a, 0.25])):
            cl.compand_a = value
            reset_qsa(module)
            bt = torch.arange(cl.max_num_tokens // PAGE_SIZE, dtype=torch.int32).view(1, -1)
            n = idx.sparse_threshold() + 5
            for pos in range(0, n, 257):
                chunk = inputs[:1, pos:min(pos + 257, n)].to(module.device).contiguous()
                module.forward(chunk, qsa_params(module, cl, [pos], bt))
            x = inputs[:1, :4].to(module.device).contiguous()
            got = cpu_tensor(module.forward(x, qsa_params(module, cl, [n], bt)))
            idx.sparse_attend = decompanded_sparse
            expected = cpu_tensor(module.forward(x, qsa_params(module, cl, [n], bt)))
            idx.sparse_attend = orig
            report.check("qsa.companding", f"{tag}/a={value}", lambda: compare_tensors(got, expected, TOL), TOL)
    finally:
        cl.compand_a, ai._bc_attn_enable, idx.sparse_attend = old_a, enabled, orig


def qsa_reduction_control(context, exported, report, tag):
    import torch
    from exllamav3.modules.attn import Attention
    def test():
        try:
            bad = Attention.tp_import(context, exported, {}, skip_reduction=True)
        except (AssertionError, ValueError) as exc:
            assert "reduc" in str(exc).lower(), f"wrong rejection reason: {exc}"
        else:
            bad.unload()
            raise AssertionError("QSA accepted skip_reduction=True")
        ordinary = dict(exported, qsa_indexer=None, cache_layers=[])
        key = exported["kwargs"]["key"]
        control = Attention.tp_import(context, ordinary, {key: (0, 0, "heads")})
        assert control.tp_reduce and control.num_kv_heads == 0
        class ReductionObserved(Exception):
            pass
        class Backend:
            def all_reduce(self, *a, **kw):
                raise ReductionObserved()
        try:
            control.forward(torch.zeros((1, 1, control.hidden_size), device=context["device"], dtype=torch.half), {"backend": Backend()})
        except ReductionObserved:
            return {"ordinary_zero_head_rank_reduction": True, "qsa_combined_reduction_rejected": True}
        finally:
            control.unload()
        raise AssertionError("ordinary sharded-attention control suppressed its reduction")
    report.check("qsa.reduction", tag, test)


class PLEDriver:
    """Exercise GDNState's real controller and worker dispatch without Model.load_tp.

    A tiny cache view maps this process's controller object ID to the cache ID
    exported by the parent, just as the TP controller/worker boundary does.
    State tensors, GDNState, clear/stash/rewind/un-stash/delete are engine objects.
    """
    def __init__(self, module, context=None):
        from types import SimpleNamespace
        from exllamav3.modules.gated_delta_net import GDNState
        self.module, self.context = module, context
        self.imported = context is not None
        self.views, self.handles = [], []
        for rl in module.recurrent_layers:
            class CacheView:
                def get_all_recurrent_layers(self):
                    return {(module.layer_idx, 0): self.layer}

                def get_recurrent_layer(self, key):
                    assert key == (module.layer_idx, 0)
                    return self.layer
            view = CacheView()
            view.layer = rl
            def dispatch(fn, params, layer=rl):
                # Only the ID crosses the wire; never replace the underlying state.
                return fn(context, layer.cache_id, *params[1:])
            view.model = SimpleNamespace(loaded_tp=self.imported, tp_dispatch_all=dispatch)
            self.views.append(view)
            self.handles.append({s: GDNState(view, s, 0) for s in range(rl.max_batch_size)})

    def params(self, cache_no, slots, ids, save=False):
        import torch
        handles = [self.handles[cache_no][s] for s in slots]
        if self.imported:
            handles = [h.tp_export() for h in handles]
            for h in handles:
                h.cache = self.module.recurrent_layers[cache_no].cache_id
        return dict(input_ids=ids, recurrent_states=handles,
                    recurrent_slots=torch.tensor(slots, dtype=torch.long), recurrent_history=save,
                    position=handles[0].position, backend=NoReduction())

    def advance(self, cache_no, slots, n, save=False):
        for slot in slots:
            h = self.handles[cache_no][slot]
            h.position += n
            h.last_history = n if save else 0
            h.post_advance()

    def clear(self, cache_no, slot):
        from exllamav3.modules.gated_delta_net import GDNState
        with self.unchanged_except(cache_no, slot):
            self.handles[cache_no][slot] = GDNState(self.views[cache_no], slot, 0)

    def stash(self, cache_no, slot):
        with self.unchanged_except(-1, -1):
            return self.handles[cache_no][slot].stash()

    def unstash(self, cache_no, slot, stashed):
        with self.unchanged_except(cache_no, slot):
            h = self.handles[cache_no][slot]
            h.position = stashed["position"]
            h.unstash(stashed)

    def rewind(self, cache_no, slot, rejected):
        with self.unchanged_except(cache_no, slot):
            self.handles[cache_no][slot].rewind(rejected)

    @contextlib.contextmanager
    def unchanged_except(self, cache_no, slot):
        import torch
        before = {k: cpu_tensor(v) for k, v in self.snapshot().items()}
        yield
        after = self.snapshot()
        for c, rl in enumerate(self.module.recurrent_layers):
            untouched = [s for s in range(rl.max_batch_size) if c != cache_no or s != slot]
            for prefix in ("ids", "conv"):
                torch.testing.assert_close(cpu_tensor(after[f"{prefix}{c}"][untouched]), before[f"{prefix}{c}"][untouched], **EXACT)

    def saved_tensors(self, stashed):
        if self.imported:
            return self.context["recurrent_cache"][stashed["tp_handle"]][0]
        return stashed[(self.module.layer_idx, 0)]

    def delete(self, stashed):
        if self.imported:
            from exllamav3.cache.recurrent import mp_cache_recurrent_del
            handle = stashed["tp_handle"]
            mp_cache_recurrent_del(self.context, 0, handle)
            assert handle not in self.context["recurrent_cache"]

    def snapshot(self):
        import torch
        result = {}
        for i, rl in enumerate(self.module.recurrent_layers):
            assert rl.id_state.device.type == "cpu", "carried PLE IDs moved to GPU"
            assert rl.conv_state.device == torch.device(self.module.device), "convolution on wrong rank"
            result[f"conv{i}"] = rl.conv_state[:, :, :rl.win]
            result[f"ids{i}"] = rl.id_state[:, :rl.ctx]
        return result


def exact_hash_ids(embedding, history):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    out_len = history.shape[1] - embedding.context_len
    n = history.shape[0] * out_len * embedding.num_heads
    uids = torch.empty(n, dtype=torch.int64, pin_memory=True)
    inverse = torch.empty(n, dtype=torch.int64, pin_memory=True)
    heads = torch.empty(n, dtype=torch.int32, pin_memory=True)
    count = ext.ngram_hash_cpu(history.contiguous(), out_len, embedding.layer_multipliers,
                              embedding.head_offsets, embedding.head_vocab_sizes,
                              embedding.heads_per_ngram, embedding.eos_token_id, uids, inverse, heads)
    actual = uids[:count][inverse].view(history.shape[0], out_len, embedding.num_heads)
    expected = embedding.compute_ngram_ids(history, out_len)
    torch.testing.assert_close(actual, expected, **EXACT)
    return actual


def ple_scenario(module, data, rec, context=None):
    import torch
    from exllamav3.cache.recurrent import RecurrentCache
    streams, ids = data["streams"], data["ids"]
    driver = PLEDriver(module, context)
    emb = module.ple_embedding

    def call(name, cache_no, slots, rows, start, n, save=False, family="ple.state", record=True):
        x = streams[rows, start:start+n].to(module.device).contiguous()
        tokens = ids[rows, start:start+n].clone()
        p = driver.params(cache_no, slots, tokens, save)
        history, _, _ = module._state_history(module._prepare_ids(tokens), p)
        hashed = exact_hash_ids(emb, history)
        before = {k: cpu_tensor(v) for k, v in driver.snapshot().items()}
        y = module.forward(x, p)
        driver.advance(cache_no, slots, n, save)
        state = driver.snapshot()
        # Unrelated caches and nonparticipating slots must not change.
        for c, rl in enumerate(module.recurrent_layers):
            untouched = [s for s in range(rl.max_batch_size) if c != cache_no or s not in slots]
            for prefix in ("ids", "conv"):
                torch.testing.assert_close(cpu_tensor(state[f"{prefix}{c}"][untouched]),
                                           before[f"{prefix}{c}"][untouched], **EXACT)
        tensors = {"output": y, "delta": y - x, "hashes": hashed, **state}
        if record:
            rec.emit(family, name, tensors, EXACT)
        return {k: cpu_tensor(v) for k, v in tensors.items()}

    x = streams[:, :4].to(module.device).contiguous()
    h = module._history(module._prepare_ids(ids[:, :4]))
    out = module.forward(x, {"input_ids": ids[:, :4], "position": 0})
    ref_delta, _ = module.forward_streams_reference(x, emb.forward_reference(h, {}), {})
    rec.report.check("numerics", f"{rec.phase}/{rec.root.parent.name}/stateless-fast-reference",
                     lambda: compare_tensors(cpu_tensor(out), cpu_tensor(x + ref_delta), TOL), TOL)
    rec.emit("numerics", "stateless-position-zero", {"output": out, "hashes": exact_hash_ids(emb, h)}, EXACT)
    for cache_no in range(2):
        call(f"cache{cache_no}/prefill257", cache_no, [1, 5], [0, 1], 0, 257)
        # Swapping rows/slots preserves sequence identity; logical slots are never compacted.
        for start in range(257, 262):
            call(f"cache{cache_no}/decode{start}", cache_no, [5, 1], [1, 0], start, 1)
        call(f"cache{cache_no}/verify4", cache_no, [1, 5], [0, 1], 262, 4)
        for hist in (1, 3, 8):
            for reject in sorted(set([0, 1, hist])):
                committed = driver.stash(cache_no, 1)
                saved_before = [cpu_tensor(t) for t in driver.saved_tensors(committed)]
                # Oracle consumes only accepted tokens, then the continuation.
                accepted = hist - reject
                if accepted:
                    call("oracle-accepted", cache_no, [1], [0], 266, accepted, record=False)
                expected = call("oracle-continuation", cache_no, [1], [0], 280, 1, record=False)
                driver.unstash(cache_no, 1, committed)
                call(f"cache{cache_no}/spec{hist}-reject{reject}", cache_no, [1], [0], 266, hist,
                     save=True, family="ple.checkpoints")
                driver.rewind(cache_no, 1, reject)
                actual = call(f"cache{cache_no}/continue{hist}-reject{reject}", cache_no, [1], [0], 280, 1,
                              family="ple.checkpoints")
                def rewind_check():
                    errors = {k: compare_tensors(actual[k], expected[k], EXACT if k.startswith("ids") or k == "hashes" else TOL)
                              for k in actual}
                    for a, b in zip(driver.saved_tensors(committed), saved_before):
                        compare_tensors(cpu_tensor(a), b, EXACT)
                    return errors
                rec.report.check("ple.checkpoints", f"{rec.phase}/{rec.root.parent.name}/cache{cache_no}/rewind{hist}-{reject}", rewind_check, TOL)
                driver.unstash(cache_no, 1, committed)
                driver.delete(committed)
        committed = driver.stash(cache_no, 1)
        saved = [cpu_tensor(t) for t in driver.saved_tensors(committed)]
        call(f"cache{cache_no}/after-stash", cache_no, [1], [0], 284, 3)
        driver.clear(cache_no, 1)
        for a, b in zip(driver.saved_tensors(committed), saved):
            compare_tensors(cpu_tensor(a), b, EXACT)
        driver.unstash(cache_no, 3, committed)
        rec.emit("ple.checkpoints", f"cache{cache_no}/clear-unstash-slot3", driver.snapshot(), EXACT)
        call(f"cache{cache_no}/restored-continuation", cache_no, [3], [0], 288, 4, family="ple.checkpoints")
        driver.delete(committed)
        # Real LRU eviction/deletion through RecurrentCache (capacity one checkpoint).
        h1, h5 = driver.handles[cache_no][3], driver.handles[cache_no][5]
        rc = RecurrentCache(driver.views[cache_no].model, max_size=h1.checkpoint_size)
        rc.put("first", h1)
        old = rc["first"].get("tp_handle")
        rc.put("second", h5)
        assert list(rc) == ["second"] and rc.metrics["stash_evictions"] == 1
        if context is not None:
            assert old not in context["recurrent_cache"]
        driver.delete(rc["second"])
        rc.clear()
        rec.emit("ple.checkpoints", f"cache{cache_no}/eviction", driver.snapshot(), EXACT)
    rec.finish()


def ple_prefetch(module, data, report, tag):
    import torch
    from exllamav3.modules.ngram_embedding import PREFETCH_ENABLED
    emb = module.ple_embedding
    require(PREFETCH_ENABLED, "EXL3_NGRAM_PREFETCH=0: prefetch acceptance cannot run")
    h = module._history(module._prepare_ids(data["ids"][:, :257]))
    emb._drain_prefetch()
    base = dict(emb.prefetch_stats)
    emb.prefetch(h)
    require(bool(emb._pending), "parent/worker prefetch did not enqueue the >=256-token input")
    a = emb.forward(h, {})
    b = emb.forward(h, {})
    torch.testing.assert_close(a, b, **EXACT)
    for delta in range(3):
        stale = h.clone()
        stale[:, -1] += delta + 1
        emb.prefetch(stale)
    # More pending inputs than staging sets must retire a stale prefetch.
    emb.forward(h, {})
    now = emb.prefetch_stats
    assert now["hit"] > base["hit"] and now["miss"] > base["miss"] and now["retired"] > base["retired"]
    report.add("ple.prefetch", tag, "PASS", "hit, miss, stale retirement and repeated upload compared bitwise", stats=dict(now))
    # Deliberately leave work pending; the transport/lifecycle owner calls unload.
    emb.prefetch(h)
    require(bool(emb._pending), "pending-unload precondition was not reached")


def ple_graph_boundary(module, data, report, tag):
    """Runnable eager CPU boundary + graph of the GPU-only PLE reference body.

    No claim that enclosing PLELayer.forward in a CUDA graph is supported. CPU
    hashing, history, gather and state commits execute on every invocation.
    """
    import torch
    emb = module.ple_embedding
    device = module.device
    x = data["streams"][:1, :1].to(device).contiguous()
    ids = data["ids"][:1, :1]
    history = module._history(module._prepare_ids(ids))
    e = emb.forward(history, {}).clone()
    cs = torch.zeros((1, module.hc_mult * module.hidden_size, module.conv_state_len), device=device, dtype=torch.half)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            module.forward_streams_reference(x, e, {}, cs)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        delta, columns = module.forward_streams_reference(x, e, {}, cs)
    outputs = []
    token_context = history[:, :emb.context_len].clone()
    ref_cs = cs.clone()
    for pos in range(12):
        # This is the enforced boundary: calling it while capturing is rejected.
        require(not torch.cuda.is_current_stream_capturing(), "PLE CPU history/hash/gather must execute eagerly")
        tokens = module._prepare_ids(data["ids"][:1, 256+pos:257+pos])
        history = torch.cat((token_context, tokens), dim=1)
        live_x = data["streams"][:1, 256+pos:257+pos].to(device).contiguous()
        live_e = emb.forward(history, {})
        expected, ref_columns = module.forward_streams_reference(live_x, live_e, {}, ref_cs)
        expected = cpu_tensor(expected)
        ref_cs = ref_columns[:, :, -module.conv_state_len:].clone()
        x.copy_(live_x)
        e.copy_(live_e)
        graph.replay()
        compare_tensors(cpu_tensor(delta), expected, TOL)
        cs.copy_(columns[:, :, -module.conv_state_len:])
        token_context = history[:, -emb.context_len:].clone()
        outputs.append(tensor_digest(delta)["sha256"])
    assert len(set(outputs)) > 1, "graph replay froze changing token/stream decisions"
    report.add("ple.graph", tag, "PASS", "12 changing-input replays with eager CPU history/hash/gather and state commits", TOL,
               boundary="GPU-only forward_streams_reference captured; PLELayer.forward remains eager",
               full_model_graph_compatibility="not established")


def ple_reload_cycles(module, context, data, report, tag):
    """Repeat real export/import/unload in the SAME process, after kernel warmup."""
    import torch
    from exllamav3.model.model_tp_shared import SMProducer
    rank = torch.device(module.device).index
    x = data["streams"][:1, :4].to(module.device).contiguous()
    ids = data["ids"][:1, :4]
    expected = cpu_tensor(module.forward(x, {"input_ids": ids, "position": 0}))
    samples = []
    for cycle in range(3):
        producer = SMProducer()
        exported = module.tp_export({}, producer)
        handles = list(module.ple_embedding.handles or [])
        module.unload()  # includes the deliberately queued prefetch on cycle zero
        assert not module.ple_embedding._pending and module.ple_embedding._executor is None
        assert all(h.fd is None for h in handles)
        new_module, new_context = import_replica("ple", rank, producer, exported, {})
        close_consumer(new_context)
        producer.clear()
        producer.buf[:] = 0x5A
        producer.close()
        del exported, producer, handles, module, context
        gc.collect()
        module, context = new_module, new_context
        got = module.forward(x, {"input_ids": ids, "position": 0})
        compare_tensors(cpu_tensor(got), expected, EXACT)
        del got
        # Check a quiescent point with a loaded module on every cycle; persistent
        # kernel caches have already warmed up before the first recorded sample.
        torch.cuda.synchronize(rank)
        samples.append(memory_sample(rank))
    assert samples[-1]["fds"] <= samples[0]["fds"], f"file descriptor count grew across identical reloads: {samples}"
    assert samples[-1]["allocated"] <= samples[0]["allocated"], f"live CUDA allocations grew across identical reloads: {samples}"
    report.add("ple.resources", tag + "/same-process-reload", "PASS",
               "three actual SHM reloads preserve outputs; live CUDA allocations and fd count do not grow", samples=samples)
    return module, context


def table_fixture_cases():
    formats = [("raw", "float16", 0, False), ("raw", "bfloat16", 0, False)]
    formats += [("trellis", "int16", k, bias) for k in range(1, 9) for bias in (False, True)]
    return [dict(format=f, dtype=d, K=k, bias=b, disk=disk, layout=layout)
            for f, d, k, b in formats for disk in (False, True)
            for layout in ("single", "nonadjacent", "coalesced")]


def write_table_fixture(module, case, directory):
    """Small legal tables, exact real served PLE projections/norms/conv unchanged."""
    import torch
    from safetensors.torch import save_file
    from exllamav3.loader import SafetensorsCollection
    from exllamav3.modules.ngram_embedding import NGramEmbedding
    from exllamav3.modules.quant.exl3_lib.ngram_codec import pack_rows
    from types import SimpleNamespace
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    old = module.ple_embedding
    key = old.key
    heads = old.num_heads
    # The short final shard is guaranteed; each hash head has its own small table.
    sizes = torch.full((heads,), 17, dtype=torch.long)
    offsets = torch.arange(heads, dtype=torch.long) * 17
    total = int(sizes.sum())
    generator = torch.Generator().manual_seed(1234)
    if case["format"] == "trellis":
        states = torch.randint(0, 65536, (total, 160), generator=generator, dtype=torch.long)
        rows = pack_rows(states, torch.full((total,), 0.1, dtype=torch.half), case["K"])
        aux = {f"{key}.head_offsets": offsets, f"{key}.head_vocab_sizes": sizes,
               f"{key}.layer_multipliers": torch.tensor([13 + 2*i for i in range(old.ngram_size)], dtype=torch.long)}
        if case["bias"]:
            aux[f"{key}.head_bias"] = torch.randn((heads, 160), generator=generator).half() * 0.01
        suffix = "trellis"
    else:
        rows = (torch.randn((total, 160), generator=generator) * 0.1).to(getattr(torch, case["dtype"]))
        parent = key.rsplit(".", 1)[0]
        aux = {f"{parent}.ngram_heads_offsets": offsets, f"{parent}.ngram_heads_vocab_sizes": sizes,
               f"{parent}.layer_multipliers": torch.tensor([13 + 2*i for i in range(old.ngram_size)], dtype=torch.long)}
        suffix = "weight"
    save_file(aux, str(directory / "aux.safetensors"))
    width = max(1, total // 3)
    if total % width == 0:
        width += 1
    boundaries = list(range(width, total, width))
    if case["layout"] == "single":
        save_file({f"{key}.{suffix}": rows}, str(directory / "rows.safetensors"))
        boundaries = []
    elif case["layout"] == "nonadjacent":
        for i, start in enumerate(range(0, total, width)):
            save_file({f"{key}.shard_{i}.{suffix}": rows[start:start+width].contiguous()}, str(directory / f"rows-{i}.safetensors"))
    else:
        save_file({f"{key}.shard_{i}.{suffix}": rows[start:start+width].contiguous()
                   for i, start in enumerate(range(0, total, width))}, str(directory / "rows.safetensors"))
    collection = SafetensorsCollection(str(directory))
    fixture = NGramEmbedding(SimpleNamespace(stc=collection), key, old.ngram_size, old.heads_per_ngram,
                            old.ple_embed_dim, old.eos_token_id, stream_from_disk=case["disk"], out_dtype=old.out_dtype)
    fixture.load(module.device)
    if case["format"] == "trellis" and not case["bias"]:
        assert fixture.head_bias is not None and int(fixture.head_bias.count_nonzero()) == 0
    stores = fixture.handles if case["disk"] else fixture.tables
    if case["layout"] == "coalesced" or not case["disk"]:
        assert len(stores) == 1, "adjacent shards/RAM slab not coalesced by actual loader"
    if case["layout"] == "nonadjacent" and case["disk"]:
        assert len(stores) > 1 and fixture.rows_per_shard == width
    module.ple_embedding = fixture
    module.modules[module.modules.index(old)] = fixture
    old.unload()
    probes = sorted(set([0, total-1] + [p for b in boundaries for p in (b-1, b)]))
    return collection, probes


def ple_fixture_scenario(module, data, rec):
    import torch
    emb = module.ple_embedding
    probes = torch.tensor(data["probes"], dtype=torch.long)
    packed = emb._fetch_packed(probes)
    rows = emb.fetch_rows(probes)
    ids = data["ids"][:, :257]
    history = module._history(module._prepare_ids(ids))
    hashed = exact_hash_ids(emb, history)
    fast, ref = emb.forward(history, {}), emb.forward_reference(history, {})
    rec.report.check("ple.tables", f"{rec.phase}/{rec.root.parent.name}/gather-reference",
                     lambda: compare_tensors(cpu_tensor(fast), cpu_tensor(ref), TOL), TOL)
    x = data["streams"][:, :257].to(module.device).contiguous()
    y = module.forward(x, {"input_ids": ids, "position": 0})
    rec.emit("ple.tables", "boundary-rows-hashes-output", {"packed": packed, "rows": rows, "hashes": hashed, "output": y}, EXACT)
    rec.finish()


def import_replica(kind, rank, producer, exported, plan):
    import torch
    from exllamav3.model.model_tp_shared import SMConsumer
    from exllamav3.model.model_tp_fn import mp_model_append
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    consumer = SMConsumer(producer, device=rank)
    context = dict(device=device, consumer=consumer, plan={device: plan}, modules=[],
                   kv_modules=[], recurrent_modules=[], recurrent_cache={})
    if kind == "ple":
        mp_model_append(context, exported)
        module = context["modules"][0]
    else:
        from exllamav3.modules.attn import Attention
        module = Attention.tp_import(context, exported, plan)
        context["modules"] = [module]
    return module, context


def close_consumer(context):
    # Match mp_close_consumer's close + drop, and collect the remote arena wrapper.
    consumer = context.pop("consumer")
    consumer.close()
    del consumer
    gc.collect()


def live_cuda_weight(module):
    # get_tensors() is a serialization interface: FP16 returns weight.T.contiguous(),
    # which is a COPY. Independence probes must mutate the actual live backing.
    for m in module:
        for owner in (m, getattr(m, "inner", None)):
            if owner is None:
                continue
            for name in ("conv_w", "weight", "trellis", "suh", "svh"):
                t = getattr(owner, name, None)
                if t is not None and t.is_cuda and t.numel() and t.is_contiguous():
                    return t
    raise NotRun("no contiguous live CUDA weight available for storage-independence probe")


def fingerprint_diff(actual, expected):
    """Field-by-field difference for the modules that disagree, so a mismatch reports WHAT differs.

    2026-09-16: this failure printed the keys that disagreed and nothing else, while the fingerprint compares loaded
    tensor digests AND constructor settings. "Which four modules" does not localise a cause; "which field" does. The
    same rule as every instrument defect found today: a check that fails must say why, or the next hour is spent
    guessing. Returns a JSON-serialisable dict.
    """
    out = {}
    for k in sorted(set(actual) | set(expected)):
        a, e = actual.get(k), expected.get(k)
        if a == e:
            continue
        entry = {}
        if isinstance(a, dict) and isinstance(e, dict):
            for f in sorted(set(a) | set(e)):
                if a.get(f) != e.get(f):
                    av, ev = a.get(f), e.get(f)
                    if isinstance(av, dict) and isinstance(ev, dict):
                        sub = {x: {"actual": av.get(x), "expected": ev.get(x)}
                               for x in sorted(set(av) | set(ev)) if av.get(x) != ev.get(x)}
                        entry[f] = {"kind": "mapping", "fields_differing": sub}
                    else:
                        entry[f] = {"kind": "scalar", "actual": av, "expected": ev}
        else:
            entry = {"kind": "type", "actual_class": type(a).__name__, "expected_class": type(e).__name__}
        out[k] = entry
    return out

def validate_replica(module, expected, kind, rank):
    import torch, json as _json, os as _os
    actual = module_fingerprint(module)
    keys = [m.key for m in module]
    assert len(keys) == len(set(keys)), "child registered more than once"
    if actual != expected:
        diff = fingerprint_diff(actual, expected)
        path = _os.environ.get("EP_ACCEPT_DIFF_DUMP")
        if path:
            try:
                with open(path, "w") as fh:
                    _json.dump(diff, fh, indent=1, default=str)
            except Exception as exc:
                print(f"WARN: could not write fingerprint diff to {path}: {exc}", flush=True)
        differing = ", ".join(sorted(diff))
        summary = []
        for k, entry in sorted(diff.items())[:4]:
            fields = ", ".join(sorted(entry.get("fields_differing", entry)))[:200]
            summary.append(f"{k.split('.')[-1]}: {fields}")
        assert actual == expected, ("loaded weight/config fingerprint differs: " + differing +
                                    " || differing fields: " + " ; ".join(summary))
    for m in module:
        for key, t in m.get_tensors().items():
            if t is not None and t.is_cuda:
                assert t.device.index == rank, f"{key} resides on wrong rank"
    for m in module:
        for key, t in m.get_tensors().items():
            if t is not None and t.is_cuda:
                assert t.device.index == rank, f"{key} resides on wrong rank"
    if kind == "qsa":
        assert module.num_q_heads > 0 and module.num_kv_heads > 0
        assert not module.tp_reduce and not module.tp_span_heads_norm
        assert set(module.tp_cache_lookup) == {101, 202}
        assert len(module.cache_layers) == 2
        for cl in module.cache_layers:
            assert module.tp_cache_lookup[cl.cache_id] is cl
            assert cl.raw_k.device.index == rank and cl.raw_k.shape == cl.raw_k_shape
            assert cl.pooled.shape == cl.pooled_shape and cl.shape[2] == module.num_kv_heads
        assert module.cache_layers[0].raw_k.data_ptr() != module.cache_layers[1].raw_k.data_ptr()
    else:
        assert module.layer_idx < 0
        assert set(module.tp_recurrent_lookup) == {101, 202}
        assert len(module.recurrent_layers) == 2
        for rl in module.recurrent_layers:
            assert module.tp_recurrent_lookup[rl.cache_id] is rl
            assert rl.id_state.device.type == "cpu" and rl.conv_state.device.index == rank
        assert module.recurrent_layers[0].conv_state.data_ptr() != module.recurrent_layers[1].conv_state.data_ptr()
        emb = module.ple_embedding
        assert emb._executor is None and not emb._pins and not emb._pending, "inherited async prefetch state"
        assert emb.prefetch_stats == {"hit": 0, "miss": 0, "retired": 0}
        if emb.tables is not None:
            assert all(t.is_shared() for t in emb.tables), "private per-rank host table clone"
        if emb.handles is not None:
            assert emb._owns_handles, "worker does not own disk handles"
            assert all(h.fd is None for h in emb.handles), "disk descriptor inherited instead of opened lazily"
    return {"rank": rank, "pid": os.getpid(), "device": str(module.device),
            "full_loaded_weight_hashes": actual}


def exercise_replica(module, context, spec, report):
    import torch
    data = torch.load(spec["input"], map_location="cpu", weights_only=True)
    root = Path(spec["reference"])
    phase = spec["tag"]
    if spec["kind"] == "qsa":
        def companding_metadata():
            exported_probe = spec["companding_probe"]  # exported in parent, actually crossed spawn IPC
            restored = exported_probe["cls"](None, module, **exported_probe["args"])
            assert restored.compand_a == 0.25 and restored.cache_id == 303
            assert restored.raw_k_shape == module.cache_layers[0].raw_k_shape
            assert restored.pooled_shape == module.cache_layers[0].pooled_shape
            return {"compand_a": restored.compand_a, "allocation": "metadata-only as prescribed"}
        report.check("qsa.companding", phase + "/metadata-0.25", companding_metadata)
        qsa_reduction_control(context, spec["exported"], report, phase)
        qsa_nc(module, data, Recorder(report, root / "nc", phase))
        qsa_cached(module, data, Recorder(report, root / "batch", phase))
        qsa_cached(module, data, Recorder(report, root / "single", phase))
        qsa_cached(module, data, Recorder(report, root / "single", phase + "-bc"), graph=True)
        qsa_companding(module, data, report, phase)
        report.add("qsa.reduction", phase + "/runtime", "PASS", "all isolated forwards completed with backend.all_reduce instrumented to raise")
    else:
        # Lifetime phase happens first: no gather has occurred on imported tables yet.
        require("consumer" not in context, "consumer still alive before post-parent-close gather")
        if spec.get("fixture"):
            ple_fixture_scenario(module, data, Recorder(report, root / "fixture", phase))
        else:
            ple_scenario(module, data, Recorder(report, root / "state", phase), context)
            report.check("ple.graph", phase + "/boundary", lambda: ple_graph_boundary(module, data, report, phase))
        report.add("ple.lifetime", phase, "PASS", "first worker gather and subsequent outputs completed after parent unload, collection close, arena overwrite/unlink and consumer close",
                   table_policy="shared_host_readonly" if module.ple_embedding.tables is not None else "worker_readonly_handles")
        ple_prefetch(module, data, report, phase)
    # Peaks include load and all scenario shapes, with the reference unloaded first.
    sample = memory_sample(module.device)
    estimate = spec["estimate"][torch.device(module.device).index]
    report.add("memory", phase, "PASS" if sample["peak_allocated"] <= estimate else "FAIL",
               f"peak allocated {sample['peak_allocated']} bytes; planner {estimate} bytes (reserved recorded separately)",
               estimate=estimate, **sample)


def worker_main(conn, kind, rank, producer, exported, plan, expected, spec, output):
    """Spawn target. Exceptions always send a traceback; parent has a finite deadline."""
    report = Report(kind, output)
    module = context = None
    global HOST_MONITOR
    HOST_MONITOR = HostMonitor()
    try:
        import torch
        torch.manual_seed(1234)
        torch.cuda.set_device(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        with torch.inference_mode():
            module, context = import_replica(kind, rank, producer, exported, plan)
            info = validate_replica(module, expected, kind, rank)
            conn.send(("ready", info))
            saved = target = None
            while True:
                op = conn.recv()
                if op == "mutate":
                    target = live_cuda_weight(module)
                    saved = target.clone()
                    target.reshape(-1)[0].add_(1)
                    assert module_fingerprint(module) != expected, "independence probe failed to mutate live module weights"
                    conn.send(("mutated", tensor_digest(target)))
                elif op == "restore":
                    target.copy_(saved)
                    saved = target = None
                    conn.send(("restored", None))
                elif op == "fingerprint":
                    conn.send(("fingerprint", module_fingerprint(module)))
                elif op == "close-consumer":
                    close_consumer(context)
                    conn.send(("closed", None))
                elif op == "run":
                    spec["exported"] = exported
                    exercise_replica(module, context, spec, report)
                    if kind == "ple":
                        data = torch.load(spec["input"], map_location="cpu", weights_only=True)
                        module, context = ple_reload_cycles(module, context, data, report, spec["tag"])
                    # Drain pending prefetch and prove all worker-owned disk handles close.
                    handles = list(getattr(getattr(module, "ple_embedding", None), "handles", None) or [])
                    module.unload()
                    if kind == "ple":
                        assert module.ple_embedding._executor is None and not module.ple_embedding._pending
                        assert all(h.fd is None for h in handles), "worker table fd leaked after unload"
                        report.add("ple.resources", spec["tag"] + "/unload", "PASS", "pending work drained; all owned table file descriptors closed",
                                   after=memory_sample(rank))
                    conn.send(("done", report.rows))
                    break
                else:
                    raise RuntimeError(f"unknown worker command: {op}")
    except BaseException as e:
        status = "NOT-RUN" if isinstance(e, NotRun) else "FAIL"
        report.add("transport", "worker-exception", status, f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        try:
            conn.send(("error", report.rows))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if context is not None and "consumer" in context:
            close_consumer(context)
        report.flush()
        conn.close()
        HOST_MONITOR.close()


def receive(conn, process, timeout, expected):
    if not conn.poll(timeout):
        raise RuntimeError(f"worker deadline ({timeout}s) expired awaiting {expected}; pid={process.pid}, exit={process.exitcode}")
    try:
        kind, value = conn.recv()
    except EOFError as e:
        raise RuntimeError(f"worker exited before {expected}; pid={process.pid}, exit={process.exitcode}") from e
    if kind != expected:
        if kind == "error" and not any(r["status"] == "FAIL" for r in value):
            raise NotRun(f"worker unable to run {expected}: {value}")
        raise RuntimeError(f"worker returned {kind} while awaiting {expected}: {value}")
    return value


def merge_rows(report, rows, prefix):
    for row in rows:
        item = dict(row, id=f"{prefix}/{row['id']}")
        assert item["id"] not in report.ids
        report.ids.add(item["id"])
        report.rows.append(item)
    report.flush()


def worker_rows(path):
    journal = Path(path).with_name("checks.jsonl")
    if journal.exists():
        try:
            return [json.loads(line) for line in journal.read_text().splitlines() if line]
        except json.JSONDecodeError as exc:
            raise NotRun(f"worker journal truncated/corrupt: {journal}: {exc}") from exc
    return json.loads(Path(path).read_text())["checks"]


def transport_round(module, args, report, data_path, reference, tag, collection, fixture=False):
    import torch
    from exllamav3.model.model_tp_shared import SMProducer
    from exllamav3.model.model_tp_alloc import TPAllocator
    from exllamav3.constants import PAGE_SIZE
    kind = args.command
    expected = module_fingerprint(module)
    atomic_json(Path(reference) / "loaded-module.json", expected)
    count = 2 * qsa_geometry(module.qsa_indexer, PAGE_SIZE)[1] if kind == "qsa" else 2 * 320
    allocator = TPAllocator(module.make_tp_allocation({}), count, count, dev_limits={"attn": 1})
    estimates = [int(v) for v in allocator.initial_split([torch.cuda.get_device_properties(i).total_memory for i in range(2)])[0]]
    plans = [("allocator-limit1", allocator.compile_tp_plan())]
    if kind == "qsa":
        plans += [("empty-rank1", [{module.key: (0, module.num_kv_heads, "heads")},
                                   {module.key: (module.num_kv_heads, module.num_kv_heads, "heads")}]),
                  ("no-attention-key", [{}, {}])]
    else:
        plans = [("full-replicas", [{module.key_proj.key: (0, 0, "channels"), module.value_proj.key: (0, 0, "channels")} for _ in range(2)])]
    producer = SMProducer()
    producer_open = True
    parent_loaded = True
    try:
        if kind == "ple":
            data = torch.load(data_path, map_location="cpu", weights_only=True)
            module.prefetch(data["ids"][:, :257], {"position": 0})
            require(bool(module.ple_embedding._pending), "parent prefetch missing before export")
        exported = module.tp_export({}, producer)  # Exactly one export per round, shared by both ranks/all QSA plans.
        companding_probe = None
        if kind == "qsa":
            from exllamav3.cache.qsa import CacheLayer_qsa_quant
            probe = CacheLayer_qsa_quant(None, module, 303, module.cache_layers[0].max_num_tokens,
                                        k_bits=args.k_bits, v_bits=args.v_bits, compand_a=0.25)
            companding_probe = probe.tp_export({})
        for plan_name, plans_for_ranks in plans:
            processes, pipes = [], []
            pseudo = pseudo_context = None
            prefix = f"{tag}/{plan_name}"
            try:
                ranks = (0, 1) if kind == "qsa" else (1,)
                for rank in ranks:
                    parent_conn, child_conn = mp.get_context("spawn").Pipe()
                    spec = dict(kind=kind, input=str(data_path), reference=str(reference), k_bits=args.k_bits, v_bits=args.v_bits,
                                tag=f"rank{rank}-{plan_name}", fixture=fixture, estimate=estimates, companding_probe=companding_probe)
                    proc = mp.get_context("spawn").Process(target=worker_main,
                        args=(child_conn, kind, rank, producer.export(), exported, plans_for_ranks[rank], expected,
                              spec, str(report.directory / "workers" / tag / plan_name / f"rank{rank}")))
                    proc.start()
                    child_conn.close()
                    processes.append(proc)
                    pipes.append(parent_conn)
                if kind == "ple":
                    # The output pseudo-worker is in the controller process, as in production.
                    pseudo, pseudo_context = import_replica(kind, 0, producer, exported, plans_for_ranks[0])
                    report.check("transport", prefix + "/pseudo0", lambda: validate_replica(pseudo, expected, kind, 0))
                for rank, conn, proc in zip(ranks, pipes, processes):
                    info = receive(conn, proc, args.timeout, "ready")
                    report.add("transport", prefix + f"/rank{rank}", "PASS", "real spawned consumer imported exact full weights/config and both caches", **info)
                # Independence is established by a mutation, not comparison of virtual pointers.
                if kind == "qsa":
                    pipes[0].send("mutate")
                    receive(pipes[0], processes[0], args.timeout, "mutated")
                    pipes[1].send("fingerprint")
                    other = receive(pipes[1], processes[1], args.timeout, "fingerprint")
                    assert other == expected, "rank0 write changed rank1 weights"
                    if parent_loaded:
                        assert module_fingerprint(module) == expected, "worker write changed original weights"
                    pipes[0].send("restore")
                    receive(pipes[0], processes[0], args.timeout, "restored")
                else:
                    t = live_cuda_weight(pseudo)
                    saved = t.clone()
                    t.reshape(-1)[0].add_(1)
                    assert module_fingerprint(pseudo) != expected, "pseudo-worker probe did not mutate live weights"
                    pipes[0].send("fingerprint")
                    assert receive(pipes[0], processes[0], args.timeout, "fingerprint") == expected
                    assert module_fingerprint(module) == expected
                    t.copy_(saved)
                    del t, saved
                report.add("transport", prefix + "/independence", "PASS", "mutating rank0 CUDA weights left rank1 and original unchanged")
                if parent_loaded:
                    module.unload()
                    collection.close()
                    parent_loaded = False
                    gc.collect()
                    torch.cuda.empty_cache()
                if kind == "ple":
                    # Reuse the actual arena before close: dangling aux views must not survive.
                    producer.clear()
                    producer.buf[:] = 0xA5
                    for conn in pipes:
                        conn.send("close-consumer")
                    for conn, proc in zip(pipes, processes):
                        receive(conn, proc, args.timeout, "closed")
                    close_consumer(pseudo_context)
                    exported = None  # release parent persistent-table descriptors as well
                    producer.close()
                    producer_open = False
                    del producer
                    gc.collect()
                for conn in pipes:
                    conn.send("run")
                if kind == "ple":
                    # The pseudo-worker's peaks must exclude the now-unloaded reference.
                    torch.cuda.reset_peak_memory_stats(0)
                    local_report = Report(kind, report.directory / "workers" / tag / "pseudo0")
                    local_spec = dict(kind=kind, input=str(data_path), reference=str(reference),
                                      tag="pseudo0", fixture=fixture, estimate=estimates)
                    exercise_replica(pseudo, pseudo_context, local_spec, local_report)
                    data = torch.load(data_path, map_location="cpu", weights_only=True)
                    pseudo, pseudo_context = ple_reload_cycles(pseudo, pseudo_context, data, local_report, "pseudo0")
                    handles = list(pseudo.ple_embedding.handles or [])
                    pseudo.unload()
                    assert not pseudo.ple_embedding._pending and pseudo.ple_embedding._executor is None
                    assert all(h.fd is None for h in handles)
                    local_report.add("ple.resources", "pseudo-unload", "PASS", "pseudo-worker pending work drained and handles closed", after=memory_sample(0))
                    merge_rows(report, local_report.rows, prefix + "/pseudo0")
                for rank, conn, proc in zip(ranks, pipes, processes):
                    rows = receive(conn, proc, args.timeout, "done")
                    merge_rows(report, rows, prefix + f"/rank{rank}")
                    proc.join(30)
                    assert proc.exitcode == 0, f"worker {rank} did not exit cleanly: {proc.exitcode}"
            finally:
                for conn in pipes:
                    conn.close()
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(10)
                # Preserve detailed findings even if a worker failed, died or timed out.
                known = {row["id"] for row in report.rows}
                for rank in ranks:
                    child_report = report.directory / "workers" / tag / plan_name / f"rank{rank}" / "report.json"
                    if child_report.is_file():
                        rows = worker_rows(child_report)
                        merge_rows(report, [r for r in rows if not r["id"].endswith("/pending")
                                           and f"{prefix}/rank{rank}/{r['id']}" not in known], prefix + f"/rank{rank}")
                if kind == "ple":
                    local_path = report.directory / "workers" / tag / "pseudo0" / "report.json"
                    if local_path.is_file():
                        rows = worker_rows(local_path)
                        merge_rows(report, [r for r in rows if not r["id"].endswith("/pending")
                                           and f"{prefix}/pseudo0/{r['id']}" not in known], prefix + "/pseudo0")
                if pseudo_context is not None and "consumer" in pseudo_context:
                    close_consumer(pseudo_context)
                if pseudo is not None:
                    pseudo.unload()
    finally:
        if parent_loaded:
            module.unload()
            collection.close()
        if producer_open:
            producer.close()


def load_one(args, key, cache_mode, no_reconstruct, report, tag):
    import torch
    from exllamav3 import Config, Model
    from exllamav3.cache.qsa import CacheLayer_qsa, CacheLayer_qsa_quant
    from exllamav3.modules.ple import PLELayerState
    from exllamav3.constants import PAGE_SIZE
    config = Config.from_directory(args.model)
    config.infer_params.no_reconstruct = no_reconstruct
    model = Model.from_config(config)
    require(model.caps.get("supports_tp") is False, "qwen4_exp loader capability gate must remain closed")
    module = model.find_module(key)
    if args.command == "qsa":
        threshold, length, per_row = qsa_geometry(module.qsa_indexer, PAGE_SIZE)
        for cid in (101, 202):
            cls = CacheLayer_qsa if cache_mode == "fp16" else CacheLayer_qsa_quant
            kwargs = {} if cache_mode == "fp16" else dict(k_bits=args.k_bits, v_bits=args.v_bits, compand_a=args.compand_a)
            module.cache_layers.append(cls(config, module, cid, 2 * per_row, **kwargs))
        report.environment.setdefault("shapes", {})[tag] = dict(T=threshold, L=length, page_size=PAGE_SIZE,
            cache_capacity_each=2 * per_row, per_sequence_capacity=per_row, compress_ratio=module.qsa_indexer.compress_ratio,
            score_tile=module.qsa_indexer.SEL_TILE, input=[2, length, module.hidden_size])
    else:
        for cid in (101, 202):
            module.recurrent_layers.append(PLELayerState(module, max_batch_size=8, max_history=8, cache_id=cid))
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    module.load(torch.device("cuda", 0))
    torch.cuda.synchronize()
    return module, config.stc


def make_inputs(module, args, path):
    import torch
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    generator = torch.Generator(device="cpu").manual_seed(1234)
    if args.command == "qsa":
        _, length, _ = qsa_geometry(module.qsa_indexer, PAGE_SIZE)
        data = (torch.randn((2, length, module.hidden_size), generator=generator) * 0.1).half()
    else:
        data = {"streams": torch.randn((2, 320, module.hc_mult, module.hidden_size), generator=generator) * 0.1,
                "ids": torch.randint(1, max(2, module.config.vocab_size), (2, 320), generator=generator, dtype=torch.long)}
        data["ids"][:, 248:255] = 42
        for pos in (0, module.ple_embedding.context_len - 1, 255, 256, 257, 266, 280):
            data["ids"][:, pos] = module.ple_embedding.eos_token_id
        if module.mm_token_id is not None:
            data["ids"][:, 260] = FIRST_MM_EMBEDDING_INDEX + 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def run_reference(module, args, report, data, root, fixture=False):
    for phase, write in (("reference", True), ("repeat", False)):
        if args.command == "qsa":
            qsa_nc(module, data, Recorder(report, root / "nc", phase, write))
            qsa_cached(module, data, Recorder(report, root / "batch", phase, write))
            qsa_cached(module, data, Recorder(report, root / "single", phase, write))
        elif fixture:
            ple_fixture_scenario(module, data, Recorder(report, root / "fixture", phase, write))
        else:
            ple_scenario(module, data, Recorder(report, root / "state", phase, write))
    report.flush()


def run_gpu(args, report):
    import torch
    from exllamav3 import Config, Model
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.ple import PLELayer
    torch.manual_seed(1234)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    require(model.caps.get("supports_tp") is False, "loader gate unexpectedly open")
    selected = ([m.key for m in model if isinstance(m, Attention) and m.qsa_indexer is not None]
                if args.command == "qsa" else [m.key for m in model if isinstance(m, PLELayer)])
    available = list(selected)
    if args.module:
        selected = [key for key in selected if key == args.module]
    require(bool(selected), f"no requested {args.command} module in checkpoint; available: {available}")
    report.environment.update(selected_modules=selected, available_modules=available,
                              qualification_scope="all modules" if selected == available else "selected module only")
    config.stc.close()
    del config, model
    with torch.inference_mode():
        for ordinal, key in enumerate(selected):
            modes = ["fp16", "served-kv"] if args.command == "qsa" else ["served-table"]
            for no_reconstruct in sorted(set([False, args.no_reconstruct])):
                for mode in modes:
                    tag = f"module{ordinal}-{mode}-nr{int(no_reconstruct)}"
                    root = report.directory / "artifacts" / tag
                    root.mkdir(parents=True, exist_ok=True)
                    module = collection = None
                    try:
                        module, collection = load_one(args, key, mode, no_reconstruct, report, tag)
                        path = root / "input.pt"
                        data = make_inputs(module, args, path)
                        report.environment.setdefault("inputs", {})[tag] = dict(path=str(path), sha256=sha256(path))
                        if args.command == "ple":
                            report.environment.setdefault("served_tables", {})[tag] = dict(mode=module.ple_embedding.mode,
                                K=module.ple_embedding.K, dtype=str(module.ple_embedding._row_dtype), rows=module.ple_embedding.num_rows)
                        run_reference(module, args, report, data, root)
                        transport_round(module, args, report, path, root, tag, collection)
                        report.add("transport", tag + "/completed", "PASS", f"all transport plans completed for {key}")
                    except Exception as exc:
                        report.add("transport", tag + "/aborted", "NOT-RUN" if isinstance(exc, NotRun) else "FAIL",
                                   f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                    finally:
                        if module is not None:
                            module.unload()
                        if collection is not None:
                            collection.close()
                        del module, collection
                        gc.collect()
                        torch.cuda.empty_cache()

            if args.command == "ple":
                # Format fixtures are independent of the served-table case and do not
                # substitute for it. Both use this module's real served projections.
                for i, case in enumerate(table_fixture_cases()):
                    tag = f"module{ordinal}-fixture{i:03d}"
                    root = report.directory / "artifacts" / tag
                    root.mkdir(parents=True, exist_ok=True)
                    module = collection = fixture_collection = None
                    try:
                        module, collection = load_one(args, key, "served-table", args.no_reconstruct, report, tag)
                        fixture_collection, probes = write_table_fixture(module, case, root / "tables")
                        path = root / "input.pt"
                        data = make_inputs(module, args, path)
                        data["probes"] = probes
                        torch.save(data, path)
                        atomic_json(root / "fixture.json", case)
                        report.environment.setdefault("fixtures", {})[tag] = dict(case=case, input_sha256=sha256(path),
                            table_sha256={p.name: sha256(p) for p in (root / "tables").glob("*.safetensors")})
                        run_reference(module, args, report, data, root, fixture=True)
                        collection.close()
                        transport_round(module, args, report, path, root, tag, fixture_collection, fixture=True)
                    except Exception as exc:
                        report.add("ple.tables", tag + "/aborted", "NOT-RUN" if isinstance(exc, NotRun) else "FAIL",
                                   f"{type(exc).__name__}: {exc}", case=case, traceback=traceback.format_exc())
                    finally:
                        if module is not None:
                            module.unload()
                        for coll in (collection, fixture_collection):
                            if coll is not None:
                                coll.close()
                        del module, collection, fixture_collection
                        gc.collect()
                        torch.cuda.empty_cache()


def selftest_spawn_target(conn):
    # Standard-library-only target exercising the same spawn/pipe/deadline code.
    conn.send(("ready", {"torch_imported": "torch" in sys.modules}))
    conn.recv()
    conn.close()


class HarnessSelftest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_arguments(self):
        for kind in ("qsa", "ple"):
            args = parser().parse_args([kind, "--output", str(self.root), "--module", "exact.key"])
            self.assertEqual((args.command, args.k_bits, args.v_bits, args.compand_a), (kind, 8, 8, 0.0))
            self.assertEqual(args.module, "exact.key")
        self.assertEqual(parser().parse_args(["selftest"]).command, "selftest")
        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            for argv in (["qsa"], ["ple", "--output", "x", "--k-bits", "1"], ["bogus"], ["qsa", "--out", "x"]):
                with self.assertRaises(SystemExit):
                    parser().parse_args(argv)

    def test_report_fail_closed_and_schema(self):
        report = Report("qsa", self.root)
        self.assertEqual(report.exit_code(), 2)
        for family in families("qsa"):
            report.add(family, "test", "PASS", "selftest synthetic result")
        self.assertEqual(report.exit_code(), 0)
        doc = report.document()
        validate_report(doc)
        for field in ("consequence", "tolerance", "reason"):
            broken = copy.deepcopy(doc)
            del broken["checks"][0][field]
            with self.assertRaises(ValueError):
                validate_report(broken)
        broken = copy.deepcopy(doc)
        broken["checks"][0]["tolerance"]["atol"] = float("nan")
        with self.assertRaises(ValueError):
            validate_report(broken)
        broken = copy.deepcopy(doc)
        broken["checks"].pop()
        with self.assertRaises(ValueError):
            validate_report(broken)
        broken = copy.deepcopy(doc)
        broken["checks"].append(broken["checks"][0])
        with self.assertRaises(ValueError):
            validate_report(broken)
        report.add("numerics", "failure", "FAIL", "synthetic numerical failure", TOL)
        report.add("qsa.graph", "blocked", "NOT-RUN", "synthetic missing capability")
        self.assertEqual(report.exit_code(), 1)
        validate_report(json.loads((self.root / "report.json").read_text()))
        self.assertIn("Consequence", (self.root / "summary.txt").read_text())

    def test_exception_classification(self):
        report = Report("ple", self.root)
        def absent():
            require(False, "missing symbol: foo")
        def bad():
            raise AssertionError("bad numerical output")
        self.assertEqual(report.check("preconditions", "absent", absent)["status"], "NOT-RUN")
        self.assertEqual(report.check("numerics", "bad", bad)["status"], "FAIL")

    def test_journal_retains_unflushed_worker_results(self):
        report = Report("qsa", self.root)
        report.add("numerics", "one", "PASS", "synthetic worker checkpoint")
        rows = worker_rows(self.root / "report.json")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "numerics/one")
        # The periodic snapshot is still incomplete and can never be green.
        self.assertEqual(json.loads((self.root / "report.json").read_text())["verdict"], "NOT-RUN")
        report.add("numerics", "two", "FAIL", "synthetic worker mismatch")
        self.assertEqual(json.loads((self.root / "report.json").read_text())["verdict"], "FAIL")

    def test_checkpoint_preconditions(self):
        with self.assertRaisesRegex(NotRun, "directory missing"):
            checkpoint_files(self.root / "missing")
        with self.assertRaisesRegex(NotRun, "config missing"):
            checkpoint_files(self.root)
        (self.root / "config.json").write_text("{}")
        with self.assertRaisesRegex(NotRun, "no .safetensors"):
            checkpoint_files(self.root)
        (self.root / "one.safetensors").write_bytes(b"fixture-not-real-weights")
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"x": "absent.safetensors"}}))
        with self.assertRaisesRegex(NotRun, "indexed checkpoint shard missing"):
            checkpoint_files(self.root)
        index.write_text(json.dumps({"weight_map": {"x": "one.safetensors"}}))
        self.assertEqual(len(checkpoint_files(self.root)), 3)
        self.assertEqual(len(sha256(self.root / "one.safetensors")), 64)

    def test_runtime_missing_modules_without_importing_torch(self):
        from unittest.mock import patch
        (self.root / "config.json").write_text("{}")
        (self.root / "one.safetensors").write_bytes(b"precondition-fixture")
        serving = self.root / "serving.yml"
        serving.write_text("model: {}\n")
        args = parser().parse_args(["qsa", "--model", str(self.root), "--serving-config", str(serving),
                                    "--output", str(self.root / "report")])
        report = Report("qsa", self.root / "report")
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(NotRun, "missing Python module.*torch.*exllamav3_ext"):
                runtime_preconditions(args, report)
        self.assertNotIn("torch", sys.modules)

    def test_source_preconditions_name_missing_module_and_symbol(self):
        with self.assertRaisesRegex(NotRun, "missing module:.*qsa_indexer"):
            source_preconditions(self.root, "qsa")
        package = Path(__file__).parent / "exllamav3"
        # Real baseline has no transport overrides; patch a temporary tree only.
        if package.is_dir():
            shutil.copytree(package, self.root / "package", ignore=shutil.ignore_patterns("__pycache__", "exllamav3_ext"))
            package = self.root / "package"
            patches = Path(__file__).with_name("ep-accept-patches")
            for patch in sorted(patches.glob("*.patch")):
                apply_exact_patch(package, patch)
                apply_exact_patch(package, patch)  # real patches must be idempotent too
            source_preconditions(package, "qsa")
            source_preconditions(package, "ple")
            ple = package / "modules/ple.py"
            ple.write_text(ple.read_text().replace("def tp_import(", "def absent_import("))
            with self.assertRaisesRegex(NotRun, "PLELayer.tp_import"):
                source_preconditions(package, "ple")
        else:
            # Installed-host selftest still tests missing method diagnostics, without torch.
            for rel, cls in (("modules/qsa_indexer.py", "QSAIndexer"), ("modules/attn.py", "Attention"),
                             ("model/model_tp_shared.py", "SMProducer")):
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"class {cls}:\n    pass\n")
            with self.assertRaisesRegex(NotRun, "QSAIndexer.tp_export"):
                source_preconditions(self.root, "qsa")

    def test_patch_exact_idempotent_and_rejects_drift(self):
        module = self.root / "a.py"
        module.write_text("# offset\nx = 1\ny = 2\n")
        patch = self.root / "a.patch"
        patch.write_text("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 3\n y = 2\n")
        apply_exact_patch(self.root, patch)
        self.assertEqual(module.read_text(), "# offset\nx = 3\ny = 2\n")
        apply_exact_patch(self.root, patch)
        module.write_text("x = 1\ny = 4\n")
        with self.assertRaisesRegex(NotRun, "context absent"):
            apply_exact_patch(self.root, patch)
        self.assertEqual(module.read_text(), "x = 1\ny = 4\n")

    def test_geometry_schedule_and_matrix(self):
        class Indexer:
            SEL_TILE, compress_ratio = 8192, 4
            def sparse_threshold(self):
                return 4099
        t, length, cap = qsa_geometry(Indexer(), 256)
        self.assertEqual(length, max(t + 261, 8194 * 4 + 5))
        self.assertGreaterEqual(cap, length + 256)
        self.assertEqual(cap % 256, 0)
        for start in (0, 7):
            schedule = qsa_schedule(length, t, 256, 8192 * 4, start)
            self.assertEqual(sum(n for _, n in schedule), length - start)
            self.assertTrue(any(n == 257 for _, n in schedule))
            self.assertIn((t + 4, 4), schedule)
            for boundary in (t, 256, 8192 * 4):
                for pos in (boundary - 1, boundary, boundary + 1):
                    self.assertIn((pos, 1), schedule)
        cases = table_fixture_cases()
        self.assertEqual(len(cases), 108)
        self.assertEqual({c["K"] for c in cases if c["format"] == "trellis"}, set(range(1, 9)))
        self.assertEqual({c["dtype"] for c in cases if c["format"] == "raw"}, {"float16", "bfloat16"})

    def test_missing_checkpoint_cli_writes_not_run(self):
        for kind in ("qsa", "ple"):
            output = self.root / kind
            r = subprocess.run([sys.executable, __file__, kind, "--model", str(self.root / "missing"),
                                "--output", str(output)], text=True, capture_output=True)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            doc = json.loads((output / "report.json").read_text())
            validate_report(doc)
            self.assertEqual(doc["verdict"], "NOT-RUN")
            self.assertIn("checkpoint directory missing", doc["checks"][0]["reason"])

    def test_no_gpu_imports_in_selftest_process(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("exllamav3", sys.modules)

    def test_spawn_and_timeout_without_gpu(self):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=selftest_spawn_target, args=(child,))
        proc.start()
        child.close()
        try:
            self.assertFalse(receive(parent, proc, 10, "ready")["torch_imported"])
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                receive(parent, proc, 0.02, "done")
            parent.send("finish")
            proc.join(10)
            self.assertEqual(proc.exitcode, 0)
        finally:
            parent.close()
            if proc.is_alive():
                proc.terminate()
                proc.join(5)

    def test_external_not_run_and_no_overwrite(self):
        output = self.root / "report"
        command = [sys.executable, __file__, "qsa", "--output", str(output), "--not-run-reason", "missing local served image"]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        before = (output / "report.json").read_bytes()
        self.assertIn(b"missing local served image", before)
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual((output / "report.json").read_bytes(), before)

    def test_runner_queue_contract_and_syntax(self):
        runner = Path(__file__).with_name("run-ep-accept.sh")
        require(runner.is_file(), f"selftest requires runner alongside harness: {runner}")
        source = runner.read_text()
        self.assertIn("source /srv/qwen5090/lib/gpu-queue.sh", source)
        self.assertRegex(source, r"(?m)^gpu_lock$")
        self.assertNotRegex(source, r"(?m)^\s*(exec\s+9>|flock\s)")
        self.assertIn('R="$ROOT/results/', source)
        self.assertIn("--pull=never", source)
        result = subprocess.run(["bash", "-n", str(runner)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "selftest":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(HarnessSelftest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.command == "stage":
        try:
            stage_package(args)
            return 0
        except Exception as exc:
            print(f"NOT-RUN: package staging: {exc}", file=sys.stderr)
            return 2
    output = Path(args.output)
    # Never overwrite prior evidence or compare with stale artifacts from another run.
    if (output / "report.json").exists():
        print(f"NOT-RUN: output already contains a report: {output}", file=sys.stderr)
        return 2
    report = Report(args.command, output)
    global HOST_MONITOR
    try:
        if args.not_run_reason:
            raise NotRun(args.not_run_reason)
        require(math.isfinite(args.compand_a) and args.compand_a >= 0, "--compand-a must be finite and nonnegative")
        require(args.timeout > 0, "--timeout must be positive")
        row = report.check("preconditions", "runtime", lambda: runtime_preconditions(args, report))
        if row["status"] != "PASS":
            return report.exit_code()
        if args.preflight:
            report.environment["preflight_only"] = True
            report.flush()
            return 0  # readiness only; the report remains NOT-RUN for GPU protocols
        row = report.check("provenance", "full-hashes", lambda: provenance(args, report))
        if row["status"] != "PASS":
            return report.exit_code()
        HOST_MONITOR = HostMonitor()
        run_gpu(args, report)
        for entry in report.environment["checkpoint_files"]:
            current = (Path(args.model) / entry["path"]).stat()
            require((current.st_size, current.st_mtime_ns) == (entry["bytes"], entry["mtime_ns"]),
                    f"checkpoint changed during run: {entry['path']}")
        report.add("completion", "matrix", "PASS", "every requested matrix entry reached an outcome; checkpoint sizes/mtimes unchanged")
    except (Exception, KeyboardInterrupt) as exc:
        report.add("preconditions", "aborted", "NOT-RUN" if isinstance(exc, (NotRun, KeyboardInterrupt)) else "FAIL",
                   f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if HOST_MONITOR is not None:
            HOST_MONITOR.close()
            report.environment["controller_host_peaks"] = dict(HOST_MONITOR.peaks)
        report.flush()
    print(f"{args.command.upper()}: {report.document()['verdict']} — {output / 'report.json'}", flush=True)
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
