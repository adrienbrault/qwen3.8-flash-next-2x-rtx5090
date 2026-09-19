#!/usr/bin/env python3
"""CPU-only checks for ngram-prefetch-r1 (pytest). CPU torch, no CUDA, no network.

The served source defaults to ../../src/exllamav3 (this repo's pristine copy of the served
image's exllamav3 tree, tabbyapi:nvme-tier-r4-e3det-r6-rawk); override with NGRAM_SERVED_SRC.
Tests that need it skip when it is absent.

Layers:
  1. patch + manifest pins (hashes, file list, `patch -p1 -F0` exit code, overlay hashes).
  2. flag-off identity: every served function the patch touches is textually unchanged except
     this round's deliberate extensions, and every new def carries an opt-in marker.
  3. runtime behaviour of the patched NGramEmbedding through tests/cpu_runtime.py (real package
     code, fake extension): runs under three flag combinations and asserts the records.
  4. generator wiring, source level: the verify prefetch sits between the draft readback and
     model.forward, gated on EXL3_NGRAM_PREFETCH2; the periodic report helper is flag-gated.

Run:  python3 -m pytest -q -p no:cacheprovider tests/test_ngram_prefetch_cpu.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
R1 = HERE.parent
SERVED = Path(os.environ.get("NGRAM_SERVED_SRC", R1.parent.parent / "src" / "exllamav3"))
PATCH = R1 / "served-source.patch"
MANIFEST = json.loads((R1 / "overlay/manifest.json").read_text())
FILES = sorted(MANIFEST["files"])

needs_served = pytest.mark.skipif(not SERVED.is_dir(), reason = f"served source not found at {SERVED}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_files(tmp_path: Path, files) -> Path:
    target = tmp_path / "exllamav3"
    for relative in files:
        (target / relative).parent.mkdir(parents = True, exist_ok = True)
        shutil.copy2(SERVED / relative, target / relative)
    return target


def copy_full_tree(tmp_path: Path) -> Path:
    """Full served tree (no __pycache__), so the package imports exactly as installed."""
    target = tmp_path / "exllamav3"
    shutil.copytree(SERVED, target, ignore = shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target


def apply(root: Path, patch: Path, *extra) -> subprocess.CompletedProcess:
    with patch.open("rb") as stream:
        return subprocess.run(["patch", "-p1", "-F0", "--no-backup-if-mismatch", *extra], cwd = root,
                              stdin = stream, capture_output = True)


def functions(path: Path) -> dict:
    """All defs/classes including nested ones (methods of classes, helpers inside methods)."""
    import ast as ast_mod
    src = path.read_text()
    tree = ast_mod.parse(src)
    out = {}
    for node in ast_mod.walk(tree):
        if isinstance(node, (ast_mod.FunctionDef, ast_mod.ClassDef)):
            out.setdefault(node.name, ast_mod.get_source_segment(src, node))
    return out


def top_level_names(path: Path) -> set:
    import ast as ast_mod
    tree = ast_mod.parse(path.read_text())
    return {n.name for n in tree.body
            if isinstance(n, (ast_mod.FunctionDef, ast_mod.ClassDef))}


# ---- patch + manifest -------------------------------------------------------------------------------------------

def test_patch_hash_pinned():
    assert sha256(PATCH) == MANIFEST["patch_sha256"]
    assert MANIFEST["round"] == "ngram-prefetch-r1"
    assert MANIFEST["base_image"] == "tabbyapi:nvme-tier-r4-e3det-r6-rawk"


def test_patch_touches_exactly_manifest_files():
    import re
    text = PATCH.read_text()
    assert sorted(re.findall(r"^\+\+\+ b/(\S+)", text, flags = re.M)) == FILES
    assert sorted(re.findall(r"^--- a/(\S+)", text, flags = re.M)) == FILES
    # Python only: no extension rebuild, no tier/C++ change
    assert not any(p.endswith((".cu", ".cpp", ".h", ".cuh")) for p in FILES)


@needs_served
def test_served_baseline_hashes():
    for relative, record in MANIFEST["files"].items():
        assert sha256(SERVED / relative) == record["baseline_sha256"], relative


@needs_served
def test_patch_applies_fuzz0_and_overlay_hashes(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    proc = apply(root, PATCH)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"], relative
    # a second application must fail
    again = apply(root, PATCH, "--forward", "--dry-run")
    assert again.returncode != 0


@needs_served
def test_install_py_on_scratch_root(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    overlay = R1 / "overlay"
    proc = subprocess.run([sys.executable, overlay / "install.py", "--root", root,
                           "--manifest", overlay / "manifest.json", "--patch", PATCH],
                          capture_output = True, text = True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@needs_served
def test_install_py_rejects_drifted_baseline(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    (root / FILES[0]).write_text((root / FILES[0]).read_text() + "\n# drift\n")
    overlay = R1 / "overlay"
    proc = subprocess.run([sys.executable, overlay / "install.py", "--root", root,
                           "--manifest", overlay / "manifest.json", "--patch", PATCH],
                          capture_output = True, text = True)
    assert proc.returncode != 0 and "baseline hash mismatch" in proc.stdout + proc.stderr


# ---- flag-off identity ------------------------------------------------------------------------------------------

@needs_served
def test_flag_off_functions_unchanged():
    """Every function/class of the patched files is byte-identical to the served source, except
    the ones this round deliberately extends; every new def carries an opt-in marker."""
    allowed_changed = {
        # class bodies gained this round's methods; the per-function check below still pins
        # every individual served method, so only the container is allowlisted here
        "modules/ngram_embedding.py": {"_acquire_pin", "_retire", "_drain_prefetch",
                                       "NGramEmbedding", "_PinSet", "prefetch", "forward"},
        "modules/ple.py": {"PLELayer"},
        "generator/generator.py": {"Generator", "__init__", "iterate", "iterate_gen"},
        "generator/prefill_pipeline.py": {"_Runtime", "_stage0"},
    }
    with tempfile.TemporaryDirectory() as td:
        served = copy_files(Path(td) / "served", FILES)
        patched = copy_files(Path(td) / "patched", FILES)
        proc = apply(patched, PATCH)
        assert proc.returncode == 0
        for relative in FILES:
            served_fns = functions(served / relative)
            patched_fns = functions(patched / relative)
            for name, src in served_fns.items():
                if name in patched_fns and name not in allowed_changed.get(relative, set()):
                    assert patched_fns[name] == src, f"{relative}: {name} changed"
            patched_top = top_level_names(patched / relative)
            for name, psrc in patched_fns.items():
                if name not in served_fns and name in patched_top:
                    assert ("PREFETCH2" in psrc or "NGRAM_TIMING" in psrc or "_TIMING" in psrc), \
                        f"{relative}: new top-level {name} lacks an opt-in marker"
            # the only served method allowed to differ: Generator.__init__ gains exactly the
            # timing clock line
            if relative == "modules/ngram_embedding.py" and "forward" in served_fns:
                # forward() must keep the served miss path verbatim inside the timing wrap,
                # and the served upload/dequant tail untouched
                a = served_fns["forward"]
                b = patched_fns["forward"]
                for served_line in (
                        "packed_d = pin.packed[:U].to(dev, non_blocking = True)",
                        "inv_d = pin.inverse[:n].to(dev, non_blocking = True)",
                        "out = rows.index_select(0, inv_d).view(bsz, out_len, H * ROW_DIM)"):
                    assert served_line in b, f"forward: served line changed: {served_line}"
                assert 'prefetch_stats["miss"] += 1' in b and "self._acquire_pin(n, row_words, row_dtype)" in b
                assert "self._timed(" in b
            if relative == "modules/ngram_embedding.py" and "prefetch" in served_fns:
                # flag-off path of prefetch(): the served statements survive verbatim inside
                # the two `if not PREFETCH2:` guards
                assert "if not PREFETCH2:\n            pin = self._acquire_pin(" in patched_fns["prefetch"], \
                    "prefetch: flag-off pin acquisition is not the served statement"
                served_append = (
                    "if not PREFETCH2:\n"
                    "            self._pending.append({\n"
                    '                "history": history,\n'
                    '                "pin": pin,\n'
                    '                "future": self._executor.submit(self._stage, history, pin),\n'
                    "            })")
                assert served_append in patched_fns["prefetch"], \
                    "prefetch: flag-off queueing is not the served statement"
            if relative == "generator/prefill_pipeline.py" and "_stage0" in served_fns:
                a = served_fns["_stage0"]
                b = patched_fns["_stage0"]
                b2 = b.replace(
                    "            # EXL3_NGRAM_PREFETCH2 (timing only): with the flag on, NGramEmbedding.prefetch\n"
                    "            # submits the stage to the worker without acquiring the staging set on this\n"
                    "            # thread, so issuing A_(i+1) never waits on chunk i's cold gather or on its\n"
                    "            # forward's still-in-flight H2D uploads; the discard-on-mismatch contract keeps\n"
                    "            # results untouched.\n", "", 1)
                assert b2 == a, "_stage0 changed beyond this round's comment"
            if relative == "generator/generator.py" and "iterate_gen" in served_fns:
                a = served_fns["iterate_gen"]
                b = patched_fns["iterate_gen"]
                b2 = b.replace(
                    "        if _NGRAM_PREFETCH2:\n"
                    "            # The verify window's id history is complete on the host here: the draft tokens\n"
                    "            # were just read back into draft_ids_pinned. Hand it to the PLE worker now so the\n"
                    "            # gather overlaps the remaining host work between the draft chain and the verify\n"
                    "            # forward's PLE layer. Timing only: forward() takes the set whose staged history\n"
                    "            # matches and stages inline otherwise.\n"
                    "            for m in self.model._get_prefetch_layers:\n"
                    "                m.prefetch_ids(batch_ids, params)\n", "", 1)
                assert b2 == a, "Generator.iterate_gen changed beyond the flag-gated verify prefetch"
            if relative == "generator/generator.py" and "iterate" in served_fns:
                a = served_fns["iterate"]
                b = patched_fns["iterate"]
                b2 = b.replace(
                    "        # EXL3_NGRAM_TIMING: periodic PLE exposed-wait summary (instrument only)\n"
                    "        if _NGRAM_TIMING:\n"
                    "            self._ngram_timing_maybe_report()\n\n", "", 1)
                assert b2 == a, "Generator.iterate changed beyond the flag-gated report hook"
            if relative == "generator/generator.py":
                a, b = served_fns["__init__"], patched_fns["__init__"]
                assert b.replace("        self._ngram_timing_last = None      # EXL3_NGRAM_TIMING periodic report clock\n", "", 1) == a, \
                    "Generator.__init__ changed beyond the timing clock line"


# ---- runtime scenarios through the fake extension ---------------------------------------------------------------

def run_runtime(env_extra: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = copy_full_tree(Path(td))
        proc = apply(root, PATCH)
        assert proc.returncode == 0
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("EXL3_NGRAM"):
                env.pop(k)
        env.update(env_extra)
        out = Path(td) / "records.json"
        proc = subprocess.run(
            [sys.executable, HERE / "cpu_runtime.py", str(root.parent), str(out)],
            capture_output = True, text = True, env = env, timeout = 600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return json.loads(out.read_text())


@pytest.mark.parametrize("env,expect", [
    ({}, {"prefetch2": False}),
    ({"EXL3_NGRAM_TIMING": "1"}, {"prefetch2": False}),
    ({"EXL3_NGRAM_PREFETCH2": "1", "EXL3_NGRAM_TIMING": "1"}, {"prefetch2": True}),
    ({"EXL3_NGRAM_PREFETCH2": "1"}, {"prefetch2": True}),
])
def test_cpu_runtime_scenarios(env, expect):
    data = run_runtime(env)
    assert data["ok"] is True
    rec = data["records"]
    assert rec["flags"] == expect
    assert rec["forward_equals_reference"] == {"prefill": True, "decode": True}
    assert rec["prefetch_off_vs_on"]["equal"] is True
    assert rec["prefetch_off_vs_on"]["hit"] >= 1
    assert rec["discard_mismatch"]["equal"] is True
    assert rec["discard_mismatch"]["stale_evicted"] is True
    assert rec["discard_mismatch"]["h1_equal"] and rec["discard_mismatch"]["h2_equal"]
    assert rec["discard_mismatch"]["pins_bounded"], "staging memory must stay bounded (2 sets)"
    assert rec["verify_prefetch"]["equal_reference"] is True
    if expect["prefetch2"]:
        # the verify-window hook must be consumed by the next forward
        assert rec["verify_prefetch"]["hit"] >= 1
    assert rec["pressure"]["equal"] and rec["pressure"]["pins_bounded"]
    assert rec["timing_report"]["nonneg"] is True
    timing_on = env.get("EXL3_NGRAM_TIMING") == "1" or expect["prefetch2"]
    if timing_on:
        assert rec["timing_report"]["has_prefill"] or rec["timing_report"]["has_decode"]
    else:
        assert rec["flag_off_inert"] == {"timing_entries": 0, "timing_seen": 0}


# ---- generator wiring (source level) ----------------------------------------------------------------------------

@needs_served
def test_generator_wiring():
    with tempfile.TemporaryDirectory() as td:
        root = copy_files(Path(td), FILES)
        proc = apply(root, PATCH)
        assert proc.returncode == 0
        src = (root / "generator/generator.py").read_text()
        # flag constant defined from the env
        assert '_NGRAM_PREFETCH2 = os.environ.get("EXL3_NGRAM_PREFETCH2", "0") == "1"' in src
        # the verify prefetch call is flag-gated and sits after the verifier params and
        # before the model forward
        i_params = src.index("params.update(self.draft_model.draft_verifier_params)")
        i_call = src.index("m.prefetch_ids(batch_ids, params)")
        i_fwd = src.index("batch_logits = self.model.forward(")
        assert i_params < i_call < i_fwd
        # gated
        assert "if _NGRAM_PREFETCH2:" in src
        # the periodic report helper exists and is flag-gated
        assert "_NGRAM_TIMING = os.environ.get(" in src
        assert "_ngram_timing_maybe_report" in src


@needs_served
def test_ple_prefetch_ids_builds_state_history():
    """ple.prefetch_ids reads the carried context from the recurrent layer state, the same
    lookup forward() performs."""
    with tempfile.TemporaryDirectory() as td:
        root = copy_files(Path(td), FILES)
        proc = apply(root, PATCH)
        assert proc.returncode == 0
        src = (root / "modules/ple.py").read_text()
        assert "def prefetch_ids" in src
        assert "rsg[0].cache.get_recurrent_layer(layer_instance)" in src
        assert "id_state[s, :ctx]" in src
        assert "self.ple_embedding.prefetch_ids(history)" in src
