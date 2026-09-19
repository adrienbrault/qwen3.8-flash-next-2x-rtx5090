"""CPU-only checks for mtp-kv-window r1 (pytest). CPU torch, no Triton, no CUDA, no network.

Every behavioural test runs the tree's OWN code in a subprocess (mtp_window_scenarios.py): the
served tree (MTP_WINDOW_SERVED_SRC, default: the round's src/exllamav3) and the same tree after
`patch -p1 --fuzz=0` of served-source.patch. The functions exercised are the ones the server calls:
Cache.__init__ (+ CacheLayer_quant / QSA planes), Generator.__init__,
Generator.iterate_draftmodel_mtp_gen (the MTP drafting round), Generator.iterate_gen (verify + the
MTP accept prefill that repairs the draft cache with target states, + job completion),
Job.prefill (chunked prompt prefill with the draft prefill), PageTable.defrag, BCAttn.__init__ and
BCAttn.step. Only the compiled extension and triton are stubbed (tests/stubs.py).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
SERVED = Path(os.environ.get("MTP_WINDOW_SERVED_SRC", str(HERE.parents[2] / "src" / "exllamav3")))
PATCH = R1 / "served-source.patch"
MANIFEST = json.loads((R1 / "overlay" / "manifest.json").read_text())
FILES = sorted(MANIFEST["files"])
NEW_FILES = sorted(f for f, r in MANIFEST["files"].items() if r["baseline_sha256"] is None)
PAGE = 256
W_TEST = 1024          # generator scenarios: 5 pages per slot, 4 slots

needs_served = pytest.mark.skipif(not SERVED.is_dir(), reason = f"served source not found at {SERVED}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(dest: Path) -> Path:
    target = dest / "exllamav3"
    shutil.copytree(SERVED, target, ignore = shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target


def apply(root: Path, *extra) -> subprocess.CompletedProcess:
    with PATCH.open("rb") as stream:
        return subprocess.run(["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch", *extra],
                              cwd = root, stdin = stream, capture_output = True)


@pytest.fixture(scope = "module")
def trees(tmp_path_factory):
    if not SERVED.is_dir():
        pytest.skip(f"served source not found at {SERVED}")
    served = tmp_path_factory.mktemp("served")
    copy_tree(served)
    patched = tmp_path_factory.mktemp("patched")
    root = copy_tree(patched)
    proc = apply(root)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    return {"served": served, "patched": patched}


def run(tree: Path, scenario: str, window: int | None = None, **env_extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    if window is not None:
        env["EXL3_MTP_KV_WINDOW"] = str(window)
    env.update({k: str(v) for k, v in env_extra.items()})
    proc = subprocess.run([sys.executable, str(HERE / "mtp_window_scenarios.py"), str(tree), scenario],
                          env = env, capture_output = True, text = True, timeout = 900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def slot_row(slot: int, pages_per_slot: int, width: int) -> list[int]:
    """Expected window row, written independently of the patch."""
    return [slot * pages_per_slot + (0 if q == 0 else 1 + (q - 1) % (pages_per_slot - 1)) for q in range(width)]


# ---- patch, manifest, packaging ---------------------------------------------------------------------------------

def test_patch_and_manifest_pins():
    assert sha256(PATCH) == MANIFEST["patch_sha256"]
    assert MANIFEST["base_image"] == "tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16"
    assert MANIFEST["selectors"] == {"EXL3_MTP_KV_WINDOW": "cache/mtp_window.py"}
    assert NEW_FILES == ["cache/mtp_window.py"]


def test_patch_targets_are_manifest_files():
    text = PATCH.read_text()
    targets = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", text, flags = re.M)))
    assert targets == FILES
    created = sorted(re.findall(r"^--- /dev/null\n\+\+\+ b/(\S+)", text, flags = re.M))
    assert created == NEW_FILES
    assert not any(p.endswith((".cu", ".cuh", ".cpp", ".h")) for p in FILES)


@needs_served
def test_served_baselines():
    for relative, record in MANIFEST["files"].items():
        path = SERVED / relative
        if record["baseline_sha256"] is None:
            assert not path.exists(), relative
        else:
            assert sha256(path) == record["baseline_sha256"], relative


@needs_served
def test_patch_applies_fuzz0_by_exit_code(tmp_path: Path):
    root = copy_tree(tmp_path)
    leftovers = set(root.rglob("*.orig")) | set(root.rglob("*.rej"))    # the served image ships one .orig
    proc = apply(root)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    assert set(root.rglob("*.orig")) | set(root.rglob("*.rej")) == leftovers
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"], relative
    assert apply(root, "--forward", "--dry-run").returncode != 0      # a second application is refused
    assert apply(root, "-R").returncode == 0                          # and it reverses to the served bytes
    for relative, record in MANIFEST["files"].items():
        if record["baseline_sha256"] is None:
            assert not (root / relative).exists() or (root / relative).stat().st_size == 0
        else:
            assert sha256(root / relative) == record["baseline_sha256"], relative


def install(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(R1 / "overlay" / "install.py"), "--root", str(root)],
                          capture_output = True, text = True)


@needs_served
def test_install_py_on_scratch_root(tmp_path: Path):
    root = copy_tree(tmp_path)
    proc = install(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"], relative
    assert install(root).returncode != 0                               # never twice


@needs_served
def test_install_py_refuses_drift(tmp_path: Path):
    root = copy_tree(tmp_path)
    with (root / "generator/job.py").open("a") as f:
        f.write("\n# drift\n")
    proc = install(root)
    assert proc.returncode != 0 and "baseline hash mismatch" in proc.stdout + proc.stderr
    root2 = copy_tree(tmp_path / "second")
    (root2 / "cache/mtp_window.py").write_text("# someone else's file\n")
    proc = install(root2)
    assert proc.returncode != 0 and "cache/mtp_window.py" in proc.stdout + proc.stderr


def test_dockerfile_rules():
    text = (R1 / "Dockerfile.box").read_text()
    assert "<<" not in text                                            # legacy builder: no heredocs
    assert "ARG BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16" in text
    assert not re.search(r"^\s*ENV\s+EXL3_MTP_KV_WINDOW", text, flags = re.M)
    last = [l for l in text.splitlines() if l.strip()][-1]
    assert "import" in last and "assert" in last


# ---- flag parsing -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, ok", [("", 0), ("0", 0), ("4096", 4096), ("16384", 16384),
                                     ("100", None), ("-256", None), ("abc", None), ("256", None)])
def test_flag_parsing(trees, raw, ok):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import stubs; "
            "stubs.install(); from exllamav3.cache import mtp_window as w; print(w.MTP_KV_WINDOW)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    env["EXL3_MTP_KV_WINDOW"] = raw
    proc = subprocess.run([sys.executable, "-c", code, str(trees["patched"]), str(HERE)],
                          env = env, capture_output = True, text = True)
    if ok is None:
        assert proc.returncode != 0 and "EXL3_MTP_KV_WINDOW" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert int(proc.stdout.strip().splitlines()[-1]) == ok


# ---- flag unset: the patched tree behaves exactly like the served tree -------------------------------------------

IDENTITY = ["alloc", "alloc_small_window", "gen_init", "draft_round", "verify_round", "verify_two_rounds",
            "job_prefill", "defrag"]


@pytest.mark.parametrize("scenario", IDENTITY)
def test_flag_off_identical_to_served(trees, scenario):
    assert run(trees["patched"], scenario) == run(trees["served"], scenario)


def test_flag_off_bc_step_identical_to_served(trees):
    served = run(trees["served"], "bc_step")
    patched = run(trees["patched"], "bc_step")
    keys = [k for k in served if k.startswith("unwindowed_")]
    assert keys and {k: patched[k] for k in keys} == {k: served[k] for k in keys}
    assert run(trees["patched"], "bc_init")["qsa_scan_cap"] == 0


def test_flag_off_rawk_alloc_identical(trees):
    assert run(trees["patched"], "alloc", EXL3_QSA_RAWK_RING = 1) == run(trees["served"], "alloc", EXL3_QSA_RAWK_RING = 1)


# ---- allocation and the byte arithmetic of impl-status.md -------------------------------------------------------

def test_window_allocation_and_bytes(trees):
    off = run(trees["patched"], "alloc", EXL3_QSA_RAWK_RING = 1)
    pool = 966656
    layer_bytes = off["draft"]["layers"][0]["storage_size"]
    assert layer_bytes == pool * 1172                                 # 1,172 B per token and layer at 8,8 + ring
    assert all(l["storage_size"] == layer_bytes for l in off["main"]["layers"])
    expected = {4096: (17, 34816), 16384: (65, 133120)}
    for w, (pages, tokens) in expected.items():
        on = run(trees["patched"], "alloc", window = w, EXL3_QSA_RAWK_RING = 1)
        assert on["main"] == off["main"]                              # the main cache is untouched
        d = on["draft"]
        assert d["window"] == {"window_tokens": w, "slots": 8, "pages_per_slot": pages,
                               "scan_tokens": pages * PAGE, "pool_tokens": tokens}
        assert d["max_num_tokens"] == tokens
        layer = d["layers"][0]
        assert layer["qshape_k"][0] == tokens // PAGE and layer["pooled_shape"][0] == tokens // PAGE
        assert layer["raw_k_shape"][:2] == [tokens // PAGE, 20]         # the raw-key ring stacks
        assert layer["scan_tokens"] == pages * PAGE
        assert layer["storage_size"] == tokens * 1172
        freed = layer_bytes - layer["storage_size"]
        equal_bytes_pool = (13 * layer_bytes - layer["storage_size"]) // (12 * 1172)
        assert (freed, equal_bytes_pool // 16384 * 16384) == {
            4096: (1092116480, 1032192), 16384: (976904192, 1032192)}[w]


def test_draft_cache_mode_alternative_bytes(trees):
    """The zero-code alternative (draft_cache_mode Q6 / Q4 on the MTP layer), same pool."""
    base = run(trees["served"], "alloc", EXL3_QSA_RAWK_RING = 1)["draft"]["layers"][0]["storage_size"]
    for bits, per_token, pool in ((6, 916, 983040), (4, 660, 999424)):
        size = run(trees["served"], "alloc", EXL3_QSA_RAWK_RING = 1,
                   SCENARIO_DRAFT_BITS = bits)["draft"]["layers"][0]["storage_size"]
        assert size == 966656 * per_token
        assert (13 * base) // (12 * 1172 + per_token) // 16384 * 16384 == pool


def test_small_window_refused(trees):
    out = run(trees["patched"], "alloc_small_window", window = 1024)
    assert out["error"] and "sparse threshold" in out["error"]


# ---- the served call paths with the window on -------------------------------------------------------------------

def test_generator_init(trees):
    out = run(trees["patched"], "gen_init", window = W_TEST)
    c = out["clamped"]
    assert c["tiers"] == {"cpu_tier": ["main"], "nvme": ["main", None]}     # no draft pages in either tier
    assert c["state"]["max_batch_size"] == 4                           # checked AFTER the recurrent clamp
    assert c["state"]["leases"] == []
    assert "slots=4" in c["state"]["window"] and "pool_tokens=5120" in c["state"]["window"]
    assert "window slots but max_batch_size is 4" in out["too_few_slots"]["error"]
    served = run(trees["served"], "gen_init", window = W_TEST)         # the served tree ignores the flag
    assert served["clamped"]["tiers"] == {"cpu_tier": ["main", "draft"], "nvme": ["main", "draft"]}


def test_drafting_round_rows(trees):
    out = run(trees["patched"], "draft_round", window = W_TEST)
    calls = out["calls"]
    assert [c["kind"] for c in calls] == ["forward"] * 3
    assert [c["cache_seqlens"] for c in calls] == [[2900, 700], [2901, 701], [2902, 702]]
    for c in calls:
        assert c["cache_is_draft"]
        assert c["block_table"] == [slot_row(0, 5, 16), slot_row(1, 5, 16)]
    assert out["state"]["leases"] == [0, 1]
    off = run(trees["patched"], "draft_round")
    assert off["calls"][0]["block_table"] == [list(range(10, 22)), [30, 31, 32] + [0] * 9]


def test_verify_round_accept_prefill_and_release(trees):
    out = run(trees["patched"], "verify_round", window = W_TEST)
    assert out["slots_before"] == {"a": 0, "b": 1, "c": 2}
    assert out["accepted"] == {"a": 3, "b": 0, "c": 0}
    # a accepted 3 drafts: one accept prefill of positions 2901..2903 through a's window row
    assert out["calls"] == [{
        "kind": "prefill", "ids": [[21, 22, 23]], "block_table": [slot_row(0, 5, 16)],
        "cache_seqlens": [2901], "cache_is_draft": True, "target_hidden": [[0.0, 1.0, 2.0]],
    }]
    assert out["carry"] == {"a": [[3.0]], "b": [[100.0]]}              # the MTP carry is unchanged
    assert out["active"] == ["a", "b"] and out["deallocated"] == {"a": 0, "b": 0, "c": 1}
    assert out["state"]["leases"] == [0, 1]                            # c's slot came back


def test_verify_rows_follow_batch_order(trees):
    out = run(trees["patched"], "verify_two_rounds", window = W_TEST)
    assert out["slot_a"] == 0
    assert out["rounds"] == [[[slot_row(0, 5, 16)]], [[slot_row(0, 5, 16)]]]   # a's row in both rounds


def test_job_prefill_rows_across_chunks(trees):
    out = run(trees["patched"], "job_prefill", window = W_TEST)
    assert out["chunks"] == 4 and out["kv_position"] == 1499
    calls = out["calls"]
    assert [c["kind"] for c in calls] == ["prefill"] * 4
    assert [c["cache_seqlens"] for c in calls] == [[0], [512], [1024], [1280]]
    assert [len(c["ids"][0]) for c in calls] == [512, 512, 256, 219]
    assert all(c["block_table"] == [slot_row(0, 5, 16)] and c["cache_is_draft"] for c in calls)
    # the shifted target states still chain across chunks (carry of the previous chunk first)
    assert [c["target_hidden"][0][0] for c in calls] == [0.0, 511.0, 511.0, 255.0]
    assert out["state"]["leases"] == [0]
    off = run(trees["patched"], "job_prefill")
    assert all(c["block_table"] == [list(range(20, 27))] for c in off["calls"])


def test_defrag_leaves_windowed_draft_cache(trees):
    assert run(trees["patched"], "defrag", window = W_TEST)["rotated"] == ["main"]
    assert run(trees["patched"], "defrag")["rotated"] == ["draft", "main"]


def test_bc_step_scan_clamp(trees):
    out = run(trees["patched"], "bc_step")
    # below the threshold: dense, untouched; inside one cycle: untouched; beyond: one ring cycle,
    # with regime and the bsz-1 rope position still taken from the true length
    assert out["window_b1_100"] == {"regime": 0, "t_total": 101, "position": 0}
    assert out["window_b1_1000"] == {"regime": 1, "t_total": 1001, "position": 1000}
    assert out["window_b1_5000"] == {"regime": 1, "t_total": 1280, "position": 5000}
    assert out["window_b1_100000"] == {"regime": 1, "t_total": 1280, "position": 100000}
    assert out["window_b2_100000"] == {"regime": 1, "t_total": 1280, "position": 0}
    assert out["unwindowed_b1_100000"] == {"regime": 1, "t_total": 100001, "position": 100000}


def test_bc_init_derives_the_cycle(trees):
    out = run(trees["patched"], "bc_init", window = W_TEST)
    assert out == {"qsa_scan_cap": 5 * PAGE, "threshold": 131, "pool_tokens": 4 * 5 * PAGE}


def test_slot_reuse_keeps_only_matching_pages(trees):
    out = run(trees["patched"], "requeue_affinity", window = W_TEST)
    assert (out["sa"], out["sb"], out["sd"], out["sa2"]) == (0, 1, 2, 0)
    # a released at 2,900 tokens (current page 11): pages 9 and 10 (ring 1, 2) and the sink stay
    # tagged; the continuation reads them back, the rest of the slot is cleared
    assert out["intact"]["a2"] == [True, True, True, False, False]
    assert out["zeroed"]["a2"] == [False, False, False, True, True]
    assert out["zeroed"]["d"] == [True] * 5                          # unrelated prompt: all cleared
    assert out["intact"]["b"] == [True] * 5                          # other active slots untouched
    assert out["se"] == 0 and out["intact"]["e"] == [True, False, False, False, False]


def test_ring_contract(trees):
    out = run(trees["patched"], "ring_contract")
    assert out["offsets_cycle"] == list(range(out["P"]))              # one cycle maps every page once
    assert out["offsets_range"] == [0, out["P"] - 1]                  # never outside the slot run
    for r in out["good"]:
        assert r["checks"] > 100000 and r["violations"] == 0 and r["out_of_run"] == 0, r["first"]
    assert out["bad_short_ring"]["violations"] > 0                    # the checker has teeth
    assert out["bad_no_sink"]["violations"] > 0
