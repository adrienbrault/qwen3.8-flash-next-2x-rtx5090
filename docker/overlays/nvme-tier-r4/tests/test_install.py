"""install.py on simulated site-packages trees: the stack base, the base + recurrent-tip-r1 (installed first with
tip's own installer), the base + mtp-pruned-r1 (the served chain's exllamav3 files; round 4), idempotence, refusal on a
foreign baseline or a ple.py without the ple-ckpt-clone r1 fix, and the default path of the ported generator.py.

Every simulated base gets ple-ckpt-clone r1 applied with its own fix_ple.py (install.py requires it), except where a
test says otherwise."""
import ast
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import harness

DELIV = harness.DELIV
ROOT = harness.ROOT
SRC = harness.SRC
TIPDIR = ROOT / "ref" / "recurrent-tip-r1"
PRUNEDDIR = ROOT / "ref" / "mtp-pruned-r1"
PLEDIR = ROOT / "ref" / "ple-ckpt-clone-r1"
PRUNED = "stack-r4-e3r2+mtp-pruned-r1"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run(args, **kw):
    return subprocess.run([sys.executable] + [str(a) for a in args], capture_output = True, text = True, **kw)


def install_into(s, script, tmp_path):
    """Run another round's installer (it finds exllamav3 on sys.path) against the simulated tree."""
    r = run([script], env = dict(os.environ, PYTHONPATH = str(s)), cwd = tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def site(tmp_path, base = "stack", ple_fix = True):
    """site-packages with a copy of src/ (= stack-r4-e3r2), then the base's rounds in the served order: mtp-pruned-r1
    (base "pruned") or recurrent-tip-r1 (base "tip"), then ple-ckpt-clone r1."""
    s = tmp_path / "site-packages"
    shutil.copytree(SRC, s / "exllamav3", ignore = shutil.ignore_patterns("__pycache__", "exllamav3_ext"))
    if base == "pruned":
        install_into(s, PRUNEDDIR / "install.py", tmp_path)
    elif base == "tip":
        install_into(s, TIPDIR / "install.py", tmp_path)
    if ple_fix:
        r = install_into(s, PLEDIR / "fix_ple.py", tmp_path)
        assert "ple-ckpt-clone: patched" in r.stdout, r.stdout
    return s


def test_install_on_stack_base(tmp_path):
    s = site(tmp_path)
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3", "--check"])
    assert r.returncode == 0 and "baseline OK (stack-r4-e3r2)" in r.stdout, r.stdout + r.stderr
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0, r.stderr
    for rel in json.loads((DELIV / "manifest.json").read_text())["variants"]["stack-r4-e3r2"]["post"]:
        assert sha(s / rel) == sha(DELIV / "overlay" / rel)
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0 and "already installed (stack-r4-e3r2)" in r.stdout


def test_install_on_base_plus_tip(tmp_path):
    s = site(tmp_path, "tip")
    assert (s / "exllamav3/cache/recurrent_tip.py").exists()
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0 and "stack-r4-e3r2+recurrent-tip-r1" in r.stdout, r.stdout + r.stderr
    assert sha(s / "exllamav3/generator/generator.py") == sha(DELIV / "overlay-tip/exllamav3/generator/generator.py")
    assert sha(s / "exllamav3/cache/recurrent_tip.py") == sha(TIPDIR / "overlay/exllamav3/cache/recurrent_tip.py")
    gen = (s / "exllamav3/generator/generator.py").read_text()
    assert "self.tip_stash = TipStash.install(self)" in gen and "DiskPageCache.install(self" in gen
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0 and "already installed (stack-r4-e3r2+recurrent-tip-r1)" in r.stdout


def test_refuses_foreign_baseline(tmp_path):
    s = site(tmp_path)
    p = s / "exllamav3/generator/pagetable.py"
    p.write_text(p.read_text() + "\n# drift\n")
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode != 0 and "matches no baseline" in r.stderr
    assert not (s / "exllamav3/generator/disk_cache.py").exists()


def test_generator_port_is_additive_and_guarded():
    """The ported generator.py only adds lines to its base, and every added statement outside the flag block is the
    tier construction or a call under `if self.disk_page_cache is not None`."""
    import difflib
    for base, ported in ((SRC / "generator/generator.py", DELIV / "overlay/exllamav3/generator/generator.py"),
                         (TIPDIR / "overlay/exllamav3/generator/generator.py",
                          DELIV / "overlay-tip/exllamav3/generator/generator.py"),
                         (PRUNEDDIR / "overlay/exllamav3/generator/generator.py",
                          DELIV / "overlay-pruned/exllamav3/generator/generator.py")):
        a = base.read_text().splitlines()
        b = ported.read_text().splitlines()
        added = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a = a, b = b, autojunk = False).get_opcodes():
            assert op in ("equal", "insert"), (op, a[i1:i2])
            if op == "insert":
                added += b[j1:j2]
        code = [l.strip() for l in added if l.strip() and not l.strip().startswith("#")]
        assert code == [
            '_NVME_TIER = os.environ.get("EXL3_NVME_TIER", "")',
            "self.disk_page_cache = None",
            "if _NVME_TIER:",
            "from .disk_cache import DiskPageCache",
            "self.disk_page_cache = DiskPageCache.install(self, cache, draft_cache, _NVME_TIER)",
            "if self.disk_page_cache is not None:",
            "self.disk_page_cache.pump()",
            "if self.disk_page_cache is not None:",
            "self.disk_page_cache.on_busy()",
            "if self.disk_page_cache is not None:",
            "self.disk_page_cache.on_idle()",
        ], code
        ast.parse(ported.read_text())


def test_other_overlay_files_default_path_guards():
    """pagetable.py / recurrent.py / async_generator.py: every added statement sits under a disk_tier /
    disk_page_cache guard (or is the attribute initialisation, the new chain_to_root / on_evict methods)."""
    import difflib
    for rel in ("generator/pagetable.py", "cache/recurrent.py", "generator/async_generator.py"):
        a = (SRC / rel).read_text().splitlines()
        b = (DELIV / "overlay/exllamav3" / rel).read_text().splitlines()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a = a, b = b, autojunk = False).get_opcodes():
            assert op in ("equal", "insert"), (rel, op, a[i1:i2])
    variants = json.loads((DELIV / "manifest.json").read_text())["variants"]
    for rel, want in variants["stack-r4-e3r2"]["pre"].items():
        p = ROOT / "src" / rel
        assert (sha(p) if p.exists() else None) == want
    # the pruned variant differs from the stack only in generator.py, whose baseline is mtp-pruned-r1's own post hash
    pm = json.loads((PRUNEDDIR / "manifest.json").read_text())
    pre = variants[PRUNED]["pre"]
    assert {k: v for k, v in pre.items() if k != "exllamav3/generator/generator.py"} == \
        {k: v for k, v in variants["stack-r4-e3r2"]["pre"].items() if k != "exllamav3/generator/generator.py"}
    assert pre["exllamav3/generator/generator.py"] == pm["post"]["generator/generator.py"] == \
        sha(PRUNEDDIR / "overlay/exllamav3/generator/generator.py")
    # and mtp-pruned-r1 replaces none of this round's other files
    ours = {rel.removeprefix("exllamav3/") for rel in pre}
    assert set(pm["post"]) & ours == {"generator/generator.py"}
    assert variants[PRUNED]["depends"] == {"exllamav3/" + k: v for k, v in pm["post"].items() if k != "generator/generator.py"}


@pytest.mark.parametrize("base", ["stack", "tip", "pruned"])
def test_in_image_smoke_on_installed_tree(tmp_path, base):
    """Run tests/in_image_smoke.py against an installed simulated site-packages tree (stack base, base + tip r1, or
    base + mtp-pruned-r1 = the served chain's exllamav3 files).
    The CUDA-dependent package __init__ and generator.generator import chain are replaced by the harness stubs;
    generator._NVME_TIER is read from the installed file's flag line by executing only that line."""
    s = site(tmp_path, base)
    assert run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"]).returncode == 0
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys, types, re, runpy\n"
        f"sys.path.insert(0, {str(harness.HERE)!r})\n"
        f"os.environ['NVME_TEST_TREE'] = {str(s / 'exllamav3')!r}\n"
        "import harness\n"
        "harness.load()\n"
        f"f = {str(s / 'exllamav3/generator/generator.py')!r}\n"
        "g = types.ModuleType('exllamav3.generator.generator'); g.__file__ = f\n"
        "line = [l for l in open(f) if l.startswith('_NVME_TIER = ')][0]\n"
        "exec(line, {'os': os}, g.__dict__)\n"
        "sys.modules['exllamav3.generator.generator'] = g\n"
        f"runpy.run_path({str(DELIV / 'tests/in_image_smoke.py')!r}, run_name = '__main__')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_NVME_TIER") and k != "NVME_STACKED"}
    r = run([driver], env = env)
    want = {"stack": "stack-r4-e3r2", "tip": "stack-r4-e3r2+recurrent-tip-r1", "pruned": PRUNED}[base]
    assert r.returncode == 0 and f"nvme-tier smoke OK ({want})" in r.stdout, r.stdout + r.stderr


def test_tip_variant_tolerates_tip_module_fix(tmp_path):
    """A later recurrent_tip.py (e.g. a tip bug fix that leaves generator.py alone) is reported, not refused; a
    missing recurrent_tip.py with the tip generator installed is refused."""
    s = site(tmp_path, "tip")
    tipmod = s / "exllamav3/cache/recurrent_tip.py"
    tipmod.write_text(tipmod.read_text() + "\n# later tip fix\n")
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3", "--check"])
    assert r.returncode == 0 and "note: exllamav3/cache/recurrent_tip.py differs" in r.stdout, r.stdout + r.stderr
    tipmod.unlink()
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode != 0 and "recurrent_tip.py: absent (required)" in r.stderr, r.stdout + r.stderr


def test_install_on_pruned_base(tmp_path):
    """The served chain's exllamav3 files (stack + mtp-pruned-r1 + ple-ckpt-clone r1; tool-choice-r1 touches TabbyAPI
    only): the variant is detected from the installed hashes, the merged generator.py goes in, mtp-pruned-r1's own
    files are left alone, and a second run is a no-op."""
    s = site(tmp_path, "pruned")
    pm = json.loads((PRUNEDDIR / "manifest.json").read_text())
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3", "--check"])
    assert r.returncode == 0 and f"baseline OK ({PRUNED})" in r.stdout, r.stdout + r.stderr
    assert "ple-ckpt-clone r1" in r.stdout and "the tested file" in r.stdout and "note:" not in r.stdout, r.stdout
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0 and f"{PRUNED}: 5 files installed" in r.stdout, r.stdout + r.stderr
    m = json.loads((DELIV / "manifest.json").read_text())
    for rel, f in m["variants"][PRUNED]["post"].items():
        assert sha(s / rel) == f["sha256"] == sha(DELIV / f["from"]), rel
    assert sha(s / "exllamav3/generator/generator.py") == sha(DELIV / "overlay-pruned/exllamav3/generator/generator.py")
    for rel, h in pm["post"].items():
        if rel != "generator/generator.py":
            assert sha(s / "exllamav3" / rel) == h, rel
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode == 0 and f"already installed ({PRUNED})" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("base", ["stack", "tip", "pruned"])
def test_refuses_without_ple_fix(tmp_path, base):
    """No variant installs on a ple.py whose stash still returns the live view (R526 try 4): nothing is copied."""
    s = site(tmp_path, base, ple_fix = False)
    gen_before = sha(s / "exllamav3/generator/generator.py")
    for args in (["--check"], []):
        r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"] + args)
        assert r.returncode != 0 and "lacks ple-ckpt-clone r1" in r.stderr, r.stdout + r.stderr
    assert not (s / "exllamav3/generator/disk_cache.py").exists()
    assert sha(s / "exllamav3/generator/generator.py") == gen_before


def test_ple_fix_of_another_shape_is_accepted_with_a_note(tmp_path):
    """The requirement is the fixed stash line, not one exact ple.py: a later ple.py that keeps the fix installs and
    says it is not the tested file."""
    s = site(tmp_path, "pruned")
    p = s / "exllamav3/modules/ple.py"
    p.write_text(p.read_text() + "\n# a later, unrelated ple.py change\n")
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3", "--check"])
    assert r.returncode == 0 and "note: not the tested file" in r.stdout, r.stdout + r.stderr


def test_pruned_variant_is_not_picked_from_a_tag(tmp_path):
    """A pruned tree whose generator.py drifted matches no baseline (detection is by hash): refused, nothing copied."""
    s = site(tmp_path, "pruned")
    g = s / "exllamav3/generator/generator.py"
    g.write_text(g.read_text() + "\n# drift\n")
    r = run([DELIV / "install.py", "--pkg-dir", s / "exllamav3"])
    assert r.returncode != 0 and "matches no baseline" in r.stderr and PRUNED in r.stderr, r.stdout + r.stderr
    assert not (s / "exllamav3/generator/disk_cache.py").exists()
