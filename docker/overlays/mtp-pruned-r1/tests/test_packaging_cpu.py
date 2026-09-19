"""CPU test of the packaging: install.py against a copy of the pristine src/ tree (= the stack image's package)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

from _harness import BASELINE, DELIV, OVERLAY


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_manifest_matches_src_and_overlay():
    m = json.load(open(DELIV / "manifest.json"))
    for f, h in m["pre"].items():
        assert sha(BASELINE / f) == h, f            # src/ is still pristine and is the manifest's base
    for f in m["new"]:
        assert not (BASELINE / f).exists()
    for f, h in m["post"].items():
        assert sha(OVERLAY / f) == h, f
    overlay_files = sorted(str(p.relative_to(OVERLAY)) for p in OVERLAY.rglob("*.py"))
    assert overlay_files == sorted(m["post"])


def test_install_on_pristine_copy(tmp_path):
    root = tmp_path / "site"
    shutil.copytree(BASELINE, root / "exllamav3", ignore = shutil.ignore_patterns("*.so", "__pycache__"))
    env = dict(os.environ, PYTHONPATH = str(root))
    r = subprocess.run([sys.executable, str(DELIV / "install.py")], env = env, capture_output = True, text = True)
    assert r.returncode == 0, r.stderr
    m = json.load(open(DELIV / "manifest.json"))
    for f, h in m["post"].items():
        assert sha(root / "exllamav3" / f) == h
    # a second install must refuse: the base is no longer the stack's
    r2 = subprocess.run([sys.executable, str(DELIV / "install.py")], env = env, capture_output = True, text = True)
    assert r2.returncode != 0 and "base is not stack-r4-e3r2" in r2.stderr
