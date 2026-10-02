#!/usr/bin/env python3
"""Fail-closed installation/landing; no shell patching or Docker RUN heredoc."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PACKET = Path(__file__).resolve().parent

def roots():
    spec = importlib.util.find_spec('exllamav3')
    assert spec and spec.submodule_search_locations, 'installed exllamav3 absent'
    return Path(next(iter(spec.submodule_search_locations))), Path('/app')

def destination(name, exl3, app):
    if name.startswith('exllamav3/'):
        return exl3 / name.removeprefix('exllamav3/')
    return app / name.removeprefix('tabbyapi/')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    exl3, app = roots()
    base = json.loads((PACKET / 'base-sha256.json').read_text())
    overlay = json.loads((PACKET / 'overlay-sha256.json').read_text())
    unchanged = json.loads((PACKET / 'unchanged-sha256.json').read_text())
    # Check every input before copying any output. Installing on r1/refbase fails here.
    for name, sha in unchanged.items():
        assert digest(destination(name, exl3, app)) == sha, f'unchanged served source drift: {name}'
    expected = overlay if a.check else base
    for name, sha in expected.items():
        assert digest(destination(name, exl3, app)) == sha, f'landing/base hash mismatch: {name}'
    if not a.check:
        for name, sha in overlay.items():
            src = PACKET / 'overlay' / name
            assert digest(src) == sha, f'packet corrupted: {name}'
            dst = destination(name, exl3, app)
            dst.write_bytes(src.read_bytes())
            # Prevent stale bytecode on COPY-preserved timestamps.
            for pyc in (dst.parent / '__pycache__').glob(dst.stem + '.*.pyc'):
                pyc.unlink()
    for name, sha in overlay.items():
        dst = destination(name, exl3, app)
        assert digest(dst) == sha, name
        ast.parse(dst.read_text())
    print('R827 LANDING PASS: exact served base, overlay and untouched prefill/target sources')

if __name__ == '__main__':
    main()
