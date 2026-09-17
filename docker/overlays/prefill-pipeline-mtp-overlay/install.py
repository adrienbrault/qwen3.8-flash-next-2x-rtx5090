"""Hash-checked Python-only overlay installer; does not import torch/exllamav3."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def install(root=None, check_only=False):
    here = Path(__file__).resolve().parent
    if root is None:
        spec = importlib.util.find_spec('exllamav3')
        if spec is None or spec.origin is None:
            raise SystemExit('Cannot locate installed exllamav3')
        root = Path(spec.origin).parent
    manifest = json.loads((here / 'manifest.json').read_text())
    writes = []
    for relative, entry in manifest.items():
        dest = root / relative
        current = hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else None
        candidate = (here / 'payload' / relative).read_bytes()
        assert hashlib.sha256(candidate).hexdigest() == entry['candidate'], relative
        if current not in entry['accepted_base'] + [entry['candidate']]:
            raise SystemExit(f'Source mismatch: {dest}; got {current}. Merge/review against this image before overlaying.')
        compile(candidate, str(dest), 'exec')
        writes.append((dest, candidate))
    if not check_only:
        for dest, candidate in writes:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(candidate)
    print(f'{"Validated" if check_only else "Installed"} {len(writes)} Python files at {root}; native library untouched.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, help='exllamav3 package root; default discovers installation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    install(args.root, args.check_only)
