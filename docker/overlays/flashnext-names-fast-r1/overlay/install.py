"""All-file preflight before Python-only replacement; retain original-code controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def install(engine, app, originals, check=False):
    manifest = json.loads((HERE / "manifest.json").read_text())
    pending = []
    for relative, pins in manifest.items():
        area, name = relative.split("/", 1)
        root = engine if area == "exllamav3" else app
        dest = root / name
        payload = (HERE / "payload" / relative).read_bytes()
        current = hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else None
        if hashlib.sha256(payload).hexdigest() != pins["candidate"]:
            raise SystemExit("Payload drift: " + relative)
        if current != pins["base"]:
            raise SystemExit("Served source drift: " + relative + " got " + str(current))
        compile(payload, str(dest), "exec")
        pending.append((relative, dest, payload))
    if not check:
        for relative, dest, payload in pending:
            if dest.exists():
                backup = originals / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, backup)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        for relative, dest, payload in pending:
            if dest.read_bytes() != payload:
                raise SystemExit("Post-install mismatch: " + relative)
    print(("Validated" if check else "Installed") + f" {len(pending)} Python files; native extension unchanged")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=Path)
    p.add_argument("--app", type=Path, default=Path("/app"))
    p.add_argument("--originals", type=Path, default=Path("/opt/flashnext-names-r1/original"))
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    engine = a.engine
    if engine is None:
        spec = importlib.util.find_spec("exllamav3")
        if spec is None or spec.origin is None:
            raise SystemExit("Installed exllamav3 not found")
        engine = Path(spec.origin).parent
    install(engine, a.app, a.originals, a.check_only)
