#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:59f5950c8e52489b97e5d871d0ec43b6f1e45d4eaf29fd382e75dfa4278a9ef7
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-device-draft -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/shared-overlap.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.shared-overlap -t tabbyapi:flashnext-v164-shared-overlap .
sudo docker image inspect tabbyapi:flashnext-v164-shared-overlap -f '{{.Id}}' > native-image-id.txt
