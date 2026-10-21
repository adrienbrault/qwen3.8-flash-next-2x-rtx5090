#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:94d8fbce504a8e74892fd64b4f3519240141f14b3243e32d827cb80813aaf47c
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-prefill-e3 -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/prefill-pipeline.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.prefill-pipeline -t tabbyapi:flashnext-v164-prefill-pipeline .
sudo docker image inspect tabbyapi:flashnext-v164-prefill-pipeline -f '{{.Id}}' > native-image-id.txt
