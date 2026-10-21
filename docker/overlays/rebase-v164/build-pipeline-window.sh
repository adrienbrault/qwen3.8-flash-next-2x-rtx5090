#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:593be05828dfb9ba6638f0670880cd2ba2930a06a84ce5a2c09af2729e3fe613
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-shared-overlap -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/pipeline-window.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.pipeline-window -t tabbyapi:flashnext-v164-pipeline-window .
sudo docker image inspect tabbyapi:flashnext-v164-pipeline-window -f '{{.Id}}' > native-image-id.txt
