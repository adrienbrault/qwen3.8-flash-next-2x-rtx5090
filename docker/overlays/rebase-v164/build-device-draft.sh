#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:a6e6fa483121311d5abf27009221d4ef8257beba785fd0d4ce2d0236fdd5f20f
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-batch-verify -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/device-draft.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.device-draft -t tabbyapi:flashnext-v164-device-draft .
sudo docker image inspect tabbyapi:flashnext-v164-device-draft -f '{{.Id}}' > native-image-id.txt
