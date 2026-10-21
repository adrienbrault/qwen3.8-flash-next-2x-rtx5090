#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:fae5d087059c4fbfa72c67bba16bfd88d67403d037ab3fe829610b4a7b32b774
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-prefill-pipeline -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/batch-verify.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.batch-verify -t tabbyapi:flashnext-v164-batch-verify .
sudo docker image inspect tabbyapi:flashnext-v164-batch-verify -f '{{.Id}}' > native-image-id.txt
