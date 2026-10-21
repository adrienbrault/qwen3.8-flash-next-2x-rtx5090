#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:e9606573db85321dc7693c9a05129e8d1b074f4099ddc818e201da3315c57717
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-repair-source -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text());f='repair/prefill-e3.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.prefill-e3 -t tabbyapi:flashnext-v164-prefill-e3 .
sudo docker image inspect tabbyapi:flashnext-v164-prefill-e3 -f '{{.Id}}' > native-image-id.txt
