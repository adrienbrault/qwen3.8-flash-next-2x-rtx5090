#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parent=sha256:144ae9a34e458388d32cfba2a7e1444878225415fbe48c5ae0cf066adddd6683
[[ "$(sudo docker image inspect tabbyapi:flashnext-v164-prefill-policy -f '{{.Id}}')" = "$parent" ]]
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.loads(Path('manifest.json').read_text())
f='repair/recovery-policy.patch'
assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==m['patch_sha256'][f]
PY
sudo docker build --network=none --build-arg BASE="$parent" -f Dockerfile.repair-source -t tabbyapi:flashnext-v164-repair-source .
sudo docker image inspect tabbyapi:flashnext-v164-repair-source -f '{{.Id}}' > native-image-id.txt
