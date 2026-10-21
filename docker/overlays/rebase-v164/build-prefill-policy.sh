#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
expected=sha256:ec2ac7be9ebf809bf8cb56a9d2f1e41f548a8c3c132ca46c94c97a804aed2c2a
[ "$(sudo docker image inspect tabbyapi:flashnext-v164-native-fixed --format '{{.Id}}')" = "$expected" ]
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('manifest.json').read_text())
assert hashlib.sha256(Path('repair/prefill-policy.patch').read_bytes()).hexdigest()==m['patch_sha256']['repair/prefill-policy.patch']
PY
sudo docker build --network=none -f Dockerfile.prefill-policy -t tabbyapi:flashnext-v164-prefill-policy .
sudo docker image inspect tabbyapi:flashnext-v164-prefill-policy --format '{{.Id}}' > native-image-id.txt
