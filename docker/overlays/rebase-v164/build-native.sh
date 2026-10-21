#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
expected=$(python3 -c 'import json;print(json.load(open("manifest.json"))["parent_image_id"])')
[ "$(sudo docker image inspect tabbyapi:r828-prompt-lookup-r3 --format '{{.Id}}')" = "$expected" ]
python3 prepare.py
(cd out/engine && git apply --unsafe-paths ../../native.patch && git apply --unsafe-paths ../../repair/recovery-policy.patch && git apply --unsafe-paths ../../repair/prefill-e3.patch && git apply --unsafe-paths ../../repair/prefill-pipeline.patch)
sudo docker build --network=none -f Dockerfile.native -t tabbyapi:flashnext-v164-native .
sudo docker image inspect tabbyapi:flashnext-v164-native --format '{{.Id}}' > native-image-id.txt
