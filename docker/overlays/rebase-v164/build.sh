#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
expected=$(python3 -c 'import json;print(json.load(open("manifest.json"))["parent_image_id"])')
[ "$(sudo docker image inspect tabbyapi:r828-prompt-lookup-r3 --format '{{.Id}}')" = "$expected" ]
python3 prepare.py
sudo docker build --network=none -t tabbyapi:flashnext-v164-r1 .
sudo docker image inspect tabbyapi:flashnext-v164-r1 --format '{{.Id}}' > image-id.txt
