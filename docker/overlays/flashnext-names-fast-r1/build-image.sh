#!/usr/bin/env bash
# Operator only. Prepared by the CPU round; never invoked by it.
set -euo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
BASE=tabbyapi:tune268-r2-pf2lock-r3
PIN=sha256:da386167cc762f08763a3ab41d98e50403504790b4495053d568f70e61432021
TAG=tabbyapi:flashnext-names-fast-r1
case "${DOCKER_HOST:-unix://default}" in
  unix://*) ;; *) echo 'Refusing remote DOCKER_HOST' >&2; exit 2;;
esac
case "$(docker context inspect --format '{{.Endpoints.docker.Host}}')" in
  unix://*) ;; *) echo 'Refusing a remote Docker endpoint' >&2; exit 2;;
esac
test "$(docker image inspect -f '{{.Id}}' "$BASE")" = "$PIN"
if docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing candidate $TAG" >&2; exit 2
fi
# Docker has a local image-config digest, not a registry manifest digest here.
# Check that exact parent before/after a no-pull build; installer also pins every source byte.
OVERLAY_SHA=$(shasum -a 256 "$P/overlay/manifest.json" | cut -d' ' -f1)
docker build --pull=false --network=none --build-arg "BASE=$BASE" --build-arg "OVERLAY_SHA=$OVERLAY_SHA" -f "$P/Dockerfile" -t "$TAG" "$P"
test "$(docker image inspect -f '{{.Id}}' "$BASE")" = "$PIN"
mkdir -p "$P/out"
docker image inspect -f '{{.Id}} {{index .Config.Labels "flashnext.names.parent"}}' "$TAG" | tee "$P/out/candidate-image.txt"
