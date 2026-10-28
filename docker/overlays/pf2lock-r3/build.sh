#!/usr/bin/env bash
# Build this incremental overlay on the previous public chain tag; no GPU or network used.
set -euo pipefail
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  patch_sha=$(sha256sum pf2lock.patch | cut -c1-64)
else
  patch_sha=$(shasum -a 256 pf2lock.patch | cut -c1-64)
fi
read -r -a docker_cmd <<< "${DOCKER:-docker}"
"${docker_cmd[@]}" build --network=none -f Dockerfile \
  --build-arg BASE="${BASE:-tabbyapi:tune268-r2}" --build-arg PATCH_SHA="$patch_sha" \
  --build-arg MODULE_BASE_SHA=b9b227f87b2ec1fb3b635c90268bcb789ebb3ae9fb9ce8dccdcbb556416838b8 \
  --build-arg MODULE_NEW_SHA=e04ef2536c35788160b429d79e751c9402ecddd085c171ed6f2391aa128f65fc \
  -t "${TAG:-tabbyapi:tune268-r2-pf2lock-r3}" .
