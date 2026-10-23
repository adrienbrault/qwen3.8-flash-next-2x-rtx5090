#!/usr/bin/env bash
# Build this incremental overlay on the previous public chain tag; no GPU or network used.
set -euo pipefail
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  patch_sha=$(sha256sum tune268-r2.patch | cut -c1-64)
else
  patch_sha=$(shasum -a 256 tune268-r2.patch | cut -c1-64)
fi
read -r -a docker_cmd <<< "${DOCKER:-docker}"
"${docker_cmd[@]}" build --network=none -f Dockerfile \
  --build-arg BASE="${BASE:-tabbyapi:tune268-r1}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-6}" --build-arg PATCH_SHA="$patch_sha" \
  -t "${TAG:-tabbyapi:tune268-r2}" .
