#!/usr/bin/env bash
# Build off the pinned predecessor from local upstream checkouts.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exl3=${1:?usage: build-image.sh EXLLAMAV3_GIT_REPO TABBYAPI_GIT_REPO}
tabby=${2:?usage: build-image.sh EXLLAMAV3_GIT_REPO TABBYAPI_GIT_REPO}
source "$here/sources.env"
work=$(mktemp -d "${TMPDIR:-/tmp}/rebase167-r2.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
ctx=$work/context

export_tree() {
    local repo=$1 upstream=$2 expected=$3 patch=$4 dest=$5
    local staging=$work/${patch%.patch}
    git clone --quiet --no-checkout --shared -- "$repo" "$staging"
    git -C "$staging" read-tree --reset -u "$upstream"
    git -C "$staging" apply --index --unidiff-zero "$here/$patch"
    local actual
    actual=$(git -C "$staging" write-tree)
    [ "$actual" = "$expected" ] || {
        printf >&2 'Tree mismatch for %s: got %s, expected %s\n' "$patch" "$actual" "$expected"
        return 1
    }
    mkdir -p "$dest"
    git -C "$staging" archive "$actual" | tar -xf - -C "$dest"
    printf 'Verified %s tree %s\n' "$patch" "$actual"
}

export_tree "$exl3" "$EXL3_UPSTREAM" "$EXL3_TREE" exllamav3.patch "$ctx/ctx/exllamav3-src"
export_tree "$tabby" "$TABBY_UPSTREAM" "$TABBY_TREE" tabbyapi.patch "$ctx/ctx/tabbyapi"
cmp "$ctx/ctx/tabbyapi/rebase167_runtime_audit.py" "$here/runtime_audit.py"
[ "${VERIFY_ONLY:-0}" = 0 ] || exit 0

# A separately qualified reconstruction can provide a new predecessor ID.
base=${BASE_OVERRIDE:-$BASE}
[ "$(sudo docker image inspect --format '{{.Id}}' "$base")" = "$base" ] || {
    printf >&2 'Load predecessor image %s, or set BASE_OVERRIDE to a reconstructed predecessor image ID.\n' "$base"
    exit 3
}
if [ -e /srv/qwen5090/gpu-exclusive.lock ]; then
    exec 9</srv/qwen5090/gpu-exclusive.lock
    flock -s -n 9
fi

cp -R "$here/native" "$here/cpu" "$ctx/"
sudo docker build --platform linux/amd64 --network=none --pull=false \
    --build-arg BASE="$base" --build-arg EXL3_REV="$EXL3_REV" --build-arg TABBY_REV="$TABBY_REV" \
    --build-arg ROUND=r2 --build-arg MAX_JOBS="${MAX_JOBS:-4}" \
    -t "${IMAGE_TAG:-tabbyapi:rebase167-r2}" -f "$ctx/native/Dockerfile" "$ctx"
