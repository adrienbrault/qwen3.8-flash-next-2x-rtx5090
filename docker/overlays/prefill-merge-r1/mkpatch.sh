#!/usr/bin/env bash
# Regenerate fix.patch (diff -ruN base src), SHA256SUMS.base / SHA256SUMS.src, and check that base/ is still the served
# tree's bytes and that the patch applies with --fuzz=0 to a fresh copy of it and reproduces src/ byte for byte.
# src/ (the patched files) is not in this repository: create it by applying fix.patch to a copy of base/
#   mkdir src && cp -R base/. src/ && patch -p1 --fuzz=0 -d src < fix.patch
# and the served tree with docker/overlays/rebase-dev-r3/prepare-tree.sh. CPU only, no network after that.
# Run after any edit under src/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SERVED=$HERE/../rebase-dev-r3/out/rebase-dev-r3/exllamav3
cd "$HERE"
( cd base && find . -type f | LC_ALL=C sort | sed 's#^\./##' | while read -r f; do
    cmp -s "$f" "$SERVED/$f" || { echo "base/$f differs from the served tree"; exit 1; }; done )
set +e; diff -ruN base src > fix.patch; rc=$?; set -e
[ $rc = 1 ] || { echo "diff rc $rc (expected 1: differences)"; exit 1; }
( cd base && find . -type f | LC_ALL=C sort | sed 's#^\./##' | xargs sha256sum ) > SHA256SUMS.base
( cd src && find . -type f | LC_ALL=C sort | sed 's#^\./##' | xargs sha256sum ) > SHA256SUMS.src
T=$(mktemp -d "${TMPDIR:-/tmp}/pm-r1.XXXXXX"); trap 'rm -rf "$T"' EXIT
cp -R "$SERVED" "$T/exllamav3"
patch -p1 --fuzz=0 --forward --dry-run -d "$T/exllamav3" < fix.patch > /dev/null
patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$T/exllamav3" < fix.patch > /dev/null
[ -z "$(find "$T/exllamav3" -name '*.rej' -o -name '*.orig')" ] || { echo "rejects/orig files after apply"; exit 1; }
( cd "$T/exllamav3" && sha256sum -c --quiet "$HERE/SHA256SUMS.src" ) || { echo "applied tree != src/"; exit 1; }
echo "fix.patch $(wc -l < fix.patch) lines, sha256 $(sha256sum fix.patch | cut -c1-64); applies --fuzz=0 and reproduces src/"
