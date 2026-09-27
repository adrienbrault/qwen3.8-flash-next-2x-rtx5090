#!/usr/bin/env bash
# Materializes the build context that Dockerfile.box COPYs and that this repository does not carry (25 MB):
#   out/rebase-dev-r3/exllamav3   upstream ExLlamaV3 dev 5783a93 (v1.5.2) + ported-vs-dev.patch = the served engine tree
#   out/rebase-dev-r3/tests       a copy of tests/ (the CPU suites and the TabbyAPI call-site audit the build runs)
# Checks the upstream tree and the result against the SHA-256 recorded when the served image was built (impl-status.md):
# `find exllamav3 -type f | LC_ALL=C sort | xargs sha256sum | sha256sum` in the tree's parent directory, the value
# Dockerfile.box takes as TREE_SHA and stores in the label local.rebase.tree_sha256. Idempotent: an existing tree with the
# right SHA-256 and an identical tests/ copy is kept. Needs git and network access to GitHub; no docker, no GPU.
#   bash docker/overlays/rebase-dev-r3/prepare-tree.sh
# Verified 2026-09-27: a fresh fetch of 5783a93 hashes to VANILLA_SHA, and ported-vs-dev.patch applied with `git apply`
# gives TREE_SHA and a tree byte-identical to the one the served image was built from.
set -euo pipefail
cd "$(dirname "$0")"
UPSTREAM=${UPSTREAM:-https://github.com/turboderp-org/exllamav3.git}
REV=5783a9360b749a1d5bde6862dc35e7766e542ca1
VANILLA_SHA=3f7b139b86f0aba7a860278c61c5c93ed2a944901660f64f6e775004dd6786ad
TREE_SHA=150497b44e6f49b06b8a13f2568e68fc41841bc28964928509e181b633285501
OUT=out/rebase-dev-r3
die(){ echo "prepare-tree: FAILED: $*" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then S256=(sha256sum); else S256=(shasum -a 256); fi
treesha(){ ( cd "$1" && find exllamav3 -type f | LC_ALL=C sort | xargs "${S256[@]}" | "${S256[@]}" | cut -c1-64 ); }

if [ -d "$OUT/exllamav3" ] && [ "$(treesha "$OUT")" = "$TREE_SHA" ] && diff -rq tests "$OUT/tests" >/dev/null 2>&1; then
  echo "prepare-tree: $OUT current (tree $TREE_SHA)"; exit 0; fi

W=$(mktemp -d "${TMPDIR:-/tmp}/prepare-tree.XXXXXX"); trap 'rm -rf "$W"' EXIT
git -C "$W" init -q
git -C "$W" remote add origin "$UPSTREAM"
git -C "$W" fetch -q --depth 1 origin "$REV" || die "fetch of $REV from $UPSTREAM"
git -C "$W" checkout -q FETCH_HEAD
[ "$(git -C "$W" rev-parse HEAD)" = "$REV" ] || die "checked out $(git -C "$W" rev-parse HEAD), not $REV"
got=$(treesha "$W"); [ "$got" = "$VANILLA_SHA" ] || die "upstream tree at $REV hashes to $got, not $VANILLA_SHA"
git -C "$W" apply "$PWD/ported-vs-dev.patch" || die "ported-vs-dev.patch does not apply to $REV"
got=$(treesha "$W"); [ "$got" = "$TREE_SHA" ] || die "patched tree hashes to $got, not $TREE_SHA"
rm -rf "$OUT"; mkdir -p "$OUT"
cp -R "$W/exllamav3" "$OUT/exllamav3"
cp -R tests "$OUT/tests"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
got=$(treesha "$OUT"); [ "$got" = "$TREE_SHA" ] || die "copied tree hashes to $got, not $TREE_SHA"
echo "prepare-tree: $OUT = upstream $REV + ported-vs-dev.patch, tree $TREE_SHA"
