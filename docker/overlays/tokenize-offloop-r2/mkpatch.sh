#!/usr/bin/env bash
# Regenerate app.patch (base/app -> src/app, -p1 in /app) and exl3.patch (base/exllamav3 -> src/exllamav3, -p1 in the
# installed exllamav3 package dir), the four SHA256SUMS.{app,exl3}.{base,src} files and SHA256SUMS.tests (tests + landing), and check that
#   - base/ is still the served bytes:
#       base/app/backends/exllamav3/model.py, base/app/endpoints/OAI/router.py = the same files in $TABBY (the /app of the
#             served image; loop-think r4/r5 and rebase-dev r3 do not touch these two files)
#       base/app/common/sampling.py = $LT5_SRC/common/sampling.py (loop-think r5's patched file; r5's sampling.py is r4's)
#       base/exllamav3/tokenizer/tokenizer.py = the same file under $EXL3 (the served rebase-dev r3 package)
#   - each patch applies with --fuzz=0 to a fresh copy of those sources and reproduces src/ byte for byte.
# base/app/ and src/ (TabbyAPI files, before and after the patch) are not in this repository: rebuild base/app from $TABBY and
# $LT5_SRC as listed above, and src/ by applying app.patch and exl3.patch to base/.
#   TABBY=<the /app tree of tabbyapi:rebase-dev-r3-loopthink5> EXL3=<its site-packages/exllamav3> \
#     LT5_SRC=<TabbyAPI 53da7919 with loop-think r5's fix.patch applied> bash mkpatch.sh
# CPU only, no network. Run after any edit under src/, tests/ or the landing. Never hand-edit the patches.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TABBY=${TABBY:?set TABBY to the served /app tree}
EXL3=${EXL3:?set EXL3 to the served exllamav3 package directory}
LT5_SRC=${LT5_SRC:?set LT5_SRC to a TabbyAPI tree with loop-think r5 applied}
cd "$HERE"
for f in backends/exllamav3/model.py endpoints/OAI/router.py; do
  cmp -s "base/app/$f" "$TABBY/$f" || { echo "base/app/$f != $TABBY/$f"; exit 1; }
done
cmp -s base/app/common/sampling.py "$LT5_SRC/common/sampling.py" || { echo "base/app/common/sampling.py != loop-think r5 src/"; exit 1; }
( cd base/exllamav3 && find . -type f | LC_ALL=C sort | sed 's#^\./##' | while read -r f; do
    cmp -s "$f" "$EXL3/$f" || { echo "base/exllamav3/$f differs from the served tree"; exit 1; }; done )
T=$(mktemp -d "${TMPDIR:-/tmp}/tok-offloop.XXXXXX"); trap 'rm -rf "$T"' EXIT
gen(){  # $1 = subtree (app | exllamav3), $2 = patch file
  rm -rf "$T/a" "$T/b"; cp -R "base/$1" "$T/a"; cp -R "src/$1" "$T/b"
  find "$T/a" "$T/b" -name __pycache__ -type d -prune -exec rm -rf {} +
  # header timestamps normalised: the patch (and so the image label) is a function of base/ + src/ only; a new file's
  # epoch side keeps a fixed UTC epoch stamp (the creation marker GNU patch reads), other stamps are dropped
  set +e; ( cd "$T" && diff -ruN a b ) > "$2.raw"; rc=$?; set -e
  [ $rc = 1 ] || { echo "diff $1 rc $rc (expected 1: differences)"; exit 1; }
  tab=$(printf '\t')
  sed -E -e "s#^(---|\+\+\+) ([^$tab]+)${tab}1970-01-01 .*\$#\1 \2${tab}1970-01-01 00:00:00.000000000 +0000#" \
         -e "/${tab}1970-01-01 00:00:00.000000000 [+]0000\$/!s#^(---|\+\+\+) ([^$tab]+)${tab}.*\$#\1 \2#" "$2.raw" > "$2"; rm -f "$2.raw"; }
gen app app.patch
gen exllamav3 exl3.patch
sums(){ ( cd "$1" && find . -type f ! -path '*/__pycache__/*' | LC_ALL=C sort | sed 's#^\./##' | xargs sha256sum ); }
sums base/app > SHA256SUMS.app.base; sums src/app > SHA256SUMS.app.src
sums base/exllamav3 > SHA256SUMS.exl3.base; sums src/exllamav3 > SHA256SUMS.exl3.src
# the test code and the landing that vouch for an image (checked in the Dockerfile; hashed into its patch label)
( ls tests/*.py tests/run_offline.sh landing_tokenize_offloop.py | LC_ALL=C sort | xargs sha256sum ) > SHA256SUMS.tests
check(){  # $1 = scratch tree holding the served files at their paths, $2 = patch, $3 = src sums
  patch -p1 --fuzz=0 --forward --dry-run -d "$1" < "$2" > /dev/null
  patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$1" < "$2" > /dev/null
  [ -z "$(find "$1" -name '*.rej' -o -name '*.orig')" ] || { echo "rejects/orig after applying $2"; exit 1; }
  ( cd "$1" && sha256sum -c --quiet "$HERE/$3" ) || { echo "$2 applied != src/"; exit 1; }; }
mkdir -p "$T/app/backends/exllamav3" "$T/app/endpoints/OAI" "$T/app/common"
cp "$TABBY/backends/exllamav3/model.py" "$T/app/backends/exllamav3/model.py"
cp "$TABBY/endpoints/OAI/router.py" "$T/app/endpoints/OAI/router.py"
cp "$LT5_SRC/common/sampling.py" "$T/app/common/sampling.py"
check "$T/app" app.patch SHA256SUMS.app.src
cp -R "$EXL3" "$T/exl3"
check "$T/exl3" exl3.patch SHA256SUMS.exl3.src
for p in app.patch exl3.patch SHA256SUMS.tests; do echo "$p $(wc -l < $p | tr -d ' ') lines, sha256 $(sha256sum $p | cut -c1-64)"; done
echo "image label local.tokoffloop.patch_sha256 = $(cat exl3.patch app.patch SHA256SUMS.tests | sha256sum | cut -c1-64)"
echo "both apply --fuzz=0 to the served sources and reproduce src/"
