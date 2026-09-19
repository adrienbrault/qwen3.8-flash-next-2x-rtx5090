#!/usr/bin/env bash
# push-to-flan.sh — copy scripts to the GPU host and PROVE they arrived.
#
#   ./push-to-flan.sh r373-restore.sh r375-246-c8.sh
#
# WHY THIS EXISTS. Three times on 2026-09-16 a copy to the host produced a 0-byte file and the check I ran did not
# notice:
#
#   * `for f in a b c; do ssh flan "sudo cat > /srv/qwen5090/$f"; done`  -- no input redirection, so cat truncated
#     each target and read nothing. Three scripts became empty, `bash -n` passed them (an empty script is valid bash),
#     and the chain logged three experiments as "DONE in 0s": they ran, and did nothing.
#   * `ssh flan 'sudo tee ... >/dev/null' < file` worked repeatedly and then produced 0 bytes anyway.
#
# The lesson is not "use tee" or "use cat" -- it is that a transfer needs a check that CAN fail. This script compares
# byte counts and md5s on both ends and exits non-zero on any mismatch, so an empty or truncated copy cannot pass.
#
# It also refuses to leave a partially-written file in place: the transfer lands in /tmp and is moved into position
# only after scp has succeeded.
set -uo pipefail
SSH=${SSH:-flan}
DEST=${DEST:-/srv/qwen5090}
[ $# -gt 0 ] || { echo "usage: $0 <file> [file...]" >&2; exit 2; }

# Both sides normalise: macOS `wc -c < f` pads the number with spaces, so a raw string compare reported a
# MISMATCH on values that were identical -- the guard has to be right about what it is comparing.
size_of(){ wc -c < "$1" | tr -d ' '; }
md5_local(){ md5 -q "$1" 2>/dev/null | tr -d '\n' || md5sum "$1" | awk '{print $1}'; }
fail=0
for f in "$@"; do
  [ -s "$f" ] || { echo "REFUSING: $f is empty or missing locally"; fail=1; continue; }
  base=$(basename "$f")
  if [ "$(dirname "$f")" != "." ] && [ "${KEEP_PATH:-0}" = 1 ]; then
    sub=$(dirname "$f"); ssh -o BatchMode=yes "$SSH" "mkdir -p $DEST/$sub" || { echo "FAILED mkdir $sub"; fail=1; continue; }
    target="$DEST/$sub/$base"
  else
    target="$DEST/$base"
  fi
  if ! scp -q "$f" "$SSH:/tmp/push-$base"; then echo "FAILED scp $f"; fail=1; continue; fi
  if ! ssh -o BatchMode=yes "$SSH" "sudo mv /tmp/push-$base '$target' && sudo chmod +x '$target'"; then
    echo "FAILED install $target"; fail=1; continue
  fi
  lsz=$(size_of "$f"); rsz=$(ssh -o BatchMode=yes "$SSH" "sudo wc -c < '$target' | tr -d ' '")
  lmd5=$(md5_local "$f"); rmd5=$(ssh -o BatchMode=yes "$SSH" "md5sum <(sudo cat '$target') | cut -d' ' -f1 | tr -d '\n'")
  if [ "$lsz" != "$rsz" ] || [ "$lmd5" != "$rmd5" ]; then
    echo "MISMATCH $base: local $lsz/$lmd5 host $rsz/$rmd5"; fail=1; continue
  fi
  echo "ok  $base  $lsz bytes  md5 ${lmd5:0:12}  ->  $target"
done
exit "$fail"
