#!/usr/bin/env bash
# Restore the validated Flash-Next daily. Run on flan, or ssh flan 'bash -s' < this file.
set -euo pipefail
export HOME=${HOME:-/root}
export USER=${USER:-$(id -un)}
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
D=/srv/qwen5090
if [ "$(readlink /proc/self/fd/9 2>/dev/null || true)" != "$D/gpu-exclusive.lock" ]; then
  exec 9>"$D/gpu-exclusive.lock"
fi
flock -n 9 || { echo 'SUMMARY serve-flashnext-daily REFUSED: GPU lock busy'; exit 1; }
for mark in "$D"/gpu-queue/*; do
  [ -e "$mark" ] || continue
  pid=$(cat "$mark")
  [ "$pid" = "${GPU_QUEUE_SELF:-}" ] && continue
  if [[ "$pid" =~ ^[0-9]+$ ]] && [ -d "/proc/$pid" ]; then
    echo "SUMMARY serve-flashnext-daily REFUSED: queued $(basename "$mark")"; exit 1
  fi
done
expected=49ec4615f005959765c77f99e290d30255270c265bddb04418774d0adfa248c3
[ "$(sha256sum "$D/launch-flashnext.sh" | cut -d' ' -f1)" = "$expected" ]
[ "$(sudo docker image inspect -f '{{.Id}}' tabbyapi:tune268-r2-pf2lock-r3)" = sha256:da386167cc762f08763a3ab41d98e50403504790b4495053d568f70e61432021 ]
test -d "$D/models/qwen3.8-flash-next-exl3-2.68bpw-selfcal"
# Clean environment prevents experiment selectors leaking into the validated launcher.
env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash "$D/launch-flashnext.sh"
curl -fsS -m 10 http://127.0.0.1:8022/v1/model | python3 -c 'import json,sys; assert json.load(sys.stdin)["id"] == "qwen3.8-flash-next-exl3-2.68bpw-selfcal"'
printf '%s\n' flashnext > "$D/default-daily"
echo 'SUMMARY serve-flashnext-daily DONE: Flash-Next healthy :8022; default-daily=flashnext'
