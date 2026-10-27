#!/usr/bin/env bash
# Teach the served chat template that reasoning_effort "none" means thinking off (2026-10-27). Qwen's template accepts only low / medium / xhigh and raises on anything else; its own
# off switch is enable_thinking: false. OpenAI-style clients (Hermes title generation and smart approvals) send
# reasoning_effort "none" to mean off, which raised "Unexpected reasoning effort none". The added block maps "none" to
# enable_thinking = false before the effort check; every other value renders byte-identically.
# Idempotent. Backs up the template once as chat_template.jinja.pre-effort-none, then restarts the :8022 daily
# container (same config, model reload) only when the template changed, and checks none / low / default on the API.
# Usage: sudo bash scripts/template-effort-none.sh (on the serving host)
set -euo pipefail
PACK=${PACK:-/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.68bpw-selfcal}
T=$PACK/chat_template.jinja
MARK='reasoning_effort == '"'"'none'"'"''
if ! grep -qF "$MARK" "$T"; then
  [[ -e $T.pre-effort-none ]] || cp -p "$T" "$T.pre-effort-none"
  python3 - "$T" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = "{%- if enable_thinking is undefined or enable_thinking is true %}"
assert s.count(anchor) == 1, "template anchor not found exactly once"
block = ("{#- reasoning_effort 'none' (OpenAI-style clients) = thinking off; scripts/template-effort-none.sh #}\n"
         "{%- if reasoning_effort is defined and reasoning_effort == 'none' %}\n"
         "    {%- set enable_thinking = false %}\n"
         "{%- endif %}\n")
s = s.replace(anchor, block + anchor)
tmp = p + ".next"; open(tmp, "w").write(s)
import os, shutil; shutil.copymode(p, tmp); os.replace(tmp, p)
PY
  echo "template patched: $T"
  docker restart flashnext >/dev/null
  echo "flashnext restarted $(date -u +%FT%TZ)"
else
  echo "template already maps none"
fi
for i in $(seq 1 90); do curl -sf -m3 localhost:8022/v1/models >/dev/null && break; sleep 5; done
M=$(basename "$PACK")
for e in none low default; do
  extra=$([[ $e == default ]] && echo '' || echo ",\"reasoning_effort\":\"$e\"")
  curl -s -m120 localhost:8022/v1/chat/completions -H content-type:application/json \
    -d "{\"model\":\"$M\",\"max_tokens\":400,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Reply with the number only.\"}]$extra}" |
    python3 -c "import json,sys;d=json.load(sys.stdin);m=d.get('choices',[{}])[0].get('message',{}) if 'choices' in d else {};print('$e', repr((m.get('content') or '')[:20]), 'reasoning_chars', len(m.get('reasoning_content') or m.get('reasoning') or ''), d.get('detail','')[:120] if 'detail' in d else '')"
done
