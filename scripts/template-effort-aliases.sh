#!/usr/bin/env bash
# Map the remaining OpenAI effort levels onto the served Flash-Next template's own (2026-11-26, spec score 1.2: the
# :8022 daily answered "high" and "max" with HTTP 400; user: "make sure our recipes are great"). Qwen's template accepts
# low / medium / xhigh (template-effort-none.sh added none = thinking off). This adds, before the effort check:
#   minimal -> low, high -> xhigh, max -> xhigh
# Those three levels raised before, so every request that rendered before renders byte-identically: the script renders
# the old and new template for none / low / medium / xhigh / unset, with and without tools, and refuses to install
# unless all are identical, and checks the three aliases now render the mapped level's prompt.
# It does NOT restart the daily: TabbyAPI reads the template at model load, so the next daily boot serves it (use
# RESTART=1 to restart the flashnext container when nothing is measuring on it).
# Idempotent. Backs up the template once as chat_template.jinja.pre-effort-aliases.
# Usage: sudo bash scripts/template-effort-aliases.sh (on the serving host)
set -euo pipefail
PACK=${PACK:-/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.68bpw-selfcal}
T=$PACK/chat_template.jinja
MARK="flan/flashnext-v164/template-effort-aliases.sh"
if grep -qF "$MARK" "$T"; then echo "template already maps the aliases"; exit 0; fi
if [[ ${REQUIRE_NONE:-1} == 1 ]]; then
  grep -qF "reasoning_effort == 'none'" "$T" || { echo "FAILED: run template-effort-none.sh first"; exit 1; }
fi
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -p "$T" "$W/old.jinja"
python3 - "$W/old.jinja" "$W/new.jinja" <<'PY'
import sys
s = open(sys.argv[1]).read()
anchor = "{%- if enable_thinking is undefined or enable_thinking is true %}"
assert s.count(anchor) == 1, "template anchor not found exactly once"
block = ("{#- OpenAI effort aliases: minimal -> low, high / max -> xhigh; flan/flashnext-v164/template-effort-aliases.sh #}\n"
         "{%- if reasoning_effort is defined and reasoning_effort in ('minimal', 'high', 'max') %}\n"
         "    {%- set reasoning_effort = {'minimal': 'low', 'high': 'xhigh', 'max': 'xhigh'}[reasoning_effort] %}\n"
         "{%- endif %}\n")
open(sys.argv[2], "w").write(s.replace(anchor, block + anchor))
PY
# Render check in the served image's Python (its jinja2 is what TabbyAPI uses).
IMG=${IMG:-$(docker inspect flashnext --format '{{.Config.Image}}')}
docker run --rm -i --network none --entrypoint python3 -v "$W:/w:ro" "$IMG" - > "$W/check.txt" <<'PY'
import jinja2, jinja2.sandbox
def load(p):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    def raise_exception(m): raise ValueError(m)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(open(p).read())
old, new = load("/w/old.jinja"), load("/w/new.jinja")
msgs = [{"role": "user", "content": "What is 17*23?"}]
tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
def render(t, effort, tl):
    kw = dict(messages=msgs, add_generation_prompt=True, bos_token="", eos_token="")
    if tl: kw["tools"] = tools
    if effort is not None: kw["reasoning_effort"] = effort
    try: return t.render(**kw)
    except Exception as e: return "RAISE " + str(e)
bad = 0
for tl in (False, True):
    for e in (None, "none", "low", "medium", "xhigh"):
        a, b = render(old, e, tl), render(new, e, tl)
        if e == "none" and a.startswith("RAISE"):   # template without the none patch: must still raise identically
            ok = a == b; bad += not ok; print("identical" if ok else "DIFFERS", e, "tools" if tl else "plain", "(raises in both)"); continue
        ok = a == b and not a.startswith("RAISE")
        bad += not ok; print("identical" if ok else "DIFFERS", e, "tools" if tl else "plain")
    for e, m in (("minimal", "low"), ("high", "xhigh"), ("max", "xhigh")):
        a, b, want = render(old, e, tl), render(new, e, tl), render(old, m, tl)
        ok = a.startswith("RAISE") and b == want
        bad += not ok; print("alias ok" if ok else "ALIAS WRONG", e, "->", m, "tools" if tl else "plain")
    a = render(new, "bogus", tl); ok = a.startswith("RAISE"); bad += not ok; print("still rejects bogus" if ok else "ACCEPTS BOGUS", "tools" if tl else "plain")
raise SystemExit(1 if bad else 0)
PY
cat "$W/check.txt"
# Fail closed: an empty or short report (e.g. stdin not reaching the container) is not a pass.
[[ $(grep -cE '^(identical|alias ok|still rejects bogus) ' "$W/check.txt") == 18 ]] || { echo "FAILED: render check incomplete"; exit 1; }
[[ -e $T.pre-effort-aliases ]] || cp -p "$T" "$T.pre-effort-aliases"
cp "$W/new.jinja" "$T.next"; chmod --reference="$T" "$T.next"; chown --reference="$T" "$T.next"; mv "$T.next" "$T"
echo "template patched: $T (sha256 $(sha256sum "$T" | cut -c1-16)); served from the next daily boot"
if [[ ${RESTART:-0} == 1 ]]; then docker restart flashnext >/dev/null; echo "flashnext restarted $(date -u +%FT%TZ)"; fi
