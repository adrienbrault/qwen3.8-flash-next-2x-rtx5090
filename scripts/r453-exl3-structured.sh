#!/usr/bin/env bash
# R453 — structured output on the exllamav3 daily (user 2026-09-17: "What about exllamav3 structured output?").
# Never measured on this track. The served image has TabbyAPI's exl3 grammar path: backends/exllamav3/grammar.py wraps
# llguidance 1.8.0 (`LLGuidanceFilter`) for json_schema / regex_pattern / grammar_string; endpoints/OAI/router.py maps
# `response_format: {"type": "json_schema", "json_schema": ...}` and `{"type": "json"}` onto the sampler's json_schema.
# Questions: (1) does a schema-constrained request return valid JSON that satisfies the schema, thinking ON and OFF;
# (2) does the OpenAI wrapper form ({"name","schema"}) work or only Tabby's bare schema; (3) does constrained decoding keep the
# MTP draft (decode t/s vs an unconstrained control on the same prompt); (4) does it hold at c4 concurrency; (5) regex.
# Runs against the served daily (takes the lock so no experiment swaps the image under it). No daily restart.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-17-r453-exl3-structured; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
IMG=${IMG:-tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2}
BOOTED=0
log(){ echo "$(date -Is) [r453] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
wait_id(){ local want=$1 i; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0; sleep 2; done; return 1; }
finish(){ log "=== R453 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
GPU_QUEUE_NAME=r453-exl3-structured
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)"
if [ "$(served_id)" != "$MODEL" ] || [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" != "$IMG" ]; then
  log "served config is not the daily; booting the live launcher"; BOOTED=1
  bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; finish ABORTED; exit 3; }
  wait_id "$MODEL" || { log "BOOT UNVERIFIED"; finish ABORTED; exit 3; }
fi
python3 - "$API" "$MODEL" "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, sys, time, re, urllib.request, concurrent.futures as cf
api, model, R = sys.argv[1:4]
SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "year": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
    "rating": {"type": "number", "minimum": 0, "maximum": 10},
    "in_print": {"type": "boolean"}}, "required": ["title", "year", "tags", "rating", "in_print"], "additionalProperties": False}
PROMPT = "Invent a plausible science-fiction novel and describe it as a JSON object with the fields title, year, tags, rating (0-10) and in_print. Output only the JSON."
def call(extra, thinking, tag, max_tokens=512):
    req = {"model": model, "temperature": 0, "max_tokens": max_tokens,
           "messages": [{"role": "user", "content": PROMPT}],
           "chat_template_kwargs": {"enable_thinking": thinking}}
    req.update(extra)
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(api + "/chat/completions", data=json.dumps(req).encode(),
                      headers={"Content-Type": "application/json"}), timeout=600))
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        return {"tag": tag, "thinking": thinking, "error": f"{type(e).__name__}: {str(e)[:120]} {body[:200]!r}"}
    wall = time.time() - t
    m = d["choices"][0]["message"]; c = m.get("content") or ""; r = m.get("reasoning_content") or m.get("reasoning") or ""
    u = d.get("usage") or {}; ct = u.get("completion_tokens") or 0
    valid = False; schema_ok = False; parsed = None
    try:
        parsed = json.loads(c); valid = True
        schema_ok = (isinstance(parsed, dict) and set(parsed) == set(SCHEMA["required"]) and isinstance(parsed["title"], str)
                     and isinstance(parsed["year"], int) and isinstance(parsed["tags"], list) and 2 <= len(parsed["tags"]) <= 5
                     and all(isinstance(x, str) for x in parsed["tags"]) and isinstance(parsed["rating"], (int, float))
                     and 0 <= parsed["rating"] <= 10 and isinstance(parsed["in_print"], bool))
    except Exception:
        pass
    return {"tag": tag, "thinking": thinking, "valid_json": valid, "schema_ok": schema_ok, "finish": d["choices"][0].get("finish_reason"),
            "completion_tokens": ct, "reasoning_chars": len(r), "content_chars": len(c), "wall_s": round(wall, 2),
            "tok_per_s": round(ct / wall, 1) if wall else None, "content_head": c[:100]}
rows = []
arms = [("control-none", {}), ("tabby-json_schema", {"json_schema": SCHEMA}),
        ("oai-response_format-json_schema", {"response_format": {"type": "json_schema", "json_schema": {"name": "novel", "strict": True, "schema": SCHEMA}}}),
        ("oai-response_format-json", {"response_format": {"type": "json"}}),
        ("regex-date", {"regex_pattern": r"(19|20)[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"})]
for tag, extra in arms:
    for thinking in (False, True):
        for i in range(2):
            row = call(extra, thinking, f"{tag}#{i}"); rows.append(row); print(json.dumps(row))
# c4 concurrency with the schema, thinking off
print("=== c4 concurrent, tabby json_schema, thinking off ===")
t = time.time()
with cf.ThreadPoolExecutor(4) as ex:
    c4 = list(ex.map(lambda i: call({"json_schema": SCHEMA}, False, f"c4-schema#{i}"), range(4)))
for row in c4: rows.append(row); print(json.dumps(row))
print(json.dumps({"tag": "c4-schema-aggregate", "wall_s": round(time.time() - t, 2),
                  "agg_tok_per_s": round(sum(r.get("completion_tokens", 0) for r in c4) / (time.time() - t), 1),
                  "all_schema_ok": all(r.get("schema_ok") for r in c4)}))
json.dump(rows, open(R + "/rows.json", "w"), indent=1)
ok = [r for r in rows if r.get("tag", "").startswith(("tabby", "oai")) and "error" not in r]
print(f"SUMMARY: schema arms {sum(1 for r in ok if r['schema_ok'])}/{len(ok)} schema_ok, "
      f"{sum(1 for r in ok if r['valid_json'])}/{len(ok)} valid JSON; errors {sum(1 for r in rows if 'error' in r)}")
PY
finish DONE
