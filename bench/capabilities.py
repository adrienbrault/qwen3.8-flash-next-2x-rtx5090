#!/usr/bin/env python3
"""Daily-capability gate for the Flash-Next TabbyAPI endpoint: structured output, tool calls, vision, reasoning.

Why it exists: PROMOTION.md lists structured output as the one capability this stack had not exercised. Grammar
filters exist in this backend, but "the flag exists" is not evidence, so each capability below is asserted on the
response the server actually returned:

  json_schema  a response_format json_schema request must return content that parses as JSON AND satisfies the
               schema (required keys present, types correct, enum respected) -- not merely "looks like JSON"
  tools        a request carrying a tool must come back with parsed tool_calls whose arguments parse as JSON and
               name only parameters the tool's schema declares
  vision       a 32x32 PNG sent as a data URI must be described with the colour it actually contains
  reasoning    the same request must put its thinking in reasoning_content while the answer goes to content

Stdlib only. Every check writes a PASS/FAIL line with the evidence, and the exit code is the number of failures.

usage: capabilities.py --url http://127.0.0.1:8022/v1 --model qwen3.8-flash-next-exl3-3.05bpw --out r348.jsonl
"""
import argparse, base64, json, struct, sys, urllib.request, zlib


def png_red(size=32):
    """A solid red PNG, built here so the gate has no binary fixture to lose."""
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)   # 8-bit truecolour RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def types_ok(value, spec):
    """Minimal JSON-Schema type check for the shapes this gate asks for."""
    if spec.get("type") == "object":
        if not isinstance(value, dict):
            return False
        for k in spec.get("required", []):
            if k not in value:
                return False
        return all(types_ok(v, spec.get("properties", {}).get(k, {})) for k, v in value.items()
                   if k in spec.get("properties", {}))
    if spec.get("type") == "array":
        return isinstance(value, list) and all(types_ok(v, spec.get("items", {})) for v in value)
    if spec.get("type") == "string":
        if not isinstance(value, str):
            return False
        return value in spec["enum"] if "enum" in spec else True
    if spec.get("type") == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec.get("type") == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec.get("type") == "boolean":
        return isinstance(value, bool)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = open(a.out, "a", buffering=1)
    fails = 0

    def report(name, ok, evidence):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {name}: {evidence}", flush=True)
        log.write(json.dumps({"check": name, "pass": bool(ok), "evidence": str(evidence)[:400]}) + "\n")

    # --- 1. JSON schema structured output ---------------------------------------------------------
    schema = {"type": "object",
              "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
                             "coastal": {"type": "boolean"},
                             "climate": {"type": "string", "enum": ["arid", "temperate", "tropical", "polar"]}},
              "required": ["city", "population", "coastal", "climate"]}
    r = post(a.url, {"model": a.model, "max_tokens": 2048, "temperature": 0,
                     "messages": [{"role": "user", "content":
                                   "Give me one large coastal city with its population and climate."}],
                     "response_format": {"type": "json_schema",
                                         "json_schema": {"name": "city_facts", "schema": schema,
                                                         "strict": True}}})
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    try:
        parsed = json.loads(content)
        ok = types_ok(parsed, schema)
        report("json_schema", ok, f"parsed and schema-valid: {parsed}" if ok
               else f"parsed but violates the schema: {parsed}")
    except Exception as e:
        report("json_schema", False, f"content is not JSON ({e}): {content[:200]!r}")

    # --- 2. tool call parsing ---------------------------------------------------------------------
    tools = [{"type": "function", "function": {
        "name": "write_note",
        "description": "Write a note to a file.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["file_path", "content"]}}}]
    r = post(a.url, {"model": a.model, "max_tokens": 2048, "temperature": 0, "tools": tools,
                     "tool_choice": "auto",
                     "messages": [{"role": "user",
                                   "content": "Save a note saying 'hello from the gate' to /tmp/gate-note.txt."}]})
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        report("tools", False, f"no tool_calls; finish_reason={r['choices'][0].get('finish_reason')}, "
                               f"content={(msg.get('content') or '')[:160]!r}")
    else:
        call = calls[0]
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            allowed = set(tools[0]["function"]["parameters"]["properties"])
            ok = (fn.get("name") == "write_note" and set(args) <= allowed
                  and {"file_path", "content"} <= set(args))
            report("tools", ok, f"name={fn.get('name')} args={args}")
        except Exception as e:
            report("tools", False, f"arguments are not JSON ({e}): {fn.get('arguments')!r}")

    # --- 3. vision --------------------------------------------------------------------------------
    b64 = base64.b64encode(png_red()).decode()
    r = post(a.url, {"model": a.model, "max_tokens": 1024, "temperature": 0,
                     "messages": [{"role": "user", "content": [
                         {"type": "text", "text": "What colour fills this image? One word."},
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
    msg = r["choices"][0]["message"]
    ans = ((msg.get("content") or "") + (msg.get("reasoning_content") or "")).lower()
    report("vision", "red" in ans, f"answer: {(msg.get('content') or '')[:120]!r}")

    # --- 4. reasoning channel ---------------------------------------------------------------------
    r = post(a.url, {"model": a.model, "max_tokens": 1024, "temperature": 0,
                     "messages": [{"role": "user", "content": "How many r are in strawberry? Explain briefly."}]})
    msg = r["choices"][0]["message"]
    has_r = bool((msg.get("reasoning_content") or "").strip())
    has_c = bool((msg.get("content") or "").strip())
    report("reasoning_channel", has_r and has_c,
           f"reasoning_content {len(msg.get('reasoning_content') or '')} chars, content {len(msg.get('content') or '')} chars")

    log.close()
    print(f"failures: {fails}", flush=True)
    sys.exit(fails)


if __name__ == "__main__":
    main()
