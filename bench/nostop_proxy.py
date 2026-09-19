#!/usr/bin/env python3
"""Eval-only proxy: forwards OpenAI-compatible requests to the upstream server with the "stop" field removed.

WHY (R503, 2026-09-18): lm-eval's 5-shot gsm8k sends until=["Question:", "</s>", "<|im_end|>"] as server-side stop strings.
Qwen3.8-Flash-Next thinks first and often restates the problem as "Question: ..." inside its reasoning; TabbyAPI applies the
stop string to the reasoning text, so generation ends mid-thought with content null and lm-eval scores an empty answer.
On 3.05bpw 36 of 43 wrong answers were such empties (2.50bpw: 89 of 98). The answer text lm-eval scores is the message
content, which never contains the few-shot "Question:" continuation, so dropping the stop field changes nothing else.

usage: nostop_proxy.py --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022
"""
import argparse
import json
import signal
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--listen", default="127.0.0.1:8031")
ap.add_argument("--upstream", default="http://127.0.0.1:8022")
args = ap.parse_args()
STRIPPED = 0


class H(BaseHTTPRequestHandler):
    def _forward(self, body):
        req = urllib.request.Request(args.upstream + self.path, data=body, method=self.command,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                code, data, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            code, data, ctype = e.code, e.read(), e.headers.get("Content-Type", "application/json")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._forward(None)

    def do_POST(self):
        global STRIPPED
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            j = json.loads(body)
            if "stop" in j:
                del j["stop"]
                STRIPPED += 1
            body = json.dumps(j).encode()
        except ValueError:
            pass
        self._forward(body)

    def log_message(self, *a):
        pass


host, port = args.listen.rsplit(":", 1)
srv = ThreadingHTTPServer((host, int(port)), H)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # print the counter on kill
print(f"nostop proxy {args.listen} -> {args.upstream}", flush=True)
try:
    srv.serve_forever()
finally:
    print(f"requests with stop removed: {STRIPPED}", flush=True)
