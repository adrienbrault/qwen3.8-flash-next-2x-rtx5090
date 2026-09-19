"""Dry run of tests/gpu_nvme_ab.py against a stub TabbyAPI (HTTP) and stub docker / du commands, so the box run does
not die on a script bug. The stub mimics TabbyAPI's response shapes: /v1/model, /v1/token/encode {tokens, length},
/v1/completions with usage only when stream_options.include_usage is set, SSE streaming."""
import datetime
import http.server
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent / "gpu_nvme_ab.py"


class Stub(http.server.BaseHTTPRequestHandler):
    seen = {}

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/model":
            return self._json({"id": "stub-model"})
        self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/token/encode":
            n = int(len(body["text"].split()) * 1.4)
            return self._json({"tokens": [], "length": n})
        if self.path == "/v1/completions":
            assert body["model"] == "stub-model"
            prompt = body["prompt"]
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(min(body["max_tokens"], 20)):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"text": f" t{i}"}]}).encode() + b"\n\n")
                    time.sleep(0.001)
                self.wfile.write(b"data: [DONE]\n\n")
                return
            n = len(prompt.split())
            cached = (n // 256) * 256 if prompt in Stub.seen else 0
            Stub.seen[prompt] = True
            out = {"choices": [{"text": f"answer {n}"}]}
            if body.get("logprobs"):
                out["choices"][0]["logprobs"] = {"top_logprobs": [{"\\n\\n": -0.2, "<eos>": -1.9, "x": -5.0}]}
            if (body.get("stream_options") or {}).get("include_usage"):
                out["usage"] = {"prompt_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}}
            return self._json(out)
        self.send_error(404)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    th = threading.Thread(target = srv.serve_forever, daemon = True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def write_exe(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


def run(args, **kw):
    return subprocess.run([sys.executable, str(SCRIPT)] + [str(a) for a in args], capture_output = True, text = True,
                          timeout = 120, **kw)


def test_fill_wait_verify_decode_churn(server, tmp_path):
    out = tmp_path / "out"
    # stub docker: `docker logs --timestamps [--since T] NAME` prints a drained line stamped "now"
    docker = write_exe(tmp_path / "docker", "#!/bin/sh\n"
                       "echo \"$(date -u +%Y-%m-%dT%H:%M:%S.000000000Z) INFO: -- nvme tier: drained (stub)\"\n")
    tier = tmp_path / "tier"
    tier.mkdir()
    (tier / "seg").write_bytes(b"x" * 1000)
    du = write_exe(tmp_path / "du", "#!/bin/sh\n# du -sb PATH\necho \"1000\t$2\"\n")
    common = ["--url", server, "--out", out, "--docker", docker, "--container", "flashnext", "--timeout", "5"]

    r = run(["fill", "--sizes", "3000,6000", "--salt", "7"] + common)
    assert r.returncode == 0, r.stdout + r.stderr
    fill = json.loads((out / "fill.json").read_text())
    assert [row["warm"]["cached_tokens"] > 0 for row in fill["sizes"]] == [True, True]
    assert [row["cold"]["cached_tokens"] for row in fill["sizes"]] == [0, 0]

    r = run(["wait-drained"] + common)
    assert r.returncode == 0 and "drained" in r.stdout, r.stdout + r.stderr

    r = run(["verify"] + common)
    assert r.returncode == 0 and "VERIFY PASS" in r.stdout, r.stdout + r.stderr

    r = run(["first-token", "--size", "3000", "--tag", "stub"] + common)
    assert r.returncode == 0 and "margin 1.7" in r.stdout, r.stdout + r.stderr

    r = run(["decode", "--ntok", "3000", "--max-tokens", "20", "--salt", "8"] + common)
    assert r.returncode == 0 and "decode c4" in r.stdout, r.stdout + r.stderr

    r = run(["churn", "--dir", tier, "--cap-gb", "0.001", "--n", "2", "--ntok", "3000", "--du", du] + common)
    assert r.returncode == 0 and "CHURN" in r.stdout and "PASS" in r.stdout, r.stdout + r.stderr


def test_verify_fails_on_hash_change(server, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    Stub.seen.clear()
    prompt = "w " * 600
    (out / "fill.json").write_text(json.dumps({"sizes": [{
        "ntok": 600, "prompt": prompt,
        "cold": {"hash": "x", "cached_tokens": 0, "seconds": 1},
        "warm": {"hash": "0000000000000000", "cached_tokens": 512, "seconds": 1},
    }]}))
    Stub.seen[prompt] = True
    r = run(["verify", "--url", server, "--out", out])
    assert r.returncode == 1 and "VERIFY FAIL" in r.stdout, r.stdout + r.stderr
