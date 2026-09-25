#!/usr/bin/env python3
"""`vllm bench serve` (vLLM v0.30.0) made correct against TabbyAPI. R725, 2026-09-25.

Run INSIDE vllm/vllm-openai:v0.30.0 in place of the `vllm` entrypoint, with the same arguments:

    python3 /probes/vllm_bench_tabby.py bench serve --backend openai-chat ...

Every metric is still computed by vLLM's own serve.py (calculate_metrics, percentiles, the result JSON). The shim
replaces three things, each of which is wrong against TabbyAPI in stock v0.30.0 (line refs = the v0.30.0 tree):

 1. The openai-chat request function (lib/endpoint_request_func.py:344). Stock reads `usage` only from a frame whose
    `choices` is empty (`if choices := ... elif usage := ...`, :405/:420). TabbyAPI sends usage in a frame that also
    carries a one-element `choices` list (endpoints/OAI/utils/chat_completion.py _compose_serialize_stream_usage_chunk),
    so stock never reads it, leaves output_tokens = 0, and serve.py:603 falls back to re-tokenizing `delta.content`.
    With thinking on, every reasoning token arrives as `delta.reasoning_content`, so that fallback counts only the
    answer after </think>: throughput and TPOT come out wrong by the reasoning share. The copy here reads usage from
    any frame that carries it and advances TTFT/ITL/latency only on frames that carry text (TabbyAPI's empty-delta
    finish frame and usage frame are not tokens).
    It also splits SSE events itself: TabbyAPI's sse_starlette separates events with CRLF CRLF, and stock's
    StreamedResponseHandler (:23) splits on LF LF only. Once two events land in one read, its buffer never parses again.
 2. Length forcing. `--ignore-eos` sends `ignore_eos`, which TabbyAPI aliases to `ban_eos_token` and the exllamav3
    backend lists in UNSUPPORTED_PARAMS (common/sampling.py): it logs a warning and drops it, so the flag does nothing.
    TabbyAPI's working equivalent is `min_tokens`, which reaches Job(min_new_tokens=...). The copy sets
    min_tokens = max_completion_tokens = the dataset's per-request output length (TABBY_FORCE_LEN=0 disables).
    An --extra-body cannot do this, because it is one fixed dict for every request and ShareGPT lengths differ per
    request.
 3. Spec-Bench double templating. get_samples (datasets.py:2371-2380) does not forward --skip-chat-template to
    SpecBench.sample, and CustomDataset.sample (:2610) then applies the chat template on the client. The openai-chat
    backend sends that already-templated string as a user message, and the server templates it again. The shim
    forces skip_chat_template=True for SpecBench, so the raw turns[0] goes out as the user message, once.
    This is an upstream bug (a local draft only; nothing is filed).

The shim aborts (exit 3) on any vLLM other than 0.30.x, and it prints one `shim:` banner line that the unit greps.

R731 adds one recorder, not a correction: TABBY_SAMPLES_OUT=<path> wraps serve.get_samples and writes the sampled
requests, in the order the client sends them, as a TSV (i, sha256[:16] of the prompt, prompt_len, output_len). The
unit uses it to prove that every concurrency level got the same sample, and to map Spec-Bench categories by prompt
hash. The requests themselves are passed through unchanged.
"""
import codecs
import hashlib
import json
import os
import sys
import time
import traceback

EXPECT_VLLM = "0.30"
FORCE_LEN = os.environ.get("TABBY_FORCE_LEN", "1") != "0"


class SSESplitter:
    """Byte stream -> list of SSE `data:` payloads. Accepts LF or CRLF separators, events split across reads,
    several events in one read, and comment/ping lines (': ping ...'), which are dropped."""

    def __init__(self):
        self.buf = ""
        self._dec = codecs.getincrementaldecoder("utf-8")()

    def feed(self, chunk: bytes, final: bool = False) -> list:
        self.buf += self._dec.decode(chunk, final=final)
        # a CR at the very end may be the first half of a CRLF still in flight: leave it for the next read
        tail = "\r" if (self.buf.endswith("\r") and not final) else ""
        body = self.buf[:-1] if tail else self.buf
        body = body.replace("\r\n", "\n")
        out = []
        while "\n\n" in body:
            ev, body = body.split("\n\n", 1)
            data = [ln[5:].lstrip(" ") for ln in ev.split("\n") if ln.startswith("data:")]
            if data:
                out.append("\n".join(data))
        if final and body.strip():
            data = [ln[5:].lstrip(" ") for ln in body.split("\n") if ln.startswith("data:")]
            if data:
                out.append("\n".join(data))
            body = ""
        self.buf = body + tail
        return out


class StreamState:
    """Walk of the decoded SSE payloads of one chat stream. A frame is a token frame when its delta carries content,
    reasoning_content (TabbyAPI) / reasoning (vLLM) or tool_calls. `usage` is read from whichever frame carries it."""

    def __init__(self, st: float):
        self.st = st
        self.first = False
        self.ttft = 0.0
        self.itl = []
        self.last = st
        self.text = ""
        self.usage = None
        self.frames = 0

    def feed(self, payload: str, ts: float) -> None:
        if payload == "[DONE]":
            return
        data = json.loads(payload)
        if data.get("usage"):
            self.usage = data["usage"]
        choices = data.get("choices") or []
        if not choices:
            return
        d = choices[0].get("delta") or {}
        piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
        if not piece and not d.get("tool_calls"):
            return  # TabbyAPI's empty-delta finish frame and its usage frame: not tokens
        if not self.first:
            self.first = True
            self.ttft = ts - self.st
        else:
            self.itl.append(ts - self.last)
        self.last = ts
        self.text += piece
        self.frames += 1


def build_request_func(E):
    """E = vllm.benchmarks.lib.endpoint_request_func. Returns the TabbyAPI-correct copy of
    async_request_openai_chat_completions (same signature, same RequestFuncOutput contract)."""

    async def tabby_chat(request_func_input, session, pbar=None, mm_position="last"):
        api_url = request_func_input.api_url
        E._validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")
        if request_func_input.chat_messages is not None:
            messages = request_func_input.chat_messages
        else:
            messages = E._get_chat_messages(request_func_input, mm_position=mm_position)
        payload = {
            "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
            "messages": messages,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        E._update_payload_common(payload, request_func_input)
        if FORCE_LEN:
            payload["min_tokens"] = request_func_input.output_len
        headers = E._get_headers("application/json")
        E._update_headers_common(headers, request_func_input)

        output = E.RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        st = time.perf_counter()
        output.start_time = st
        state = StreamState(st)
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    sse = SSESplitter()
                    async for chunk_bytes in response.content.iter_any():
                        for msg in sse.feed(chunk_bytes):
                            state.feed(msg, time.perf_counter())
                    for msg in sse.feed(b"", final=True):
                        state.feed(msg, time.perf_counter())
                    output.generated_text = state.text
                    if state.usage:
                        output.output_tokens = state.usage.get("completion_tokens")
                        if state.usage.get("prompt_tokens") is not None:
                            output.prompt_len = state.usage["prompt_tokens"]
                    if state.first:
                        output.success = True
                        output.ttft = state.ttft
                        output.itl = state.itl
                    else:
                        output.success = False
                        output.error = "Never received a token frame; the request is marked failed."
                    output.latency = state.last - st
                else:
                    body = ""
                    try:
                        body = (await response.text())[:300]
                    except Exception:
                        pass
                    output.error = f"HTTP {response.status} {response.reason or ''} {body}".strip()
                    output.success = False
        except Exception:
            output.success = False
            output.error = "".join(traceback.format_exception(*sys.exc_info()))
        if pbar:
            pbar.update(1)
        return output

    return tabby_chat


def install():
    import vllm
    if not str(vllm.__version__).startswith(EXPECT_VLLM):
        print(f"shim: ABORT vllm {vllm.__version__} is not {EXPECT_VLLM}.x (the patched names are pinned to it)",
              flush=True)
        sys.exit(3)
    import vllm.benchmarks.lib.endpoint_request_func as E
    from vllm.benchmarks.datasets import datasets as D
    for name in ("ASYNC_REQUEST_FUNCS", "_get_chat_messages", "_update_payload_common", "_get_headers",
                 "_update_headers_common", "_validate_api_url", "RequestFuncOutput"):
        if not hasattr(E, name):
            print(f"shim: ABORT endpoint_request_func has no {name}", flush=True)
            sys.exit(3)
    fn = build_request_func(E)
    E.ASYNC_REQUEST_FUNCS["openai-chat"] = fn          # in place: serve.py holds this same dict object
    assert E.ASYNC_REQUEST_FUNCS["openai-chat"] is fn

    orig_sample = D.SpecBench.sample

    def specbench_sample(self, *a, **kw):
        kw["skip_chat_template"] = True
        return orig_sample(self, *a, **kw)

    D.SpecBench.sample = specbench_sample

    # R725 run 1: the v0.30.0 image has no pandas ("Please install vllm[bench]"), and SpecBench.load_data reads the
    # jsonl with pd.read_json. Same logic with the json module: turns[0] per row, category filter, seeded shuffle.
    def specbench_load_data(self):
        import json, random
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
        self.data = []
        with open(self.dataset_path) as f:
            for ln in f:
                if not ln.strip():
                    continue
                row = json.loads(ln)
                if "turns" not in row:
                    raise ValueError("JSONL file must contain a 'turns' column.")
                if (not self.category) or (self.category == row.get("category")):
                    self.data.append({"prompt": row["turns"][0]})
        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    D.SpecBench.load_data = specbench_load_data

    # R731: the sample manifest (see the module docstring). serve.py binds get_samples by name at import
    # (serve.py:45, called at :2180), so the wrapper replaces serve's own global.
    manifest = os.environ.get("TABBY_SAMPLES_OUT")
    if manifest:
        import vllm.benchmarks.serve as S
        if not hasattr(S, "get_samples"):
            print("shim: ABORT vllm.benchmarks.serve has no get_samples", flush=True)
            sys.exit(3)
        orig_get_samples = S.get_samples

        def get_samples_recorded(args, tokenizer, *a, **kw):
            reqs = orig_get_samples(args, tokenizer, *a, **kw)
            with open(manifest, "w") as f:
                f.write("i\tsha\tprompt_len\toutput_len\n")
                for i, r in enumerate(reqs):
                    p = r.prompt if isinstance(r.prompt, str) else json.dumps(r.prompt, sort_keys=True)
                    f.write(f"{i}\t{hashlib.sha256(p.encode()).hexdigest()[:16]}\t{r.prompt_len}\t"
                            f"{r.expected_output_len}\n")
            print(f"shim: {len(reqs)} samples recorded to {manifest}", flush=True)
            return reqs

        S.get_samples = get_samples_recorded
    print(f"shim: patched vllm {vllm.__version__}: openai-chat -> tabby_chat (usage from any frame, CRLF SSE, "
          f"min_tokens=output_len {'ON' if FORCE_LEN else 'OFF'}); SpecBench skip_chat_template forced, "
          f"SpecBench jsonl read without pandas; sample manifest {'ON' if manifest else 'OFF'}", flush=True)


if __name__ == "__main__":
    install()
    from vllm.entrypoints.cli.main import main
    sys.argv = ["vllm", *sys.argv[1:]]
    main()
