# tokenize-offloop-r2: each prompt is encoded once per request, and a long prompt's encode runs off the event loop

`tabbyapi:tokenize-offloop-r2` is `tabbyapi:rebase-dev-r3-loopthink5` plus two patches: [`app.patch`](app.patch) on TabbyAPI (`backends/exllamav3/model.py`, `endpoints/OAI/router.py`, `common/sampling.py`, new `common/tokenize_offloop.py`) and [`exl3.patch`](exl3.patch) on the installed ExLlamaV3 package (`tokenizer/tokenizer.py`). Python only, no extension rebuild; the engine's kernels, the 39 launcher keys and the page pool are unchanged. Served since 2026-09-29 19:01 CEST ([R808](../../../bench/results/r805-r808-tokenize-offloop.md)).

```sh
B=tabbyapi:rebase-dev-r3-loopthink5
docker build -f Dockerfile.box --build-arg BASE=$B --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --label local.tokoffloop.patch_sha256=$(cat exl3.patch app.patch SHA256SUMS.tests | sha256sum | cut -c1-64) \
  -t tabbyapi:tokenize-offloop-r2 .
```

The label is `b961eb83dd3951129a307d12d0f5e1eda45211b72a3a3da5d6890c7efe8ee9fd` for the files in this directory, the value on the served image.

## Why

- [R805](../../../bench/results/r805-r808-tokenize-offloop.md) (2026-09-29) measured that a streamed 30,818-token prompt arriving while 4 streams decode stalls every one of them for a median 52.8 ms before the server logs the prompt (16 of 16 injections; 2.8 ms for a 98-token prompt).
- The stall is two full encodes of the same prompt on the thread that also runs `generator.iterate()`, about 25 ms each, one event-loop turn apart:
  - A: the streamed router's `check_context_length` → `backends/exllamav3/model.py` `validate_context_length`, which encodes the whole prompt only to take its length;
  - B: `generate_gen` encodes the same prompt again before it builds the job.
- HF `tokenizers` `Tokenizer.encode()` holds the GIL for the whole encode, so moving it to a worker thread alone does not free the loop. `encode_batch()` releases the GIL, and returns the same ids for a one-element batch with no padding or truncation configured.

**What r2 changes against r1.** r1 ([R806](../../../bench/results/r805-r808-tokenize-offloop.md)) sent every encode through the worker and removed the stall (53.5 → 3.5 ms), but each executor hop cost the arriving request 1 to 2 decode steps of time to first token: the loop picks the worker's result up only between two synchronous `iterate()` calls (~17 ms apart at 4 streams), and the worker needs the GIL that `iterate()` mostly holds. For ~98-token prompts arriving during 4-stream decode, the first token came 41.6 ms later than without the patch (255.9 against 214.3 ms). r2 hops only when the prompt has more than `EXL3_TOKENIZE_OFFLOOP_MIN_CHARS` characters (default 12,000) and encodes shorter prompts inline. Encoding once stays on at every length. `exl3.patch` is byte-identical to r1's, and the ExLlamaV3 revision string stays `tokenize-offloop-r1`.

**Why 12,000 characters.** This model's rendered prompts run 2.87 characters per token (R806: 60,301 characters = 20,992 tokens), and the served container encodes about 0.9 µs per token (R806: 139,502 tokens in 125 ms). 12,000 characters is therefore about 4,200 tokens, or about 3.8 ms of inline encode: less than a quarter of a decode step, where a hop would cost the arriving request 1 to 2 steps. On the build host with a Qwen byte-level BPE tokenizer, a short encode costs the same through `encode` and `encode_batch` (0.090 against 0.084 ms at 300 characters), and 12,000 characters take 3.8 to 4.0 ms. `EXL3_TOKENIZE_OFFLOOP_MIN_CHARS=0` restores r1 (a hop at every length).

**Busy worker.** A short prompt also goes to the worker while an earlier hop is still queued or running there. Inline, it would wait on the ExLlamaV3 encode lock held by the worker and block the loop for the rest of the long encode; queued behind that encode on the worker, it delays only its own request. `tests/test_hops.py` covers this case. R808's measurement fires one event at a time, so its short-prompt TTFT never takes this path.

## Changes

Three knobs, all read per call. The two on/off knobs default ON, and `=0` restores the base image's code path; they serve as kill switches for a change that is meant to be bitwise (same ids, same job, same order). With both at `=0` the base image's code paths run, but the bytes differ from the base image: `encode_part_base` still takes the lock and writes the flag only on change. The ids and the flag state are the same. The launcher sets none of them.

| knob | lever | ON | `=0` |
|---|---|---|---|
| `TABBY_ENCODE_ONCE` | encode once | `validate_context_length` encodes the prompt exactly as `generate_gen` does and stores the ids tensor on the request (`BaseSamplerRequest._prompt_ids`, keyed by `(prompt, add_bos)`, value `(weakref.ref(tokenizer), ids)`). `generate_gen` reuses a `.clone()` of it when the text and `add_bos` match and the weakref still resolves to the loaded tokenizer. A reload between the check and the job (another Tokenizer instance, or a dead weakref) misses and encodes again. The key never holds the tokenizer object or its `id()`, which a new instance could reuse. | `generate_gen` encodes again |
| `EXL3_TOKENIZE_OFFLOOP` | off the loop | the check (router) and any remaining `generate_gen` encode of a prompt longer than the threshold run on a one-thread executor (`run_tokenize`); shorter ones run inline. ExLlamaV3's `encode_part_base` encodes through `encode_batch([text])[0].ids`, which releases the GIL. | inline on the loop, HF `encode()` |
| `EXL3_TOKENIZE_OFFLOOP_MIN_CHARS` | threshold | default 12000: hop when the prompt text (a list prompt: the sum) has more characters than this, or while the worker is busy. `0` = hop at every length (r1). Anything but plain ASCII digits (a sign, `_` as in `1_000`, surrounding whitespace, a decimal point, non-ASCII digits) means the default. | (no effect with `EXL3_TOKENIZE_OFFLOOP=0`) |

**ExLlamaV3** (`exl3.patch`, `tokenizer/tokenizer.py`, installed package):
- `encode_part_base` calls `encode_batch([t], add_special_tokens = False)[0].ids` instead of `encode(t, add_special_tokens = False).ids`. The ids are the same: one input, no padding or truncation configured (`no_truncation()` at init), and the same `encode_special_tokens` mode.
- The `encode_special_tokens` write and the encode that depends on it run under a per-instance `threading.Lock` (`self._encode_lock`, created first in `__init__`). The flag is shared state on the HF tokenizer. Without the lock, another thread could flip it between a worker's write and its encode, and the prompt would be tokenized in the wrong split-special mode.
- The lock is the Tokenizer's own, not `util.misc`'s RLock, because the generator takes that one while iterating (`get_id_to_piece_list`).
- The flag is written only when it changes. With tokenizers 0.23.2 (the served version, asserted by the landing check), a write waits for an `encode_batch` running on another thread (976 ms measured); another PyO3 build could raise "Already borrowed" instead. The steady state never writes, because every prompt encode uses `special=True`.
- The whole split-on-unspecial-tokens loop runs inside one lock hold, in the worker thread.
- `threading.Lock` is not FIFO. A loop-thread encode waiting behind back-to-back worker encodes gets the lock in practice, because the worker runs a few ms of GIL-held Python (ids list to tensor, return, next job) before it can take the lock again. There is no starvation guarantee.

**TabbyAPI** (`app.patch`, `/app`):
- `common/tokenize_offloop.py` (new): `run_tokenize(nchars, fn, *args, **kwargs)` calls `fn` inline when `should_offload(nchars)` is false. Otherwise it runs `fn` on a dedicated `ThreadPoolExecutor(max_workers=1)` with the context copied, as `asyncio.to_thread` does.
  - `should_offload(n)` = offload on and (`n > offload_min_chars()` or a hop is still pending on the worker).
  - The pending count is exact: it rises at submit and falls in the future's done-callback, which runs when the call ends, raises, or is cancelled while still queued. It is behind a small `threading.Lock`, separate from the encode lock.
  - Hops use `executor.submit` + `asyncio.wrap_future`, which is what `loop.run_in_executor` does. A cancelled await still cancels a queued call.
  - Calls run one at a time, in submission (= arrival) order, and stay off the default executor. Results and exceptions are those of `fn`.
- `endpoints/OAI/router.py`: `await run_tokenize(prompt_chars(prompt), model.check_context_length, prompt, ...)` on the streamed `/v1/completions` and `/v1/chat/completions` paths. The rejection still happens before the HTTP 200, with the same exception (`ContextLengthHTTPException`, `status_code` 400, identical text under all four knob settings, tested on `check_context_length` directly; the router itself is only AST-checked).
- `backends/exllamav3/model.py`: `validate_context_length` encodes via `self.tokenizer.encode(prompt, add_bos, encode_special_tokens=True, embeddings=[])`, which is `generate_gen`'s call, so `context_len` is unchanged, and stores the ids after validation passes; `_encode_prompt` (new) reuses them, else `await run_tokenize(len(prompt), self.tokenizer.encode, ...)`; `generate_gen` takes its ids from `_encode_prompt`, still before `AsyncJob(...)`.
- `common/sampling.py`: `_prompt_ids: Optional[dict] = PrivateAttr(default=None)` beside loop-think's `_loop_backstop`. `data.model_copy(deep=True)` (one per choice) deep-copies it: one tensor copy per choice. Private attributes are never serialized.

**Paths:**
- Streamed chat and streamed completions: one encode per prompt, in the worker when the prompt is long, inline otherwise (0 hops).
- Non-streamed chat and completions skip the pre-200 check (as in TabbyAPI `53da7919`), so they already encode once. That encode runs in the worker when the prompt is long.
- `n > 1`: every choice reuses the check's ids. A list-prompt `/v1/completions` stores one entry per prompt.
- `forced_tool_generation` builds another prompt string, so it misses and encodes. Multimodal requests never store or reuse ids and encode in both steps, as before.

**Not changed:**
- `/v1/token/encode` and the Kobold router still encode on the loop.
- The per-chunk fallback `self.tokenizer.encode(chunk)` (when a result carries no `token_ids`), loop-think's injection encode and ExLlamaV3's banned-string reference encode still run on the loop. They are short. The busy-worker rule only covers `run_tokenize` callers, so if one of these arrives while a worker encode holds the lock, it waits on the loop for the rest of that encode: ~0.9 µs per remaining token, at most ~110 ms at 120k tokens. The base image blocks the loop for that same prompt's encode in full, twice, so the total is lower, but a single event can wait longer than before.
- Request parsing, pydantic, the template render and JSON dumps (~2.5 ms per 30k tokens) stay on the loop.

**Characters are not tokens.** The threshold counts characters: 12,000 characters ≈ 4,200 tokens ≈ 3.8 ms inline at this model's 2.87 characters per token. CJK text runs at ~1 character per token, so a 12,000-character prompt is ~12k tokens and ~11 ms of inline loop stall below the threshold; byte-fallback-heavy text (emoji, rare scripts) can reach several tokens per character, 30 to 45 ms inline. Both are rare on this traffic. A UTF-8 byte count would be a tighter proxy at the same cost; r2 does not use one.

**Behaviour changes:**
- `generate_gen` has an `await` before `AsyncJob(...)`. A disconnect during the encode raises `CancelledError` there, before the `try`: no job is created and nothing is registered. The worker finishes the encode and drops the result.
- The router's check is a cancellation point before `EventSourceResponse`. A cancel there becomes the handler's existing 422 "cancelled by user".
- A `generate_gen` whose ids are reused runs `_encode_prompt` without suspending, and one that encodes (non-streamed, or a miss) yields to the worker. Under mixed traffic, `AsyncJob` enqueue order can differ from arrival order. This changes scheduling only, not any request's output.
- `check_context_length` runs on the worker and reads the module global `container` when it runs. If the model unloads between the handler's `check_model_container()` and the worker running a queued check, the request gets an AttributeError 500 instead of a 503.
- `Tokenizer.num_tokens` calls HF `encode` without `_encode_lock` and reads whatever `encode_special_tokens` the last locked encode left. Every worker encode is a prompt encode with the same mode, so the steady state never flips the flag.

## Tests

The build runs all of them inside the image (the landing check, the five offline tests on a synthetic tokenizer, the served `rebase-dev-r3` CPU suite against the patched package and its TabbyAPI call-site audit against the patched `/app`) and fails on any failure. On the build host (macOS, Python 3.12, tokenizers 0.23.2, torch CPU) on 2026-09-29: identity 2,750 checks / 0 failed, lock 4 / 0, gil 4 / 0, event loop 93 / 0, hops 144 / 0, landing 14 checks, the served CPU suite with 0 failures, the call-site audit passing. `test_hops.py` fails 8 checks when every call hops, 2 without the busy-worker rule, and 2 with `>=` in place of `>`.

- `tests/_load.py`: stub-package loader (no CUDA extension), a synthetic Qwen-shaped byte-level BPE tokenizer trained on the test corpus, and the corpus. No tokenizer file is shipped.
- `tests/test_identity.py`: the patched `Tokenizer.encode` returns the base file's ids for every corpus string, both `encode_special_tokens` modes and both `add_bos` values, with the knob off too; the HF flag state after a call sequence matches the base file's.
- `tests/test_lock.py`: two threads in opposite `encode_special_tokens` modes on `<|im_start|>` markup, each encode behind a proxy that sleeps 0.5 ms between the flag write and the encode. With the lock: 2 s, 0 wrong encodes. With the lock replaced by a no-op (the sensitivity leg): must find a wrong encode within 20 s (a failure on the build host, a warning inside the image build, where a busy CPU may never interleave the threads).
- `tests/test_gil.py`: a pure-Python thread keeps at least 80 % of its idle rate while another thread encodes a long prompt, and at most 60 % with `EXL3_TOKENIZE_OFFLOOP=0` (the control that shows the measure sees the GIL).
- `tests/test_event_loop.py`: a 1 ms ticker on the event loop while a long prompt goes through the router check and `generate_gen`, per knob arm: the ids the job receives equal the base file's, one encode with encode-once on and two with it off, and the ticker's largest gap is small only where the arm moves the encode. A worker thread with a GIL-holding `encode()` still stalls the loop.
- `tests/test_hops.py` (r2): short prompts take 0 hops and encode on the loop thread (streamed and non-streamed, 300 / 11,999 / 12,000 characters); 12,001 characters take 1 hop; list prompts count their total; knob parsing; `EXL3_TOKENIZE_OFFLOOP=0` never hops; the busy-worker rule; a cancelled queued call and a raising call leave the pending count at 0.
- `landing_tokenize_offloop.py`: 14 import-time checks on the real packages in the image, including `tokenizers.__version__ == "0.23.2"` and the threshold (`should_offload(12000)` false, `(12001)` true).

Outside Docker the tests take their trees from the environment, so they run anywhere torch (CPU), tokenizers 0.23.2 and TabbyAPI's CPU dependencies are installed:

```sh
export TOK_EXL3=<exllamav3 package dir with exl3.patch applied> TOK_APP=<TabbyAPI /app with app.patch applied> \
       TOK_EXL3_BASE=base/exllamav3/tokenizer/tokenizer.py
for t in test_identity test_lock test_gil test_event_loop test_hops; do python3 tests/$t.py; done
python3 tests/test_identity.py --tokenizer-dir <a model directory with tokenizer.json>   # also on a real tokenizer
```

`tests/run_offline.sh` builds those trees from the private repository's layout and is kept byte-identical to the served image's copy, because `SHA256SUMS.tests` hashes it and the image label hashes `SHA256SUMS.tests`.

## Files

- [`Dockerfile.box`](Dockerfile.box): FROM `BASE`. Checks `SHA256SUMS.tests` and that the base carries loop-think r5 and rebase-dev r3; hash-checks the installed files against `SHA256SUMS.*.base`, dry-runs, applies both patches with `--fuzz=0 --forward` and hash-checks the result against `SHA256SUMS.*.src`; then runs the landing check, the offline tests and the served CPU suite.
- `base/exllamav3/tokenizer/tokenizer.py`: the served ExLlamaV3 file, used by the identity tests as the reference. It is upstream ExLlamaV3 `dev` `5783a93`'s file (MIT); `docker/overlays/rebase-dev-r3/ported-vs-dev.patch` does not touch it.
- `app.patch`, `exl3.patch`: `diff -ruN`, `-p1` in `/app` and in the installed package directory. `SHA256SUMS.app.base`, `SHA256SUMS.app.src`, `SHA256SUMS.exl3.base`, `SHA256SUMS.exl3.src`: the touched files before and after. `SHA256SUMS.tests`: `tests/` and the landing.
- [`mkpatch.sh`](mkpatch.sh): regenerates both patches and the five `SHA256SUMS.*` files from `base/` and `src/`, and checks that each patch applies at fuzz 0 and reproduces `src/`. `base/app/` and `src/` hold TabbyAPI files and are not in this repository; the script's header says how to rebuild them. Never hand-edit the patches.
