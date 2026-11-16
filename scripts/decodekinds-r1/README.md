# decodekinds-r1

Driver and instruments of [`bench/results/decodekinds-r1.md`](../../bench/results/decodekinds-r1.md), as run on 2026-11-16.

- `run.sh`: the GPU unit. It takes the host's GPU lock and queue registration, drains the gateway, boots the served launcher twice on a test port at stock power, runs the four measurements per boot and restores the daily. Host helpers it sources (`gpu-queue.sh`, `gateway-drain.sh`, `daily-power.sh`, `serve-flashnext-daily.sh`) are the serving host's.
- `prep-dataset.py`: Spec-Bench `question.jsonl` plus 80 MBPP test prompts in Spec-Bench's jsonl format, and the prompt-sha to category map.
- `analyze.py`: the control gate and `summary.md`.
- `fn_bench.py`: the decode curve's probe as run here.
- `vllm_bench_tabby.py`: the `vllm bench serve` shim, identical to [`bench/vllm_bench_tabby.py`](../../bench/vllm_bench_tabby.py); it adds per-request reasoning-frame counts and TabbyAPI's accepted/rejected draft counts to the audit file.

- `server_steps.py`: per-request decode-step counters from the container log (`server-steps.jsonl`), the input of the step-based rate.
- `thinking_split.py`: thinking versus answer frames of the decode curve's two requests.

`run.sh` runs from the layout its installer builds: `vendor/fn_bench.py` (this folder's `fn_bench.py`), `vendor/r828/` (a copy of `docker/overlays/prompt-lookup-r3/`), `vendor/mia/` (MiaAI-Lab's `bench_decode.py` and a wrapper that runs it against a base URL and model) and `clock-readback.py` (an NVML read of clock offsets and power limits); the last two are not included. The MiaAI-Lab benchmark is run from its pinned upstream blob (AGPL-3.0) and is not included; R828's rewrite client is [`docker/overlays/prompt-lookup-r3/gate.py`](../../docker/overlays/prompt-lookup-r3/gate.py).
