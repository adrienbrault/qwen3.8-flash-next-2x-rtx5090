# CLAUDE.md — Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

This file is the working agreement for this repository. Read it before changing anything here.

## What this repo is

Serving configuration, launcher, instruments and measurements for `qwen3.8-flash-next-exl3-3.05bpw` served by
**TabbyAPI on top of ExLlamaV3** on the two RTX 5090 cards in the `flan` box, port **8022**. The LLM stack on
that box has a second home in `kubernetes-home` (`flan/launch-flashnext-tabby.sh` is a mirror of
`scripts/launch-flashnext.sh` here, `flan/r338-tabby-sampling.md` holds the first results, `flan/docker/` holds
the image recipe). The two must never disagree about the served configuration.

## Visibility — read this before pushing anywhere

The **Flash-Next / llama.cpp / ExLlamaV3 track is NOT public** (user, 2026-09-12; recorded in the sibling
`kubernetes-home/AGENTS.md`). Concretely:

- Do **not** push this repository anywhere public, and do not create a public repository for it.
- Do **not** mirror any of it into the public `~/Developer/ai/qwen3.8-27b-rtx5090` repo. Only the 27B vLLM daily
  belongs there.
- The 27B vLLM daily's launcher (`qwen3.8-27b-rtx5090/scripts/serve-*.sh`) and this launcher are different
  engines and different worlds. `launch-flashnext.sh` in `kubernetes-home/flan/` is the *llama.cpp* variant of
  this model; it is not this script.

## Operating rules for the box

- **`flashnext` and the 27B vLLM daily cannot coexist.** Both need both cards resident. Taking the box means the
  daily is down, and the daily must be restored when the box is handed back
  (`bash /srv/qwen5090/daily-restore-retry.sh`); decide that per run, never leave it down by accident.
- Any GPU-exclusive script on flan sources `/srv/qwen5090/lib/gpu-queue.sh` before its flock, so a chain of
  experiments never has to wait through a daily down/up.
- **Repo-first, then apply.** Change the launcher here, then apply it with
  `ssh flan 'bash -s' < scripts/launch-flashnext.sh`. No ad-hoc edits on the box: if it is not in this repo it
  did not happen.
- A restart costs ~11 s of model load (warm kernel caches) plus a ~0.3 s warmup. The server is shared with the
  user's DSH sessions — warn before restarting unless the box has been handed over.

## Measurement rules

These are not style preferences. Each one is a mistake that has already produced a wrong number here.

1. **Force the length.** `max_tokens` alone measures the model's stopping behaviour, not throughput: on this
   checkpoint a bare `/completions` prompt at T=0 ends on EOS after 1–430 tokens. `ignore_eos`/`ban_eos_token`
   is accepted by TabbyAPI and **ignored** by the exllamav3 backend. Use **`min_tokens`**, which reaches
   `Job(min_new_tokens=...)` and suppresses EOS.
2. **A forced length can still stop early — record the finish reason.** exllamav3 has a loop detector; a forced
   16,384-token greedy code generation stopped at 7,107 with `loop detected`. A run whose `finish_reason` is not
   `length` is not a throughput sample.
3. **Report the conditions with the number, always:** kind (code vs prose), concurrency, forced length, prompt
   depth, sampler, and the results directory. Decode rate here is content-dependent by ~2× (see below).
4. **Trust `usage.completion_tokens` only on an image with the R338 requeue fix.** The unpatched engine
   under-reports any generation past the ~2048-token requeue boundary by ~5×, and the same number feeds the
   logged T/s. `bench/probe.py` records the server's count and the client's frame count side by side; frames must
   never exceed tokens.
5. **Warm the batch shape.** The engine compiles kernels per batch shape: an unwarmed first round at a new
   concurrency has read 10× low.
6. **No summary may hide the work done.** `bench/probe.py` writes one JSONL line per request. A summary table is
   a convenience, never the record.

## Layout

| path | what it is |
| --- | --- |
| `scripts/launch-flashnext.sh` | the launcher; writes the config and the sampler preset, then starts the container |
| `bench/probe.py` | decode/concurrency/depth instrument: forced length, per-request records |
| `bench/results/` | dated results, one file or directory per run, with the raw records next to the prose |
| `docs/CONFIG.md` | every setting in the served config and why it has that value |
| `docs/MEASUREMENTS.md` | the measured numbers, each dated and tied to a results directory |
| `docs/GOTCHAS.md` | the traps, in the form "what it looks like / what it is" |
