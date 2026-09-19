# CLAUDE.md — Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

This file is the working agreement for this repository. Read it before changing anything here.

## What this repo is

Serving configuration, launcher, instruments and measurements for `qwen3.8-flash-next-exl3-3.05bpw` served by
**TabbyAPI on top of ExLlamaV3** on the two RTX 5090 cards in the `flan` box, port **8022**. The LLM stack on
that box has a second home in `kubernetes-home` (`flan/launch-flashnext-tabby.sh` is a mirror of
`scripts/launch-flashnext.sh` here, `flan/r338-tabby-sampling.md` holds the first results, `flan/docker/` holds
the image recipe). The two must never disagree about the served configuration.

## Visibility — read this before pushing anywhere

This repository is prepared for publication on GitHub as `qwen3.8-flash-next-2x-rtx5090` (the user calls it the new public repo, 2026-09-17 and 2026-09-19). It has not been pushed. Creating the GitHub repository and every push are the user's call: never push without an explicit request.

- Treat every commit as public already: GitHub keeps serving a commit by its SHA after a history rewrite. Run `scripts/check-public-hygiene.sh` before every commit and `scripts/check-public-hygiene.sh --tree` before the first push.
- Stage explicit paths; `git add -A` and `git add .` are forbidden.
- Do **not** mirror any of it into the public `qwen3.8-27b-rtx5090` repo; only the 27B vLLM daily belongs there. The llama.cpp work on this model stays in the private infrastructure repo.

## Prose (README, THIRD_PARTY.md, docs/)

Enforced by `scripts/check-prose.sh`, which `scripts/check-public-hygiene.sh` runs on every staged Markdown file.

- Declarative sentences. Each states what was measured, when, on which configuration, where the raw output is, and what it means.
- No evaluative or promotional words, no rhetorical devices, no exclamation marks, no questions in running text, no jokes, no asides about how the work felt. The banned list is in the script; extend it when a new one gets through.
- Numbers carry their conditions: kind (code or prose) for every decode rate, prompt tokens and tokens per second for every prefill figure, concurrency, forced length, sampler, results directory.
- Comparisons are between ExLlamaV3 (TabbyAPI) and vLLM on this checkpoint. No other model or daily is compared against or named in the measurements.
- A line that must keep a flagged word (a quoted error string, a proper name) carries `prose-ok: <reason>`.

## Operating rules for the box

- **`flashnext` and the 27B vLLM daily cannot coexist.** Both need both cards resident. Taking the box means the
  daily is down, and the daily must be restored when the box is handed back
  (`bash /srv/qwen5090/daily-restore-retry.sh`); decide that per run, never leave it down by accident.
- Any GPU-exclusive script on flan sources `/srv/qwen5090/lib/gpu-queue.sh` before its flock, so a chain of
  experiments never has to wait through a daily down/up.
- **Repo-first, then apply.** Change the launcher here, then apply it with
  `ssh flan 'bash -s' < scripts/launch-flashnext.sh`. No ad-hoc edits on the box: if it is not in this repo it
  did not happen.
- A restart costs about 21 s from launch to serving with warm kernel caches (2.50 bpw pack, 2026-09-19). The server is shared with the
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
| `scripts/launch-flashnext.sh` | the served launcher; writes the config and the sampler preset, then starts the container |
| `scripts/launchers/` | the launcher of each promotion and experiment |
| `scripts/r*.sh` | one driver per experiment, named after its R number |
| `docker/` | the image chain: Dockerfiles, patches, overlays with SHA-pinned installers |
| `bench/*.py` | the instruments |
| `bench/results/<rNNN-slug>.md` | one write-up per experiment: heading names the finding, first line names the results directory on the box, the driver and the raw records |
| `bench/results/<date>-<rNNN-slug>/` | raw records copied from the box (audit log as `audit.txt`, records JSONL, greedy captures); no tool-eval JSON, no files over 2 MB |
| `bench/RESULTS.md` | the index, newest first: one line per result file |
| `docs/CONFIG.md` | every setting and flag in the served config and why it has that value |
| `docs/HISTORY.md` | how the served configuration changed, one row per promotion |
| `docs/PROMOTION.md` | the gates |
| `docs/GOTCHAS.md` | the traps, in the form "what it looks like / what it is" |

## Sync with the private repo

The launcher here mirrors `flan/launch-flashnext-tabby.sh` in the private infrastructure repo (private addresses and home paths scrubbed); both change in the same session. A new result means a new file in `bench/results/`, its raw records next to it, one line in `bench/RESULTS.md`, and the README numbers table if it changes a served figure. Never append prose to the index.
