# CLAUDE.md — Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

This file is the working agreement for this repository, for people and coding agents alike (`AGENTS.md` is a link to it). Read it before changing anything here.

## What this repo is

Serving configuration, launcher, image recipe, kernel overlays, instruments and measurements for Qwen3.8-Flash-Next, served as `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab` by **TabbyAPI on top of ExLlamaV3** on the two RTX 5090 cards of the `flan` box, port **8022**. The served image and its tag are in `scripts/launch-flashnext.sh` (`DAILY_IMG`) and in the README's Served configuration.

The LLM stack on that box has a second home in a private infrastructure repository: its `flan/launch-flashnext-tabby.sh` is the source of `scripts/launch-flashnext.sh` here. The two must never disagree about the served configuration.

## Visibility — read this before pushing anywhere

This repository is public on GitHub as `adrienbrault/qwen3.8-flash-next-2x-rtx5090` since 2026-09-19. Its history was rewritten on 2026-09-19 to remove local paths, private addresses, session ids and personal e-mail addresses; commits use the GitHub noreply address (`git config user.email adrienbrault@users.noreply.github.com` in this checkout).

- Treat every commit as public already: GitHub keeps serving a commit by its SHA after a history rewrite. Run `scripts/check-public-hygiene.sh` before every commit and `scripts/check-public-hygiene.sh --tree` before the first push.
- Stage explicit paths; `git add -A` and `git add .` are forbidden. Another session may have uncommitted work in this checkout: never stage, revert or overwrite it.
- Commit messages carry no session links or tool attribution lines.
- Do **not** mirror any of it into the public `qwen3.8-27b-rtx5090` repo; only the 27B vLLM daily belongs there. The llama.cpp work on this model stays in the private infrastructure repo.
- Boot logs are never published (the launcher prints the LAN address); raw records that echo a private-looking address from a dataset are stripped of the generated text rather than exempted.

## README

The README is read top-down by someone who wants to know what is served and how fast it is. Its structure is fixed; a promotion or a new measurement updates it **in place**.

**Top half = the served state only.** The intro, `## Numbers` (with `### Conditions`), `## Served configuration` and `## What the stack is` describe the configuration that is served now, with the numbers measured on it. They never contain:
- before/after comparisons with a previous image or date ("was X, now Y", "on the previous image", "(0.978 on 2026-09-20)", "983,040 until 2026-09-27");
- the story of a promotion, a rollback, a regression scare or how a question was settled;
- instrument caveats, review findings or decompositions of a discrepancy.

Those go where they belong: the promotion's row in `docs/HISTORY.md`, the round's write-up in `bench/results/`, a trap in `docs/GOTCHAS.md`. The top half may link to them in one clause.

**`## Numbers` layout**, in this order and no longer:
1. One paragraph: which image, the served configuration, when it was measured, the round and its results directory, the instrument in one sentence, and what "rate" and "aggregate" mean.
2. The decode figure, then one short paragraph saying what the solid and dashed lines are.
3. At most five bullets, one or two sentences each.
4. The prefill figure, then at most two sentences.
5. The value / source table, then the one-line "Also passing" list.

A number that needs a paragraph of explanation belongs in the write-up; the README states it with its conditions and links there.

**Decode metrics**: per-stream decode rate and decode aggregate (the sum of the decode rates of the streams running together) are the headline; time to the first token is separate; a round-wall figure appears only as a labelled secondary. Never per-stream × N as an aggregate.

**Figures** are drawn by `bench/plot.py` from raw records committed under `bench/results/`; no figure carries a number that is not in this repository. `plot.py` raises when an input file is missing; it never falls back to an older round.

The lower sections (`## Measured and not served`, `## How the numbers are measured`, `## Standard benchmark`, `## Reproducing a boot`) carry method detail and may name earlier rounds where the method comes from them.

## Prose (README, THIRD_PARTY.md, docs/)

Enforced by `scripts/check-prose.sh`, which `scripts/check-public-hygiene.sh` runs on every staged Markdown file.

- Declarative sentences. Each states what was measured, when, on which configuration, where the raw output is, and what it means.
- No evaluative or promotional words, no rhetorical devices, no exclamation marks, no questions in running text, no jokes, no asides about how the work felt. The banned list is in the script; extend it when a new one gets through.
- Numbers carry their conditions: kind (code or prose) for every decode rate, prompt tokens and tokens per second for every prefill figure, concurrency, forced length, sampler, results directory.
- Comparisons are between ExLlamaV3 (TabbyAPI) and vLLM on this checkpoint, or between two configurations of this stack in one session. No other model or daily is compared against or named in the measurements.
- One paragraph is one line in the source; no manual wrapping.
- A line that must keep a flagged word (a quoted error string, a proper name) carries `prose-ok: <reason>`.

## Operating rules for the box

- **`flashnext` and the 27B vLLM daily cannot coexist.** Both need both cards resident. Taking the box means the daily is down; decide per run whether and when it comes back, never leave it down by accident.
- Any GPU-exclusive script on flan sources `/srv/qwen5090/lib/gpu-queue.sh` before its flock, so a chain of experiments never has to wait through a daily down/up; only the last unit of a chain restores the daily.
- **Repo-first, then apply.** Change the launcher in the private repo, apply it on the box from that file, and mirror it here in the same session. No ad-hoc edits on the box.
- A restart costs about 20 s from launch to serving with warm kernel caches. Agent clients (Hermes, omp through the Olla gateway, Open WebUI) call :8022; a unit that measures on :8022 stops or drains them first and restores them after.

## Measurement rules

These are not style preferences. Each one is a mistake that has already produced a wrong number here.

1. **Force the length.** `max_tokens` alone measures the model's stopping behaviour, not throughput: on this checkpoint a bare `/completions` prompt at T=0 ends on EOS after 1–430 tokens. `ignore_eos`/`ban_eos_token` is accepted by TabbyAPI and **ignored** by the exllamav3 backend. Use **`min_tokens`**, which reaches `Job(min_new_tokens=...)` and suppresses EOS.
2. **A forced length can still stop early — record the finish reason.** exllamav3 has a loop detector; a forced 16,384-token greedy code generation stopped at 7,107 with `loop detected`. A run whose `finish_reason` is not `length` is not a throughput sample.
3. **Report the conditions with the number, always:** kind (code vs prose), concurrency, forced length, prompt depth, sampler, and the results directory. Decode rate here is content-dependent by ~2×.
4. **Token counts.** Trust `usage.completion_tokens` only on an image with the requeue token-count fix (since R747). TabbyAPI returns `"usage": null` on non-streamed chat completions: count streamed with `include_usage`, or with `/v1/token/encode`. `bench/probe.py` records the server's count and the client's frame count side by side; frames must never exceed tokens.
5. **Warm the batch shape.** The engine compiles kernels per batch shape: an unwarmed first round at a new concurrency has read 10× low.
6. **Compare in one session.** Two boots of one configuration differ by 1.5–2 % at 2–8 streams and up to 20 % at 1 stream; a comparison across days or images is not evidence of a change. A claim of change comes from alternating boots in one session (ABBA), with the spread stated.
7. **Separate step time from acceptance.** With the MTP draft on, decode rate = tokens per step / time per step. A rate that moves with the text (depth, one prompt, one sampled session) is acceptance, not speed; report tokens per step and ms per step when the two disagree.
8. **No summary may hide the work done.** `bench/probe.py` writes one JSONL line per request. A summary table is a convenience, never the record.

## Layout

| path | what it is |
| --- | --- |
| `scripts/launch-flashnext.sh` | the served launcher; writes the config and the sampler preset, then starts the container |
| `scripts/launchers/` | the launcher of each promotion and experiment |
| `scripts/r*.sh` | one driver per experiment, named after its R number |
| `docker/` | the image chain: Dockerfiles, patches, overlays with SHA-pinned installers |
| `bench/*.py` | the instruments, and `bench/plot.py` for the README figures |
| `bench/results/<rNNN-slug>.md` | one write-up per experiment: heading names the finding, first line names the results directory on the box, the driver and the raw records |
| `bench/results/<date>-<rNNN-slug>/` | raw records copied from the box (audit log as `audit.txt`, records JSONL, greedy captures); no tool-eval JSON, no boot logs, no files over 2 MB |
| `bench/RESULTS.md` | the index, newest first: one line per result file |
| `docs/CONFIG.md` | every setting and flag in the served config and why it has that value |
| `docs/HISTORY.md` | how the served configuration changed, one row per promotion, with the before/after numbers |
| `docs/PROMOTION.md` | the gates |
| `docs/GOTCHAS.md` | the traps, in the form "what it looks like / what it is" |

## Sync with the private repo

The launcher here mirrors the private `flan/launch-flashnext-tabby.sh` (private addresses and home paths scrubbed); both change in the same session. A new result means a new file in `bench/results/`, its raw records next to it, one line in `bench/RESULTS.md`, and — only if it changes a served figure — the README number updated in place. A promotion adds a row to `docs/HISTORY.md` and updates the README's served state in place. Never append prose to the index or to the top half of the README.
