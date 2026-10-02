# R828 prompt-lookup r3 (2026-10-02)

This is the historical pre-run design. Completed measurements, review corrections and promotion status are in [R827–R828c](../../../bench/results/r827-r828-prompt-lookup.md).

Implemented from the untouched `prompt-lookup-r2` packet and the served R825c source extraction. Binding rationale: `../REVIEW-R827.md`, findings 3, 6 and 7. R827b's accepted identity rule and measured speed rejection are inputs, not new claims. This session writes files and runs CPU checks; it does not build, deploy, contact endpoints, use SSH, or perform git operations.

## Served base and installation

The Dockerfile starts from `tabbyapi:r825c-hostprepare`, full parent ID `sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9`. The supplied served `exllamav3/` and `app/` files were independently SHA checked against every base and unchanged manifest entry. The supplied live environment matches `fixtures/live-container-env.txt`. The four existing-file overlays and pure helper install only after all base checks pass. `install.py --check` checks output and unchanged source hashes. The config schema and loader are explicitly included in the unchanged manifest. No extension rebuild is required.

The LIVE launcher remains md5 `262e9c31f714724409635fbac9df8ac2`, with 46 default selectors and 49 EXL3 environment entries. Candidate launchers share the same R828 full image ID, repin **R825C_IMAGE_ID**, and have 47 selectors/50 entries. The image name is `tabbyapi:r828-prompt-lookup-r3`; `/opt/r828` holds the packet. `IMAGE_ID.env` is intentionally unfilled. It must contain the operator's new R828 build ID; the R827 ID cannot be reused. The repository launcher template `../launch-flashnext-r828-lookup.sh` resolves this pin from the deployed packet. `gate.prepare` emits the fully resolved launcher before any experimental boot. The promotion unit writes that resolved repo artifact before installing LIVE.

## Eligibility and adaptive evidence

`EXL3_PROMPT_LOOKUP=0` follows the served tensor, forward, sampler and rollback paths. The extra CPU counters are additive. No SAM or lookup readback occurs. `=1` requires fixed-depth MTP. Other selector values fail closed.

Eligibility uses the actual decode-ready job count and draft row count, both one, before matching or opener readback. Jobs still in prefill do not disqualify a singleton; a mixed decode batch performs no SAM work, no opener readback and no draft overwrite. A late singleton tail can qualify. No concurrent request-level claim of permanent ineligibility follows from its nominal c2/c3 workload.

Each logical request owns an AdaptiveLookup object, retained in requeue state alongside the independent SAM and counters. State decisions occur in `begin`, using only evidence from completed rounds. `finish` publishes the current round's evidence after target acceptance. Evidence is bounded to the most recent 64 eligible checked rounds, never an unbounded request average.

PROBE is observational probation: run every original MTP forward and keep its draft, including tail columns. A valid SAM source uses an early MTP opener readback to create a shadow chain. Score its prefix against target tokens the served verifier actually consumes. Stop credit after the first shadow mismatch; positions the MTP verifier does not reach count as unaccepted. This is a conservative lower bound, not counterfactual MTP-tail logits or acceptance. PROBE therefore incurs matching/readback overhead but cannot replace a stronger MTP tail or alter its verify input. The production acceptance/stop/checkpoint logic and target-forward/sampler call sites stay unchanged.

PROBE enters ON only with at least 24 hit rounds, hit density >=50%, and accepted/copied-shadow-proposed >=90%. It enters OFF after 64 checked probation rounds without qualifying. ON can skip the two depth-three tail forwards when the MTP opener matches the lookup continuation. Actual copied acceptance then supplies the evidence. Hysteresis exit thresholds are 40% hits and 85% acceptance; the density exit needs at least 24 checks and the acceptance exit needs at least 24 hit rounds. Transitions apply to the next round, never the round that established the evidence.

OFF checks once every 32 eligible steps (its initial OFF step is a check). Its SAM work and would-be opener test occur only **after** the normal completed whole-window host readback. It adds no device synchronization and never proposes or overwrites. Four sparse observations with >=60% would-be hits admit a fresh PROBE, with no copied proposal on that re-enable decision. Shadow acceptance must qualify anew before ON. Sparse evidence is bounded to eight samples. Ineligible steps count as state residency but do not advance checked probation. They advance the OFF cadence; dense mixed traffic can delay the next eligible re-probe. All state survives physical requeue.

| Environment suffix after EXL3_PROMPT_LOOKUP_ | Default | Meaning |
| --- | ---: | --- |
| MIN_HIT | .5 | Entry hit density |
| MIN_ACC | .9 | Entry copied-tail acceptance |
| WINDOW | 64 | Rolling checked evidence / probation bound |
| MIN_PROPOSALS | 24 | Minimum hit rounds, not tail tokens |
| REPROBE | 32 | Sparse OFF cadence |
| HYST_HIT | .1 | Entry minus exit hit margin; also OFF admission margin |
| HYST_ACC | .05 | Entry minus exit acceptance margin |

The five requested thresholds and both hysteresis margins are env tunable. Invalid fractions, nonpositive limits, and proposal minimum above window fail closed for ON. The registered gate uses defaults and rejects extra runtime env. No minimum-match change is made without match-length measurements: minimum three, longest previous suffix, full current draft width, generated-token guard at three, multimodal exclusions, suffix equality and remaining-token bounds are retained. The opener remains MTP and is excluded from copied credit. EOS, checkpoint, banned-string rewind, MTP acceptance repair, recurrent history, KV page rollback, rows32 capacity, prefill pipeline and cache allocation retain the served paths.

## Counters

One unwrapped `R828_COUNTER` JSON record is emitted per logical request, joined to the real HTTP completion serial. Counters are scalar finish fields and survive Tabby's existing stream merger and physical requeue. Existing decode/check/hit/ proposed/accepted counters distinguish actual ON proposals from shadow evidence. New fields include:

- Valid SAM match lengths in buckets 3–7, 8–15, 16–31, >=32, including OFF re-probes.
- Opener mismatches; all-hit and mixed proposal rounds; skipped draft forwards.
  Mixed proposal rounds must be zero under the batch-one gate.
- Accepted draft counts by `mtp`/`copy` source and zero-based positions 0–2.
  Copied position zero is always zero.
- PROBE/ON/OFF decode-step residency and wall seconds, enable/disable/re-probe
  transitions, and ineligible steps. Wall time runs from CPU begin through round
  finish (including drafting where available); it is not CUDA kernel timing.
- Shadow hits, copied tokens proposed/accepted; OFF re-probe checks/would-be hits.

A completion's residency sum equals decode_steps. Copied accepted position counts sum to lookup_accepted. Skipped forwards equal copied proposed tail tokens under this singleton/fixed-depth design. Counters are published after the round's acceptance and before EOS results/requeue release; no GPU timing is inferred.

## Dry run, gate and promotion

`../r828-prompt-lookup-r3.sh` retains the queue-before-flock lifecycle, gateway drain, bounded cleanup, runtime/config/env/source checks, clock checks, identity clients and fn/agent speed request geometry. The dry run keeps the existing daily running and tests real short fn/chat/edit/fn-style streams and speed parsers. Candidate counter/gate paths run through explicitly synthetic CPU fixtures.

Before dry PASS, **both generated candidate launchers** run in `R828_PREBOOT_ONLY=1` mode: their original image-present and full R825C_IMAGE_ID pin checks execute; generated YAML is validated by the unchanged Tabby schema inside the existing daily with `docker exec`. No container is run, created, started or stopped by this guard path. Config and sampler outputs go under the results preflight directory, not the live config. Normal boot retains its original image-local `config.load({})` guard. Dry PASS is bound to packet/unit hashes, LIVE md5, candidate image ID and 24-hour freshness. It does not certify candidate image contents: GPU stage CPU landing and installed selftest do that before experimental boots.

Boot order: OFF1 / ON1 / ON2 / OFF2, one common image and only selector 0/1 differs. Original identity set is fn 6 + chat 6 + edit 12. OFF2 must match OFF1 24/24; OFF/OFF divergence is **INVALID**, writes no promotion eligibility, and the per-boot check exits 3 so the unit fails immediately. Each ON may diverge on at most 2/24. fn+chat requests whose adaptive ON residency is zero must match OFF 12/12: a divergence on such a request rejects regardless of the two-flip allowance. An extra batch-one fn-style code/prose set (six each, forced 256 tokens) follows the same conditional exactness rule. Its OFF/OFF divergence also invalidates. Per-request state residency is joined and recorded in the identity JSONL.

The unchanged speed gate requires agent c1 >=+2% for **each ON against each OFF**, and each fn c1/c2/c3 code/prose shape >=-1% in its ON1/OFF1 and ON2/OFF2 pair. Every ON edit identity and speed set must show actual lookup proposals/acceptance. Missing/foreign rows, invalid streams, accounting, environment or clocks invalidate. No pooling can conceal a failed comparator. Final decisions are PASS, REJECT, or INVALID, with the user-approved two-flip rule unchanged.

On PASS the unit promotes without another confirmation (pre-approved 2026-10-02): write the resolved repo launcher, copy LIVE to `.pre-r828`, write `LIVE.new`, then atomic `mv` to LIVE. Boot through clean env and run exact runtime/source/residency/ clock gates, FASTWARM success, fn c1 code paired >=-1%, exact greedy fn/chat and fn-style identity against OFF2, and each GPU's free VRAM >= current daily minus 32 MiB. The reference is captured under drain before experiments; if a chained GPU stage starts without the parent daily (absent or predecessor arm), use the bound dry run's daily reference. Any failure/signal before the promotion commit restores the launcher atomically and restores the old daily unless a queued successor owns the next experiment. After success cleanup keeps the promoted daily. Existing `.pre-r828` is preserved and causes a fail-closed refusal to overwrite the backup. Failed experiments restore only when last in the GPU queue; no daily down/up is inserted between arms.

## Replay and limits

`fixtures/r827b` contains verbatim real R827b ON1/ON2 counter files for every speed shape, selected counter/completion joins independently verified against the supplied full container logs, and original paths/SHA-256 provenance. `replay_r827b.py` consumes those real per-request totals, conserving every hit, proposal and accepted copied token in reconstructed schedules. R827b did **not** record chronological per-step traces. An exact no-lookahead replay of the original hit order is impossible; the report explicitly labels reconstruction and proxy acceptance assumptions. It cannot reconstruct actual c2/c3 singleton tails.

`R827B-REPLAY.json` records each request's decision/residency for balanced, front-loaded, back-loaded and ten shuffled schedules. Balanced proxy predictions are **95.198% agent steps ON, 0% c1 prose, 0% c1 code**. The tested scheduling range is 92.718–95.391% agent, 0% prose, 0–6.061% code. This is a sensitivity range over chosen schedules, not a confidence interval or guaranteed bound. r3's conservative PROBE scoring can reduce the agent activation fraction relative to r2 copied-acceptance proxies. GPU identity/speed and actual ON fraction remain unmeasured. Synthetic replay additionally proves low-acceptance prose never ON, agent transitions ON, OFF2 flip INVALID with nonzero intermediate exit, two ON flips eligible, three REJECT, and conditional OFF-path flips REJECT.

CPU checks execute production helpers and the actual served/overlay MTP loops on host/device doubles, assert no mixed-batch SAM/readback, OFF re-probe no opener sync, observational PROBE output, no-lookahead, hysteresis, requeue persistence, counters, strict wire parsing, real-counter conservation, and promotion failures. Mocked launcher tests execute its guard path and reject image-pin/config failures without contacting a Docker daemon. AST checks retain served target forwards, samplers, prefill/repair call sites, device mode and rollback.

Estimated GPU gate: **60–70 min, <=75 min budget**, plus promotion **~10 min**. Queue waiting and pre-existing gateway drain are outside that estimate. No GPU validation is claimed by the local CPU packet.