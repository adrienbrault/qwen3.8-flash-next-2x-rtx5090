# R827–R828c: adaptive prompt lookup beside MTP increases copy-edit decode rate at one stream

Results on flan, 2026-10-02: [`2026-10-02-r827-prompt-lookup-ygmGkW`](2026-10-02-r827-prompt-lookup-ygmGkW), [`2026-10-02-r828-prompt-lookup-r3-em6iO4`](2026-10-02-r828-prompt-lookup-r3-em6iO4), [`2026-10-02-r828c-c3-confirm-d8bApo`](2026-10-02-r828c-c3-confirm-d8bApo); drivers [`r827-prompt-lookup.sh`](../../scripts/r827-prompt-lookup.sh), [`r828-prompt-lookup-r3.sh`](../../scripts/r828-prompt-lookup-r3.sh), [`r828c-c3-confirm.sh`](../../scripts/r828c-c3-confirm.sh); request JSONL, per-request counter JSON, completion/counter log extracts, decisions, config/runtime extracts and `audit.txt` are in those directories. [`r827_r828_summary.py`](../r827_r828_summary.py) recomputes the tables from the published rows.

Promotion completed at 19:10 UTC on 2026-10-02: `tabbyapi:r828-prompt-lookup-r3`, operator-built ID `dfaed2cb`, source launcher MD5 `fb4e9d66`, 47 launcher selectors including `EXL3_PROMPT_LOOKUP=1`. The parent and rollback are `tabbyapi:r825c-hostprepare`, ID `aa04a1cb`, source launcher `262e9c31`; the public predecessor is archived as [`launch-flashnext-r825p-wholeprompt.sh`](../../scripts/launchers/launch-flashnext-r825p-wholeprompt.sh). Pool 901,120, 8 slots, 8-bit KV, split `[30,30]`, draft split `[0,32]`, policy `[[4,3],[8,2]]`, trace on, NVMe off and FASTWARM remain the served settings.

## Origin and installed mechanism

Prompt lookup beside an MTP draft is the idea in peonist-ai's [`halogen-flash-server` commit `36988cb`](https://github.com/peonist-ai/halogen-flash-server/commit/36988cb), cited in the private backlog. R1 was a Codex implementation of that idea, measured in R501; R827's r2 ports it onto the R825c installed tree. R3's adaptive trigger was written for this repository from R827's measured failure and review. No implementation from that upstream commit is claimed as copied; the ExLlamaV3 and TabbyAPI source overlays retain their respective MIT and AGPL-3.0 derivation ([THIRD_PARTY](../../THIRD_PARTY.md)).

The MTP draft supplies the opener. A previous suffix match of at least three tokens supplies the remaining draft columns only when the opener agrees, with the original verify width retained. An all-hit singleton round skips the remaining two depth-three MTP forwards; ordinary target verification, sampling, recurrent rollback, MTP cache repair and stopping still decide acceptance. The installer checks the four replaced files, new pure-Python helper and unchanged source manifests; there is no native rebuild or extra GPU cache allocation.

R3 checks actual decode-ready job count and draft row count before matching: both must be one. Probation observes full MTP drafts without replacing their tails. ON requires at least 24 hit rounds, at least 50% hits and at least 90% shadow-tail acceptance in a rolling 64 checked-round window; 64 unsuccessful probation rounds go OFF. Exit uses 40% density and 85% acceptance with minimum evidence. OFF re-probes every 32 steps using the existing completed draft-window readback; it adds CPU indexing/matching but no device synchronization. Four sparse observations with at least 60% hits reopen probation. State survives requeue; mixed batches retain MTP, while late singleton tails can perform probation checks.

## Conditions and registered rules

R827b ran 15:10–15:45 UTC and R828 ran 17:58–18:35 UTC on 2026-10-02. Each uses one candidate image for OFF1 / ON1 / ON2 / OFF2, differing only in the lookup selector. R827 uses `tabbyapi:r827-prompt-lookup`, ID `aa9ebe76`; R828 uses the promoted r3 image. Core offsets were zero, memory offsets +4,500 MHz on both cards, power limits 600 / 575 W. These are policy readbacks at boot start/end, not continuous frequency traces.

Fn speed is code / prose on 118 / 106-token short distinct prompts, concurrency 1, 2 and 3, greedy, 1,024 forced output tokens, one short batch-shape warmup and one full warmup followed by three measured rounds. Agent speed is code-file rewrite prompts at 1 stream, 1,832–8,646 input tokens, thinking off, greedy, 2,048 forced output tokens; twelve prompts repeated three times after twelve warmups, 36 measured requests per boot. Every retained speed request finishes as `length`; frames do not exceed completion tokens. Each table rate is the median measured request decode rate after its first token, not a round-wall rate or concurrency-multiplied aggregate. Counter metrics use ratios of sums over measured requests and server generation wall time, not CUDA event timing.

The speed rule requires every agent ON/OFF comparator to exceed +2%, and each paired fn shape to remain at or above −1%. The identity set is six fn, six chat and twelve code edits; edits force 3,072 tokens, the long chat 5,000, testing requeue. R827b's rule, registered after the original R827 identity failure and before R827b, requires OFF2 24/24 equal and permits at most two ON divergences. It does not retroactively pass R827's earlier exact-identity run. R3 additionally requires exact equality for jobs with no adaptive ON residency, and adds twelve fn-style code/prose requests at batch one with 256 forced greedy tokens.

## R827b: unrestricted lookup rejected

R827b reproduces the agent gain but fails five of six fn shapes on at least one paired comparison. It remains REJECT; R2 is not served. The directory `2026-10-02-r827-prompt-lookup-ygmGkW` contains each boot's `fn-*.jsonl`, `speed-*-agent.jsonl` and `counters-*.json`.

| Shape | OFF1 t/s | ON1 t/s | ON2 t/s | OFF2 t/s | ON1/OFF1 | ON2/OFF2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1-code | 265.260 | 265.930 | 266.160 | 265.660 | +0.253% | +0.188% |
| c1-prose | 287.690 | 280.230 | 280.740 | 287.430 | -2.593% | -2.328% |
| c2-code | 214.095 | 213.150 | 214.305 | 216.865 | -0.441% | -1.180% |
| c2-prose | 210.345 | 207.650 | 204.045 | 209.925 | -1.281% | -2.801% |
| c3-code | 178.750 | 175.580 | 177.110 | 178.030 | -1.773% | -0.517% |
| c3-prose | 176.340 | 173.160 | 173.070 | 176.710 | -1.803% | -2.060% |
| agent-c1 | 390.149 | 417.918 | 417.581 | 390.001 | +7.118% | +7.072% |

The agent cross-pair gains are +7.158% and +7.031%. Agent ON records 18,279 hits / 18,744 steps per boot, 36,483 / 36,558 copied tokens accepted (99.795%); tokens per step rise from 3.88369 to 3.93342 while paired ms per step fall from 10.04488 / 10.02719 to 9.49592 / 9.50525. The main gain is consistent with skipping tail drafting; wall counters do not isolate CUDA savings. Fn c1 prose copied acceptance is 156/396 (39.394%); tokens per step fall 2.82094 → 2.72340. Weaker proposals explain that acceptance loss, but not every concurrent loss: c3 prose ON1 increases tokens per step while its step time increases. Per-job hits alone do not establish zero all-hit rounds in mixed batches.

ON1 and ON2 match 23/24 original identities, OFF2 24/24. The changed edit is `edit-3-1`, whose inserted comment moves from before the last line to after it. The actual first content difference is offset 1,978; the gate's 2,120 is in sorted escaped JSON. A numerical near tie is an inference without logits or margins. Forced-length rewrite outputs repeat and truncate; the workload measures copying behavior, not general edit correctness. R2's OFF/OFF instability path reports REJECT where its header promises INVALID; R3 repairs that distinction.

## R828: adaptive trigger, original rejection retained

R828's code-edit c1 medians and all four agent gains pass; fn c1/c2 and c3 prose stay between −0.313% and +0.262%. C3 code ON2/OFF2 is −1.220%, below the registered floor. R828 therefore remains REJECT in `2026-10-02-r828-prompt-lookup-r3-em6iO4/decision.json`; R828c does not rewrite it.

| Shape | OFF1 t/s | ON1 t/s | ON2 t/s | OFF2 t/s | ON1/OFF1 | ON2/OFF2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1-code | 265.150 | 264.320 | 264.980 | 264.740 | -0.313% | +0.091% |
| c1-prose | 286.740 | 287.490 | 286.120 | 286.970 | +0.262% | -0.296% |
| c2-code | 216.600 | 216.875 | 216.465 | 216.555 | +0.127% | -0.042% |
| c2-prose | 209.795 | 209.660 | 209.375 | 209.600 | -0.064% | -0.107% |
| c3-code | 176.240 | 178.990 | 177.360 | 179.550 | +1.560% | -1.220% |
| c3-prose | 176.450 | 176.700 | 176.550 | 176.250 | +0.142% | +0.170% |
| agent-c1 | 389.854 | 414.443 | 414.994 | 389.256 | +6.307% | +6.612% |

Agent cross-pairs are +6.470% and +6.448%. Each ON boot records 18,783 measured steps: 951 probation and 17,832 ON (94.937%), 36 enables, zero disables/re-probes, first ON at round 27 or 28. There are 17,427 all-hit rounds, 34,854 skipped tail forwards, and 34,776 / 34,854 copied tokens accepted (99.776%). Tokens per step rise 3.88369 → 3.92525; paired ms per step fall 10.05511 / 10.06097 → 9.55702 / 9.55241, reductions 4.954% / 5.055%. Of 18,654 valid matches including probation, 18,210 (97.620%) are at least 32 tokens. These server-wall ratios and client medians weight requests differently.

All six fn speed shapes record zero ON residency, copied proposals, copied acceptance and skipped forwards. This supports retaining R826's fn curve with lookup inactive; R828 repeats concurrency 1–3, not the entire concurrency 1–8 curve or the standard dataset matrix. Nominal concurrency does not imply zero lookup work: c3 code ON1/ON2 has 69/52 singleton checks and 6/8 inferred opener reads, with 98.017% / 98.495% of steps ineligible. Probation and sparse OFF checks preserve MTP proposals but consume CPU time; they are not cost-free.

OFF2 / OFF1 c3-code medians differ by +1.878%, exceeding the −1% gate magnitude. ON2/OFF2 pooled tokens per step are 2.66744 / 2.70661, a 1.447% acceptance loss; ms per step fall 0.345%. Scheduling/acceptance variability is consistent with the failed comparator, but two pairs neither establish a distribution nor exclude probe overhead. Independent review called for a prospective confirmation, retaining the original rejection.

Original identities are 24/24 for each ON and OFF2 versus OFF1, plus fn-style 12/12 each. All 36 measured edit speed outputs in each comparison also equal OFF1. No-ON fn/chat/fn-style requests remain exactly equal in this set; this is conditional measured identity, not a claim of general numerical identity.

## R828c: eight new c3-code pairs and separate confirmation

The rule was registered before data: eight fresh independent boot pairs, OFF→ON / ON→OFF alternating; the same candidate image, thresholds, 118-token code prompts, concurrency 3, greedy, 1,024 forced outputs, shape/full warmups and nine measured requests per boot. Natural singleton tails remain present. The two original R828 pairs are excluded from the confirmatory estimate, with no interim stop, extra pairs, best-pair pooling or discarded valid slow boots. Results are in `2026-10-02-r828c-c3-confirm-d8bApo`, including `order.json`, all sixteen `fn-P*-c3-code.jsonl` files and per-request counters.

For each pair, g = log(median ON / median OFF). The one-sided 95% paired-t lower bound is mean(g) − 1.8946 × sample sd(g) / √8; PASS requires it strictly above log(0.99). The boot pair is the replicate. This small-sample bound assumes independent, approximately normal pair log gains; power is not established.

| Pair | Order | OFF t/s | ON t/s | g | ON checks / inferred reads | ON / OFF ineligible steps | ON / OFF ms per step | ON / OFF tokens per step |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | OFF/ON | 178.77 | 179.47 | +0.003908000 | 69 / 6 | 3411 / 3480 | 15.055745 / 15.117841 | 2.648276 / 2.648276 |
| 2 | ON/OFF | 177.44 | 177.69 | +0.001407935 | 69 / 6 | 3411 / 3455 | 15.223008 / 15.267452 | 2.648276 / 2.667438 |
| 3 | OFF/ON | 180.05 | 178.61 | -0.008029932 | 31 / 7 | 3444 / 3475 | 15.234779 / 15.123100 | 2.652086 / 2.652086 |
| 4 | ON/OFF | 178.94 | 178.78 | -0.000894554 | 48 / 5 | 3452 / 3455 | 15.188591 / 15.176842 | 2.633143 / 2.667438 |
| 5 | OFF/ON | 178.04 | 179.49 | +0.008111252 | 69 / 6 | 3411 / 3500 | 15.121002 / 15.150135 | 2.648276 / 2.633143 |
| 6 | ON/OFF | 178.67 | 177.68 | -0.005556349 | 31 / 7 | 3444 / 3500 | 15.308047 / 15.174213 | 2.652086 / 2.633143 |
| 7 | OFF/ON | 178.92 | 181.44 | +0.013986242 | 18 / 12 | 3387 / 3500 | 15.141435 / 15.149060 | 2.706608 / 2.633143 |
| 8 | ON/OFF | 177.63 | 173.03 | -0.026237745 | 27 / 4 | 3493 / 3455 | 15.177385 / 15.250789 | 2.618182 / 2.667438 |

Mean g = −0.001663144059, sample sd = 0.012195467937, lower bound = −0.009832183788, floor = −0.010050335854: PASS, margin 0.000218152065 in log-ratio units. Pair eight's −2.590% individual gain remains included. Every confirmatory ON boot records zero active lookup proposals and ON residency; inferred reads come from valid matches during observational probation. The gain order above is recomputed from raw: pairs six and seven are −0.005556349 and +0.013986242 respectively.

The promotion step rechecked the inherited non-c3 speed, identity and activation gates and then booted the resolved launcher. Post-promotion `promotion-gates.json` records free VRAM 1,099 / 1,667 MiB, equal to reference; fn c1 code with a 118-token prompt and 1,024 forced greedy tokens has median 265.13 t/s versus the same-session OFF2 264.74, ratio 1.001473; fn 6/6, chat 6/6 and fn-style 12/12 identities. Landing, clock, residency and runtime assertions pass. Source launcher pin is literal, so the promoted daily does not require an experiment-directory image pin file.

## Invalid attempts, dry runs and remaining coverage

R827 try one retained the parent's literal image-ID guard after substituting the candidate tag and aborted before serving. The source reports that failure; its raw directory was not supplied. The next exact-identity attempt in [`2026-10-02-r827-prompt-lookup-UO6HXQ`](2026-10-02-r827-prompt-lookup-UO6HXQ) invalidated on the same edit divergence later counted under the prospective R827b rule. [`2026-10-02-r827-dry-N3wl7g`](2026-10-02-r827-dry-N3wl7g) retains the successful no-restart dry parser/request records.

R828 try one read the generator in `Job.__init__`, before TabbyAPI attached it, failing even with lookup off. [`2026-10-02-r828-diag/failure.json`](2026-10-02-r828-diag/failure.json) extracts the reproduced AttributeError; adaptive initialization moved to `prepare_for_queue` with an AST regression check. Try two's OFF assertion counted passive ineligible steps; the parser was corrected, but its complete raw directory was not supplied. Dry records [`2026-10-02-r828-dry-joH7JQ`](2026-10-02-r828-dry-joH7JQ) and [`2026-10-02-r828c-dry-J4H9SI`](2026-10-02-r828c-dry-J4H9SI) bind unit/packet/image pins and preserve runtime/clock/VRAM assertions. Dry guards test the candidate launcher's image/config checks through the running parent; they do not execute candidate serving code.

Reasoning/tool/edit transitions, re-probe cost and hysteresis on live traffic remain unmeasured. High-density repeated prose can qualify, and copied acceptance does not measure foregone MTP acceptance or net wall-time benefit. Entry/exit thresholds permit temporary ON loss after a regime change; mixed batches retain state, so later singleton work can resume older evidence. The forced fn and repeated-code workloads do not establish a production latency distribution. An independent R828c review remains outstanding in the supplied findings.

The public records retain one line per measured request and all per-request counters, including every confirmatory boot. Container files here are only completion/counter metric extracts; no boot logs or full container logs are published. Inspect records retain image, environment, container continuity and restart count, omitting host/network metadata. Private deployment paths are adapted; generated text containing a private-looking address is removed rather than exempted. [`2026-10-02-r828-public-records.json`](2026-10-02-r828-public-records.json) lists copied and omitted files. Installed overlay bytes and their base/output manifests are unchanged. Public launcher fixtures have separate byte pins; deployment units retain historical on-box guards and require repinning adapted launcher/unit/packet bytes before reuse. The two CPU checks requiring the complete byte-frozen private packet archive are skipped locally; published scalar and counter replay is separate. Packet PORT documents retain their historical pre-run design, with a pointer to this completed result.
