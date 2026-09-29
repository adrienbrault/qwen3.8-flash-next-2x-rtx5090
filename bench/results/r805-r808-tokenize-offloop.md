# R805, R806, R808: a long streamed prompt stalled every decoding stream for ~53 ms while TabbyAPI encoded it twice on the event loop; `tabbyapi:tokenize-offloop-r2` encodes it once, off the loop above 12,000 characters, and is served with the output unchanged and no cost per decode step

Results on the serving host: `2026-09-29-r805-peer-gap-1035` (R805, 2026-09-29 10:37 to 10:42 UTC), `2026-09-29-r806-tokenize-offloop-1218` (R806, 12:18 to 12:43 UTC), `2026-09-29-r808-tokenize-offloop-r2-1411` (R808, 14:11 to 16:46 UTC) and `2026-09-29-r808p-promote-tokoffloop-1700` (R808p, the promotion, served from 17:01 UTC). Drivers: `r805-peer-gap.sh`, `r806-tokenize-offloop.sh`, `r808-tokenize-offloop-r2.sh` and `r808p-promote-tokoffloop.sh` with the probe `peer_gap.py`, not in this repository. Launcher: [`scripts/launchers/launch-flashnext-r808-tokoffloop.sh`](../../scripts/launchers/launch-flashnext-r808-tokoffloop.sh) = [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh), R792's launcher with only `DAILY_IMG` changed. Overlay: [`docker/overlays/tokenize-offloop-r2/`](../../docker/overlays/tokenize-offloop-r2/README.md). Raw records: on the serving host only; this write-up quotes the analyzers' summaries and an independent recomputation of every R808 gate from the raw per-event records.

## The instrument

- 4 peers decode 2,000 forced tokens each, streamed, and every chunk is timestamped on receipt. While they decode, single events fire 2 s apart: class (i), a 30,818-token prompt of which 30,720 tokens are cached, so the engine prefills 98 tokens; class (iii), a cold 98-token prompt, the same forward with a ~100-token front end; class (ii), the 30.8k-token prompt with nothing cached; and sham events, which send no request and give the null distribution.
- Per event and per peer: the baseline is the median gap between the peer's chunks outside every event window. The front-end stall (fe) is the excess over that baseline in the gaps that end between the event's submission and the server's log line for its prompt, which the server prints after tokenization. The total is the excess over every gap from submission to the event's first token. Each metric is the median over the valid peers.
- Sections A and B split fe at the one event-loop turn between the two synchronous steps of a streamed request (below).

## R805: the stall is two encodes of the prompt on the event loop (served image `tabbyapi:rebase-dev-r3-loopthink5`)

- 16 of 16 class (i) events stalled the 4 peers: fe median 52.8 ms (p10 51.1, max 77.3), at least 20 ms on every event. Class (iii): 2.8 ms. Shams: 1.5 ms. Class (ii), the same length with nothing cached: 55.1 ms, so the stall does not depend on the cache. 33 of 33 events joined to their server log line (median residual 1.0 ms); 0 failed requests.
- Fit over the events: fe = 3.9 ms + 1.66 µs per prompt token (the whole prompt, cached or not: both encodes run before the cache lookup).
- The stall is two blocks of about 25 ms (27.3 and 24.8 ms median), one event-loop turn apart:
  - A: the streamed router's `check_context_length` → `validate_context_length` in `backends/exllamav3/model.py`, a full encode used only for the prompt's length;
  - B: `generate_gen`'s encode of the same prompt before it builds the job.
- One encode costs about 0.8 µs per token. Parsing, the template render and JSON dumps take about 2.5 ms per 30k tokens. A non-streamed request skips the check and pays one encode.
- HF `tokenizers` `encode()` holds the GIL for the whole encode, so running it on a worker thread does not free the loop; `encode_batch()` releases it.
- The peers' median gap between chunks outside events was 17.3 ms (4 streams, temperature 1.0 peers). The share of wall time the stall costs scales with the rate of long-prompt arrivals; R805's synthetic rate of 0.49 arrivals per second is not a traffic measurement, so no decode gain on real traffic is stated here.

## R806: round r1 removes the stall, and costs short prompts 41.6 ms of time to the first token

Image `tabbyapi:tokenize-offloop-r1` (`0ca597dc44dd`): encode once (the check's ids reused by the job) and every encode on a one-thread worker through `encode_batch`, under a per-tokenizer lock. Four arms, one fresh boot each: both knobs on (BOTH), both off (NONE, the served code paths), encode-once only (ONCE), off-loop only (OFFLOOP).

- Class (i) fe: NONE 53.5, ONCE 29.7, OFFLOOP 4.7, BOTH 3.5 ms; section A 28.8 → 2.7 ms, B 24.7 → 0.7 ms (NONE → BOTH). Total excess per event 148.3 → 100.5 ms; the engine's part is unchanged (96.9 against 95.3 ms). BOTH's fe does not grow with the prompt length (5.0 ms + 0.00 µs per token).
- Output: BOTH equal to R792's greedy reference (`fn_greedy` 6 of 6, `chat_greedy` 6 of 6); BOTH streamed equal to NONE streamed on 12 of 12 rows, including a 120k-token prompt and n = 2; 864 of 864 encodes inside the container identical to the served tokenizer file's.
- Short prompts arriving during 4-stream decode (class (iii), ~98 tokens): time to the first token BOTH 255.9 against NONE 214.3 ms, +41.6 ms (OFFLOOP +80, ONCE +7.5). Each executor hop waits 1 to 2 decode steps: the loop picks the worker's result up only between two synchronous `iterate()` calls, and the worker needs the GIL that `iterate()` mostly holds. No gate covered this.
- Decode gate: the aggregate read −1.57 % against a −1.5 % bar. The time per decode step was unchanged (17.36 against 17.34 ms, −0.11 ± 0.27 %); the difference was MTP acceptance (51.47 against 52.45 % of proposed tokens) under temperature-1.0 peers with a different prompt nonce per arm, read by a median over cycles that picked warm-up cycles. One boot per arm cannot rule out a 1 % cost.
- Two gate defects: the stream-against-non-stream comparison failed on chat `long` in both arms with the same pair of outputs (a second-run effect, not the patch); the server token check compared against `/v1/token/encode` with its default `add_bos_token=True`, where the chat path adds no BOS (20,993 against 20,992).
- Not promoted. Round r2 hops only above a length threshold; R808 re-measures with the gate fixes.

## R808: round r2, 5 alternating boot pairs

Image `tabbyapi:tokenize-offloop-r2` (`04b06fa97c4d`, label `b961eb83…`), knobs on (BOTH) against both knobs off (NONE) in the order N B B N N B B N N B, one fresh boot and NVMe tier directory per boot, one prompt nonce (7575), greedy peers (temperature 0). r2 encodes once at every length and hops to the worker only when the prompt has more than 12,000 characters (about 4,200 tokens, about 3.8 ms of inline encode at this model's 2.87 characters per token) or while an earlier hop is pending. Every gate was pre-registered in the driver's header before the run, and an independent review recomputed each one from the raw per-event records and matched the printed values.

| gate | rule | BOTH | NONE | result |
| --- | --- | --- | --- | --- |
| stall, class (i) fe (80 events per arm) | BOTH ≤ min(10 ms, 0.2 × NONE) | 2.8 ms (p90 5.7, max 12.7) | 53.1 ms (max 60.2) | pass |
| sections A / B | A ≤ 8 ms | 2.1 / 0.6 ms | 28.4 / 24.7 ms | pass |
| total excess per class (i) event | BOTH ≤ NONE − 0.6 × NONE fe | 100.5 ms | 148.6 ms | pass |
| shams with fe ≥ 20 ms (145 per arm) | ≤ 10 % | 0.7 % | 0.7 % | pass |
| time to first token, class (iii) (60 events per arm) | BOTH ≤ NONE + 10 ms | 215.7 ms median | 219.3 ms | pass: per pair −7.2, −2.6, −2.8, −1.6, −0.3 ms, mean −2.9 [−6.2, +0.3] |
| steady-state time per decode step, median gap outside every event window | paired upper bound ≤ +1.0 % | | | +0.01 % [−0.25, +0.27], pass |
| decode aggregate, phase A, cycles 4 and later | paired lower bound ≥ −1.0 % | | | +0.75 % [+0.21, +1.28], pass |
| mean gap including the stalls | paired upper bound ≤ +1.0 % | | | −0.54 % [−1.01, −0.08], pass |
| greedy output | BOTH = NONE, and = R792 | | | 135 of 135 rows identical (`fn_greedy` 6 + 6, `chat_greedy` 6 + 6 + 2 at n = 2 and the second `long`, per pair, streamed and not); 12 of 12 against R792 |
| tokenizer | in-container ids = the served file's; `/v1/token/encode` with the chat path's `add_bos` | | | 864 of 864; ids and `usage.prompt_tokens` identical, both arms |
| VRAM, health | boot free per card, errors | 1,181 / 1,759 MiB | 1,181 / 1,759 MiB | every boot; 0 tracebacks, 0 OOM, 0 failed requests, 0 short peers |

**Decode.** The statement the data supports is the steady-state one: the time per decode step outside the events is unchanged, +0.01 % [−0.25, +0.27]. The +0.75 % aggregate is not a decode gain: it is the removed stall (the mean gap including the stalls, −0.54 %, about 48 ms per long arrival at R808's synthetic rate of about 1.2 arrivals per 12.6 s cycle) plus MTP acceptance moving with batch timing (tokens per step +0.21 % [−0.07, +0.48]; server acceptance BOTH 59.54 to 59.58 %, NONE 59.12 to 59.28 % in pairs 2 to 5). With one nonce and greedy peers the prompts were fixed, but the decoded tokens still depended on when the events landed. Per pair:

| pair | aggregate | mean gap incl. stalls | steady-state step | tokens per step | calibrator (no events) |
| --- | --- | --- | --- | --- | --- |
| 1 | +0.33 % | −0.47 % | −0.27 % | −0.14 % | −0.34 % |
| 2 | +0.68 % | −0.21 % | +0.20 % | +0.48 % | −0.19 % |
| 3 | +0.65 % | −0.45 % | +0.18 % | +0.20 % | +0.05 % |
| 4 | +0.60 % | −0.40 % | +0.09 % | +0.21 % | +1.02 % |
| 5 | +1.48 % | −1.19 % | −0.14 % | +0.29 % | +1.17 % |

The calibrator (4 streams, no events, 3 runs per boot) spreads up to 5 % within one boot, so its median moves about 1 % on noise alone; its flags on pairs 4 and 5 do not track the steady-state step time, and every decode leg passes without those two pairs. GPU clocks, power and peak VRAM were identical across the 10 boots. The absolute rates from this run are not the served regime (greedy peers at temperature 0) and are not stated here.

**Time to the first token.** Short prompts: −2.9 ms [−6.2, +0.3], so r1's +41.6 ms is gone. R806's registration named a +5 ms bar; the R808 driver registered +10 ms before the run; −2.9 ms passes both. The long arrival's own time to the first token did not change (class (i) per pair +5.2, +6.6, +0.4, −0.6, +3.6 ms, mean +3.0, not significant): the worker's pickup latency takes back the encode it saves. The reduction in stall goes to the other streams.

**Output.** Chat `long`, a long answer, equals R792's reference on the first `long` run of every boot and a second, fixed output on every later run in that boot, in both arms and both modes: second-run state, not the patch. The chat path's hop to the worker has no greedy comparison of its own (chat `long` is a long answer, not a long prompt); its ids are identical by the class (i) and (ii) events' server-reported prompt token counts (identical across all 10 boots), by the 30,720 cached tokens every class (i) event hit, and by the tokenizer check. `long100k` (about 470,000 characters) covers both hop paths, streamed and non-streamed, and was identical in every boot.

**Not covered.** `/v1/token/encode`, the Kobold router, the per-chunk fallback encode, loop-think's injection encode and ExLlamaV3's banned-string encode still run on the loop through the same locked `encode_batch`; one of them arriving while a worker encode holds the lock waits for the rest of it, at most ~110 ms at 120k tokens. The threshold counts characters: 12,000 characters of CJK text is ~11 ms of inline stall, of byte-fallback-heavy text 30 to 45 ms. The NVMe prefix tier refused every write in all 10 boots because the fast file system was below its free-space floor, so the tier path was inert in both arms; the change does not touch it.

## R808p: the promotion (2026-09-29, served from 17:01 UTC)

Under the GPU lock: the image id `04b06fa97c4d`, the new launcher identical to the live one outside `DAILY_IMG`, a backup of the live launcher. After the boot: the served model, the image id, 39 keys, pool 901,120, the knob readback `tokenize-offloop-r2 1 1 1 12000 0 1` inside the container (revision, encode once, off-loop in TabbyAPI and in ExLlamaV3, threshold, `should_offload(12000)`, `should_offload(12001)`), boot free VRAM 1,181 / 1,759 MiB, `fn_greedy` 6 of 6 identical to R792. Served since 2026-09-29 19:01 CEST. Rollback: `DAILY_IMG=tabbyapi:rebase-dev-r3-loopthink5`, R792's launcher.
