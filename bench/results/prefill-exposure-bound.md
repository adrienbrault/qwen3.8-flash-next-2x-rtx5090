# What prefill/decode overlap actually costs, measured on real traffic

> **CORRECTION, 2026-09-20, same day as first publication.** This page first claimed "the ceiling on prefill-interference work: 3.1 % of decode" and used it to close decode-priority scheduling. **That bound was wrong and is withdrawn.** It composed production's exposure share with R585's 14 % loss as though the 14 % were a loss *per exposed second*. It was not: R585's ~750-token arm was resident only about **4 %** of its window (≈0.3 s of prefill per ~8.6 s arrival), so a 4 % dose produced a 14 % loss, and scaling that *down* to a 22 % dose is only valid if the dose-response decreases with dose. It does not. The exposure measurement below stands; the pricing has been replaced with the dose-response measured directly from the same log. Decode-priority scheduling is reopened.

Measured 2026-09-20 from the container log of the 7.02-hour agent run in [`swebench-agent-cost.md`](swebench-agent-cost.md). Probe: [`bench/prefill_exposure.py`](../prefill_exposure.py). No GPU time was spent; the log already contained the measurement.

## Why this was worth computing

[R585](r585-prefill-interference.md) established that a long generation loses throughput when a large prompt arrives beside it and interleaves its chunked prefill with the decode steps. It measured two losses at two arrival sizes: **0.65×** under a fresh ~45,000-token prompt every 8 seconds, and **0.86×** under a ~750-token prompt at the same rate.

Four rounds then asked whether `chunk_size` is a lever on that mechanism ([R588](r588-chunk-vs-interference.md) and three unpublished successors). None of them answered. The question that prices all of them was never asked, and it turns out to need no GPU at all.

## The measurement

The server logs both halves of it per request:

```
#257 chat/completions: 126 tokens generated at 68.1 T/s · prompt 42,248 tokens, 99% cached,
     520 new in 0.22 s (2,364 T/s) · first token 0.24 s, total 2.09 s · draft 67/82 accepted (82%)
```

`520 new in 0.22 s` is the **measured** uncached work and the **measured** wall it was resident for, contention included — no prefill-rate model is involved. With the line's own timestamp, `first token` and `total`, each request yields a prefill interval and a decode interval, and exposure is the share of decode-stream-seconds overlapping some other request's prefill. All 9,926 requests parse. A first pass read only 9,705, because the regex matched `N% cached` while the server prints `none cached` for a request that hit no prefix at all — so the 221 it dropped were exactly the fully-uncached ones, the fresh prefills this measurement exists to count, biasing the exposure downward. The figures below are the complete set.

## What production's prefills actually look like

| | real traffic, 7.02 h | what the interference rounds used |
| --- | --- | --- |
| uncached tokens per request | median **757**, p90 1,734, p99 3,216, max **10,224** | ~49,000 |
| requests with ≥ 4,000 uncached | **35 of 9,926 (0.4 %)**, 2 % of all uncached tokens | every one |
| prefix cached | median **97 %**, p10 81 % | 0 % (salted per request) |
| decode exposed to any prefill | **22.0 %** of decode-seconds, 17.9 % of tokens | ~100 % |
| decode exposed to a ≥ 4,000-token prefill | **0.3 %** | ~100 % |

Total uncached prefill work across the window is 3,667 seconds summed over concurrent requests, against a 25,303-second window: 14.5 % duty.

## The bound

The mechanism is real. The exposure is not there.

Every harness reproduced R585's 45,000-token arrival because that is where the effect is large. Production's arrivals are the other size — median 757 uncached tokens, and the largest single prefill in seven hours is 10,224, about a fifth of what the harnesses fired every 8 to 10 seconds. Pricing production's exposure against the **size-matched** loss:

### The dose-response, measured rather than composed

Every completion line carries its own decode rate, so the log prices the lever directly. Each request's exposed fraction is computed, and requests are compared **inside cells matched on both the number of peer decode streams and generation length** — concurrency is the confound, since exposure concentrates in busy stretches and per-stream decode falls with batch size regardless of any prefill.

| peer decode streams | low-exposure | high-exposure | delta |
| --- | --- | --- | --- |
| 0 | 193.8–197.5 t/s (exp 0 %) | 185.9–221.7 (exp 4–10 %) | **+7 to +15 %** |
| 1 | 150.0–162.4 | 96.1–128.4 (exp 17–34 %) | −20 to −36 % |
| 2 | 123.6–133.5 | 75.8–96.3 (exp 22–43 %) | −22 to −40 % |
| 3 | 99.2–118.4 | 60.7–78.4 (exp 25–53 %) | −23 to −49 % |
| 4 | 89.4–101.7 | 39.5–64.4 (exp 28–65 %) | −32 to −61 % |
| 5 | 68.0–76.3 | 40.5–53.7 (exp 30–59 %) | −21 to −47 % |
| 6 | 61.1–76.2 | 45.5–52.2 (exp 30–45 %) | −15 to −40 % |

**Pooled, controlling peers and generation length: 105.2 t/s at 5 % mean exposure against 80.0 t/s at 40 % — −23.9 %.** The `peers = 0` row is the negative control and it is clean: with nothing else in the batch, exposure costs nothing, which is what rules out the comparison being an artefact of busy periods alone.

Fitting `rate = r₀(1 − k·exp)` to the pooled contrast gives `k ≈ 0.66`. At production's mean exposure of 22.0 %, decode therefore runs about **14.6 % below its unexposed rate**, and removing the overlap entirely would be worth **roughly +15 % of decode** (10–17 % depending on how the residual confound is treated).

**Limits, stated plainly.** This is observational, not an A/B. Controlling *mean* peers over a request's window does not remove burstiness *within* it, so some of the 24 % may be micro-period load rather than prefill. The mechanism is not attributed: it is consistent with prefill chunks displacing decode steps, and also with the batch-composition transient of a request joining mid-flight.

The cause is the 97 % prefix-cache hit rate. An agent session resends a growing conversation, so each request shares a long prefix with the previous one and costs a few hundred new tokens rather than a full prefill. The interference lever therefore only reaches session starts and cache misses.

## What this closes

Splitting exposure by prefill size (small < 1,500 new tokens against ≥ 1,500) at matched peers, generation length and arrival rate — 0.86/s against 0.91/s, so not an admission-count difference — the cells whose total exposure is close read only −1 % to −10 %, while the cells with large exposure gaps read −23 % to −33 %. The harm tracks **exposure-seconds**, roughly independent of how large the prefill is per second. That separates the two levers:

- **`chunk_size` stays closed.** Smaller chunks trade severity-per-second against longer residency, and the product is what matters — to first order the decode loss is the prefill compute share of the engine, which is chunk-invariant. Four rounds of nulls are consistent with that. The deployability kills stand on their own anyway: chunk 512 leaves 337 MiB free on cuda:0 against a 1,041 MiB reference, failing the headroom rule, and takes the arriving request's TTFT from 6.1 s to 12.6 s. Worth noting the exposure measurement says nothing about *tail latency*, where smaller chunks shorten each decode-stall event — if decode smoothness is ever the goal, that question is still open.
- **Deferral-style scheduling is reopened.** It removes the overlap rather than reshaping it, so the ~15 % is its prize rather than its ceiling. Its cost is TTFT on the arriving request, which nothing here prices. One caveat before building it: an overlay that skips prefill "while any peer sequence is decoding" cannot work at 4.21 mean streams, where that condition is almost always true — pure deferral would starve every arrival. The shape that can work at this concurrency is a priority weight, with decode scheduled first and prefill taking the remainder.

## Method note

Four harnesses failed for one reason, and it generalises: **when the interfering load is generated by the system under test, it cannot be held constant.** Equalise the arrival count and the dwell time diverges; equalise the dwell and the count diverges. The delivered dose is their product and the knob under test sets both factors.

The evidence is measured, not argued. A closed-loop generator (fire a prefill, wait for it, fire the next) delivered **138–140 prefills to the chunk-2048 arms against 60 to the chunk-512 arms**, because a ~49k prefill takes 23.6 s at chunk 512 and 12.2 s at 2048. The open-loop replacement — fire every 10 s on the wall clock, cap 4 in flight — moved the asymmetry rather than removing it: **15 in-flight-cap skips on the 512 arm against 0 on the 2048 arm**, with the victim at 9.4–11.8 t/s under noise against 182–205 alone, while the 2048 arm read 60–120 against 214–249. That 8× is the cap binding, not a chunk effect.

The general fix is to measure the delivered dose and compare at equal dose. The better move here was to price the prize first.
