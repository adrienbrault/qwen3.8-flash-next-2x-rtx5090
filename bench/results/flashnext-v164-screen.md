# ExLlamaV3 1.6.4 candidate: initial throughput regression

Measured 2026-10-21. Results directory `2026-10-21-152916-flashnext-v164-screen`; A was copied from the completed unchanged control in `2026-10-21-151818-flashnext-v164-screen`. [Raw records](2026-10-21-flashnext-v164-screen/) and [summarizer](../v164_summary.py), [decode client](../v164_decode_probe.py) and [agent client](../v164_agent_replay.py). This report contains completed A/B/C measurements. D (2.68 bpw) and the separate checkpoint accuracy comparison are still running or pending; no accuracy improvement is established.

A is the served `r828-prompt-lookup-r3` stack and r0b0tlab 2.50 bpw checkpoint. B uses the same checkpoint with the [pinned 1.6.4 rebase](../../docker/overlays/rebase-v164/manifest.json); C uses that rebased engine with the pinned self-calibrated 2.18 bpw pack. Candidate image ID: `sha256:4d5132f605016d227fbc934572bb13864bbcfd964870acb300ce3d8345571167`. The old native extension is replaced by the official wheel. The [audit](../../docker/overlays/rebase-v164/PATCH-AUDIT.md) records changed kernels, scheduler and recurrent state, and omitted MTP/head/embedding/host optimizations. A/B measures the whole stack change; it does not isolate an upstream version effect.

All three arms use two RTX 5090s, stock 600/575 W limits, +4500 MHz memory offset, 262144-token sequence window, 901120-token 8-bit KV pool, eight slots, manual split [30,30], and batch-indexed MTP depth 3 through four jobs / depth 2 through eight. Checkpoints load from NVMe. New native tuning caches are separate from the daily cache. This screen uses one boot per arm, rather than an alternating boot experiment; the regression is a screen result, with no isolated cause established.

Decode uses chat completions with greedy sampling and no request-level thinking override, short distinct code/prose prompts (118/106 tokens), 1024 forced output tokens, concurrency 1/4/8. Each cell has one excluded warm-up and two measured rounds. Every arm has 26 valid records per kind; all return 1024 server-counted tokens with finish reason length. Per-stream figures are the median of the two within-round stream medians. Aggregates are the median of the sums of measured stream rates, not per-stream rate multiplied by concurrency.

| Arm | Code c1 | Code c4 | Code c8 | Prose c1 | Prose c4 | Prose c8 |
|---|---:|---:|---:|---:|---:|---:|
| A: served, 2.50 | 265.85 | 157.26 | 105.69 | 288.01 | 153.13 | 106.44 |
| B: rebased, 2.50 | 179.30 | 67.00 | 50.09 | 172.76 | 70.16 | 50.39 |
| C: rebased, 2.18 | 170.14 | 68.19 | 50.66 | 174.36 | 68.44 | 49.38 |

Values above are decode tokens/s per stream. Code c4 measured aggregates A/B/C are 631.86 / 271.55 / 275.47 tokens/s; code c8 aggregates are 839.36 / 404.12 / 406.33 tokens/s. The 2.18 pack does not recover the rebased stack's throughput loss in this screen.

The agent replay uses bare completions at temperature 0.6 and six sessions with seed 20261021, nominal 30000-token initial prompts, three steps, 512 generated tokens and 1200 tool-result tokens per turn, with log-normal size/step variation, one-second think time and one-second initial staggering. All arms complete the same 20 requests, six session instances, every planned step and every requested forced output length. No respawns occur. These are synthetic agent turns, not a coding-task solve-rate evaluation.

| Arm | Request span, seconds | Output tokens / request span | Median TTFT, seconds |
|---|---:|---:|---:|
| A | 43.432 | 216.09 | 0.759 |
| B | 90.625 | 103.56 | 3.362 |
| C | 90.930 | 103.21 | 2.777 |

Request span runs from the first submission through the final completion and includes think/stagger gaps. The B/A span ratio is 2.087. The candidate is not promoted. Basic arithmetic and automatic parsed-tool smokes pass, but task accuracy, long-context retrieval and numerical fidelity are separate evidence; none is inferred from throughput or successful boot.
