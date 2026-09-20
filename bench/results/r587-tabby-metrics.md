# R587: a Prometheus `/metrics` endpoint on the TabbyAPI daily

Results directory on the serving host: `results/2026-09-20-r587-tabby-metrics`. Raw records: [`2026-09-20-r587-tabby-metrics/`](2026-09-20-r587-tabby-metrics/). Driver: [`scripts/r587-tabby-metrics.sh`](../../scripts/r587-tabby-metrics.sh). Overlay: [`docker/overlays/metrics-r1/`](../../docker/overlays/metrics-r1/).

Promoted 2026-09-20 09:04 UTC. Daily image `tabbyapi:mtpwin-r2-metrics1`, rollback `tabbyapi:mtpwin-r2` (`launch-flashnext.sh.pre-r587`).

## What the overlay adds

Two added lines in `common/gen_logging.py` (an import and a `record_completion` call at the end of `log_metrics`), two added blocks in `endpoints/core/router.py` (an import and an unauthenticated `/metrics` route beside `/health`), and two new files. The overlay is pure stdlib: `prometheus_client` is not in the container venv and the image build has no package-install step. `install.py` verifies the baseline hash of each file it patches before writing anything, so a base image that has drifted fails the build rather than producing a silently wrong overlay. The build runs 20 unit tests and a render smoke inside the Docker step.

Exported: `tabby_requests_total`, `tabby_generated_tokens_total`, `tabby_generate_seconds_total`, `tabby_prompt_tokens_total`, `tabby_cached_prompt_tokens_total`, and histograms for generated length, context depth, per-request decode rate, time to first token and queue time.

## Verification against known traffic

The endpoint was verified by driving an exact amount of traffic and asserting the counter deltas equal what was sent, rather than by reading the numbers and judging them plausible. Three completions with `min_tokens` = `max_tokens` = 256, issued as direct requests with no warmup round:

| scrape point | requests delta | generated tokens delta | `generate_seconds` delta | implied rate |
| --- | --- | --- | --- | --- |
| candidate image | 3.0 (sent 3) | 768.0 (sent 768) | 3.73 | 205.9 t/s |
| promoted daily | 3.0 (sent 3) | 768.0 (sent 768) | — | 187.3 t/s |

Both deltas are exact. The implied rate is `generated_tokens / generate_seconds`, which is the time-weighted aggregate, not a mean of per-request rates.

## Promotion gates

| gate | requirement | candidate |
| --- | --- | --- |
| c1 greedy fingerprint | `18238d63065ee16c` | `18238d63065ee16c` |
| 30k greedy fingerprint | `4a255910dee2d9c5` | `4a255910dee2d9c5` |
| free VRAM at boot | within the headroom rule | 1,041 / 2,531 MiB, identical to the reference layout |
| endpoint delta | exact | exact, twice |

The fingerprints are a sha256 of `content` + `"|"` + `reasoning_content`, first 16 hex, over one fixed 256-token completion and one fixed completion at roughly 30k prompt tokens, both greedy. The overlay adds a counter update on an already-completed request and cannot legitimately move either digest.

## Why the counters matter for this repository

Every decode rate published here before 2026-09-20 was reconstructed by hand from the container log. That reconstruction is what identified the cause of the gap between published and production decode rates in [R585](r585-prefill-interference.md), and it produced two wrong intermediate answers first: averaging decode rate per request read 176 t/s where the time-weighted truth was 65.6, and TabbyAPI log timestamps carry no date, so a window crossing midnight sorted wrong and corrupted the interval arithmetic. The counter pair

    rate(tabby_generated_tokens_total[5m]) / rate(tabby_generate_seconds_total[5m])

is the time-weighted aggregate directly, with no log parsing. The histograms are the distributions that [R583](r583-long-generation.md), R584 and R585 each required a separate measurement round to observe.

## Harness faults, recorded

The overlay passed every gate it reached on every attempt. Four attempts failed on the harness around it, and three of those failures were reported in the log as evidence against the candidate.

1. `docker inspect` on a container that is not present prints an empty line **and** exits non-zero, so reading the live image as `inspect || grep` captured both outputs and built an image tagged `"\ntabbyapi:mtpwin-r2-metrics1"`.
2. The endpoint reported 4 requests and 776 generated tokens against 3 × 256 = 768 sent. The endpoint was correct: the benchmark probe issues an 8-token warmup round before its recorded runs, and 776 = 768 + 8. The traffic was replaced with direct requests so that what is asserted is exactly what is sent.
3. The fingerprint gate grepped the greedy probe's log for a 16-hex digest. That probe emits no digests: it writes reply text to a JSONL file and prints a preview line, and its equivalence check is a separate `--compare` pass that diffs two arms' text. The grep could not match, and no match was read as "not canonical".
4. The gate was rewritten around the digest helpers from an earlier promotion round, which set their API base with a `/v1` suffix and append `/chat/completions`. This script sets the bare base because its other callers append their own `/v1`, so the lifted helpers posted to a 404 and both digests came back empty.

A gate that cannot pass is worse than no gate, because its failure is indistinguishable from a real one. The fifth attempt ran the digest helper by hand against the live daily before the round started, and confirmed it returned `18238d63065ee16c`.

## Not done here

Adding a `tabby` scrape job to the monitoring stack is a cluster change, not a GPU change, and is not part of this round.
