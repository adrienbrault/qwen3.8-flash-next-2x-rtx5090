# New self-calibrated packs on the unchanged optimized engine

Measured2026-10-21, results `2026-10-21-163522-flashnext-old-engine-screen`. The image is the served `r828-prompt-lookup-r3`, exact ID `sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724`, with all47 selectors unchanged. Only the checkpoint, isolated tuning/config paths and test port change. Both packs boot with the901120-token KV pool and262144-token window, and pass arithmetic and automatic structured-tool smoke. Neither is promoted. [Raw records and checked summary](2026-10-21-flashnext-old-engine/); the [decode client](../v164_decode_probe.py) and [agent client](../v164_agent_replay.py) match the earlier throughput screen.

| Checkpoint / engine | Code c4 tokens/s per stream | Code c4 measured aggregate | Six-agent request span |
|---|---:|---:|---:|
| Original2.50 / served control |157.26|631.86|43.432s|
| New2.18 / served engine |148.72|599.60|47.687s|
| New2.18 / rebased engine |68.19|275.47|90.930s|
| New2.68 / served engine |136.22|546.67|Invalid: prefill OOM|
| New2.68 / rebased engine |67.16|270.78|91.736s|

Code decoding uses chat completions, greedy sampling,118-token distinct prompts,1024 forced outputs and concurrency4. Each new old-engine arm has one excluded warm-up and two measured rounds, eight valid records with server count1024 and finish reason length. Per-stream values are the median of the two within-round medians; aggregates sum the actual individual rates before taking the median across rounds. The served/rebased references come from the completed [same-session throughput screen](flashnext-v164-screen.md), with one boot per configuration. This is a screen, not an alternating-boot promotion measurement.

The same six-agent replay uses temperature0.6, seed20261021 and the sizes/think/stagger settings in that earlier screen. The2.18 old-engine run completes all20 requests, all planned steps in six session instances and every requested output length. Its47.687-second span is from first request submission to final completion. It is slower than the original checkpoint's control and faster than the rebased stack in this screen.

The2.68 old-engine run returns zero generated tokens for all20 attempted agent requests. Its long-prompt prefill raises CUDA out of memory; one reported failed allocation is80MiB on GPU0. The short decode samples remain valid, but the8-second process runtime is not a replay result. The client marks the SSE connection successful despite empty aborted responses, and the original runner exits0; neither is semantic completion evidence. The updated driver explicitly checks forced token counts and every planned agent step. A separate2.68 retry with655360 cache tokens is running to test prefill headroom, without changing the engine or promoting the pack.

The old engine already includes fractional-bitrate support, so the1.6.3 converter version alone does not require a runtime rebase. The [accuracy screen](flashnext-v164-quality.md) used only the rebased engine and does not qualify these packs on the served runtime. The older runtime, custom kernels and recurrent state differ numerically; no old-engine accuracy or BF16-reference qualification is claimed here.
