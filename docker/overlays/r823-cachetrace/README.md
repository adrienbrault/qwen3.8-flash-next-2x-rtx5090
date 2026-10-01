# Checkpoint trace and interval seam

`tabbyapi:r823-cachetrace` adds an engine interval seam and checkpoint instrumentation to `tabbyapi:merge-tok-r1`. The image sets `EXL3_CACHE_TRACE=1`. `fix.patch` exposes `EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP`; `exl3.patch` instruments rendering/allocation, recurrent saves/losses and prefix selection; `app.patch` carries request and chat-boundary context through TabbyAPI. The before/after SHA-256 manifests and landing check pin the installed sources.

The trace records salted digests, lengths, role/page positions and timing/cache fields; it does not emit prompt text or token IDs. CPU token arrays remain in the bounded registry used to compute LCPs. Ordinary client identifiers are hashed; synthetic probe keys beginning `r823/` are retained for exact joins.

## Build and fixture regeneration

```sh
docker build --network=none -t tabbyapi:r823-cachetrace docker/overlays/r823-cachetrace
uv run --with tokenizers --with jinja2 bench/r823_fixtures.py --model /path/to/checkpoint --out docker/overlays/r823-cachetrace/fixtures --nonce synthetic-823-v1
```

The builder uses seed 823 and the checkpoint's tokenizer/chat template. It writes `manifest.json`, `payloads.json.gz` and `SHA256SUMS`, with exact token counts, LCPs and payload hashes. Fixture payloads and model assets are not shipped. Measurement and validation require the generated fixtures; the image build itself does not. [R823–R823p](../../../bench/results/r823-tail-checkpoints.md) describes the immutable cases, sampler, paired boots and copied records. The SSE client and validator are in [`bench/r823_reuse.py`](../../../bench/r823_reuse.py).
