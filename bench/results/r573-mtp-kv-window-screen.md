# R573: the windowed draft cache is faster, including far past its window

Results directory on the serving host: `results/2026-09-19-r573-mtp-kv-window-screen`. Raw records: [`2026-09-19-r573-mtp-kv-window-screen/`](2026-09-19-r573-mtp-kv-window-screen/). Driver: [`scripts/r573-mtp-kv-window-screen.sh`](../../scripts/r573-mtp-kv-window-screen.sh).

[R569](r569-mtp-kv-window.md) showed the window buys page pool. The open question was acceptance beyond its span: a draft that sees only a sink page plus the last 16,384 tokens could propose worse tokens deep in a long context. It does not.

Both arms at 966,656 tokens, NVMe tier off, 8 slots.

| measurement | window off | window 16,384 |
| --- | --- | --- |
| 1 stream, code / prose, 24 prompts each | — | +1.89 % / +1.48 % |
| 4 streams, code / prose | — | +2.13 % / +1.45 % |
| cold 100k-token prompt, 1 stream | 205.4 t/s | 215.0 t/s (**1.047×**) |
| 8 streams at ~16k context, per stream | 79.4 t/s | 81.2 t/s (1.023×) |
| needles at ~131k and ~240k | — | 5/5 and 5/5 |

Decode gains rather than loses, because the draft layer's attention is smaller. The layer split knob does not change this placement: `[29.5, 30.5]` and `[29, 31]` boot with exactly the free VRAM `[30, 30]` does, so the pool stays at the 1,015,808 the ladder found.
