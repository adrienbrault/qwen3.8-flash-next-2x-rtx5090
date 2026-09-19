# Boot and footprint

Date: 2026-09-16.


| quantity | value | conditions |
| --- | --- | --- |
| model load | 11.2–11.5 s | warm Triton + coop-autotune caches on disk |
| warmup (first inference after load) | 0.27–0.36 s | same |
| VRAM at idle, model resident | 31.9 GB / 30.1 GB of 32.6 GB per card | both cards held by one process |
| GPU power at idle-resident | ~227 W / ~218 W of 600/575 W | measured during a request, layer-split duty cycle |
