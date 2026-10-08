# R917: memory of the served configuration (2026-10-08)

Results directory on the box: `2026-10-08-r917-flashnext-memory-layout-124305`; driver `scripts/memory_layout.py` (the GLM-5.3-Flash repository's script, same method), run read-only against the serving container after the launcher's boot warm-up, one single-stream request and eight concurrent 256-token requests. Raw record: [`2026-10-08-r917-flashnext-memory-layout/memory.json`](2026-10-08-r917-flashnext-memory-layout/memory.json), printed summary in `summary.txt`, image in `image.txt`.

Configuration: the served daily (image `tabbyapi:r828-prompt-lookup-r3`, checkpoint `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 901,120-token 8-bit KV page pool, MTP depth 3 / 2, vision on, 4,096 MB host recurrent-state cache cap). Every routed expert is on the GPUs.

Method:

- Weights by category are the checkpoint's tensor bytes from the safetensors headers.
- The KV page pool is 901,120 tokens at 15,236 bytes per token, the figure measured in [R579](r579-promote-mtp-kv-window.md).
- The VRAM used is `nvidia-smi` per GPU. GPU "other" is that total minus the listed items: CUDA contexts, graph pools, GDN recurrent state of the 8 slots, the MTP draft cache, scratch and allocator reserve.
- Host used is the container's memory cgroup, anonymous plus shared memory; the page cache of the checkpoint reads is left out. The recurrent-state cache is a cap that fills on demand, so it is counted inside host "other" and not listed apart (the container held 3.9 GiB in total, below the 4,096 MB cap).

| | VRAM (2 × RTX 5090) | host DRAM |
|---|---|---|
| routed experts (MTP layer's included) | 36.48 GiB | |
| attention, linear attention, norms and other weights | 2.65 GiB | |
| shared experts | 0.11 GiB | |
| lm_head | 0.44 GiB | |
| MTP layer, without its routed experts | 0.09 GiB | |
| KV page pool | 12.79 GiB | |
| vision tower | 0.52 GiB | |
| embedding table | | 1.18 GiB |
| other | 7.19 GiB | 2.71 GiB |
| used | 60.3 GiB (GPU0 30.42, GPU1 29.86) | 3.9 GiB |
| capacity | 63.7 GiB | 60.4 GiB (MemTotal) |
