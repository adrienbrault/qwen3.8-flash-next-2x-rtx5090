# R480: Four slots, K8V4 and one MoE layer on the CPU: the page pool at 8-bit KV

Results directory on the serving host: `results/2026-09-18-r480-exl3-pool`. Raw records: [`2026-09-18-r480-exl3-pool/`](2026-09-18-r480-exl3-pool/). Date: 2026-09-18.


VRAM on the served configuration with 8 slots: main weights ~45.7 GiB, GDN recurrent state ~3.4 GiB (8 slots × 36 linear-attention layers × 4 fp32 copies, one per draft position plus one), page pool 262,144 tokens at 8-bit KV ~4 GiB, MTP head ~1.0 GiB, vision ~0.5 GiB. The 30.5 GiB PLE table and the 1.2 GiB token embeddings stay in host memory. Each arm below found its largest pool by booting down a ladder; the next size up failed with `RuntimeError: Insufficient VRAM in split for model and cache`. Vision on, MTP policy `[[4, 3], [8, 1]]`, window 262,144, `fn_bench` 2,048 forced tokens × 2 runs, needle at five positions, GSM8K n=200 at four concurrent requests.

| arm | pool | free MiB, card 0 / 1 | c1 / 30k greedy | code c1 / c4 | prose c1 / c4 | cold prefill, 22,625 / 90,135 tokens | needle 131k / 240k | GSM8K c4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 slots, 8,8 (previous served) | 262,144 | 947 / 2,511 | reference / reference | 214–217 / 435–442 | 168–172 / 428–436 | 3.06 s (7,390 t/s) / 8.13 s (11,090 t/s) | 5/5 / 5/5 | 0.925 |
| **4 slots, 8,8 (served since 2026-09-18 12:24)** | **360,448** | 895 / 2,581 | identical / identical | 214–217 / 429–444 | 167–172 / 422–436 | 3.12 s (7,250 t/s) / 8.19 s (11,000 t/s) | 5/5 / 5/5 | 0.925 |
| 4 slots, K8V4 (`8,4`) | 425,984 | 893 / 2,681 | differs / identical | 188–194 / 447–462 | 159–165 / 409–422 | 3.11 s (7,290 t/s) / 8.25 s (10,930 t/s) | 5/5 / 5/5 | 0.920 |
| 4 slots, 8,8, first MoE layer's routed experts on the CPU | 425,984 | 975 / 2,061 | differs / identical | 157–166 / 341–351 | 148–149 / 323–325 | 4.88 s (4,630 t/s) / 14.69 s (6,140 t/s) | 5/5 / 5/5 | 0.935 |

Four slots change no arithmetic: the same kernels run, and only the number of recurrent-state slots changes, so greedy output is byte-identical and decode and prefill are unchanged within the two-run spread. K8V4 stores V at 4 bits: the short greedy output diverges, retrieval holds at 240k, GSM8K stays within one standard error, and code decode at c1 drops about 11 %. The CPU expert path buys the same pool as K8V4 at a 20–25 % decode and 35–45 % prefill cost.

The prefill cells use only the first request of each context. The step sent the same prompt three times, so the second and third requests were served from the prefix cache in 0.19 s and 0.25 s; they are in the records and excluded here.

Promotion gates on the 4-slot configuration (R481): c1 greedy fingerprint `1474eee2f5945248` (the reference), the agent request that originally failed ends in a parsed `todo_write` tool call, eight concurrent requests on four slots all complete (439.7 t/s aggregate at 1,024 tokens, four queued behind the first four), and tool-eval 69×4 reads 84.8 ± 1.0 (CI 84.0–85.5) against 83.5–86.5 for three runs of the 8-slot configuration.
