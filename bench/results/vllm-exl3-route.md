# ExLlamaV3 against vLLM on the same checkpoint

Results directory on the serving host: `results/2026-09-18-vllm-exl3-route`. Raw records: [`2026-09-18-vllm-exl3-route/`](2026-09-18-vllm-exl3-route/). Date: 2026-09-18.


The vllm-exl3 route serves the same `qwen3.8-flash-next-exl3-3.05bpw` checkpoint through vLLM main (2026-09-18),
TP2 on both cards. Same instrument as this stack's own rows: `fn_bench`, 2,048 forced tokens, greedy, code.

| engine | profile | kind | c1 decode (t/s) | c4 aggregate (t/s) | KV pool (tokens) | GSM8K |
| --- | --- | --- | --- | --- | --- | --- |
| ExLlamaV3 + TabbyAPI (this stack, 2026-09-17) | served, 8-bit KV, MTP depth 3 | code | 207–214 | 425–450 | 262,144 | 0.9158 (n=1319) |
| vLLM | `d-mtp3` — BF16 KV, MTP depth 3 | code | 131.6 | 475.4 | 95,183 | 0.945 (n=200) |
| vLLM | `d-mtp2` — BF16 KV, MTP depth 2 | code | 121.7 | 421.4 | 108,651 | 0.92 (n=200) |
| vLLM | `e-fp8` — fp8 KV, no MTP | code | 75.6 | 222.2 | 309,657 | — |
| vLLM | `e-fp8-mtp1` — fp8 KV, MTP depth 1 | code | 117.0 | 339.2 | 159,744 | — |
| vLLM | `e-fp8-mtp2` — fp8 KV, MTP depth 2 | code | 118.1 | 406.0 | 131,072 | 0.94 (n=200) |

At c4 the vLLM route's best profile (`d-mtp3`) reads 475.4 t/s aggregate on code against 425–450 here, 1.06–1.12× this stack; at c1 it reads 131.6 t/s against 207–214 here, 62–64 % of this stack's rate. The depth-2 profile reads 421.4 and 121.7 (parity at c4, 57–59 % at c1). The KV pool tracks the KV dtype and the draft depth across the rows: 309,657 tokens with fp8 KV and no MTP on the vLLM route, 262,144 with 8-bit KV here, 159,744 with fp8 KV at MTP depth 1, 131,072 with fp8 KV at depth 2, 108,651 with BF16 KV at depth 2 and 95,183 at depth 3. fp8 KV against BF16 KV at depth 2 costs 3 % at c1 and 4 % at c4 for 1.21× the pool. Pool figures are the engine's `GPU KV cache size` lines, collected in `kv-pools.txt` in the results directory; the e-fp8-mtp1 aggregates are the `fn_bench` summary lines in `r475-audit.txt`; the d-mtp3 row is one run from `r476-d-mtp3-records.jsonl` and its GSM8K line is in `r476-audit.txt`; the e-fp8-mtp2 row is `r475b-e-fp8-mtp2-records.jsonl` with its GSM8K line in `r475b-audit.txt`.

c8, prose, prefill and long-context are **not measured on the vLLM route**. This stack's figures for those shapes,
for reference: c8 aggregate 550–604 t/s on code; prose c1 160.6–165.3 t/s; a 27,501-token prompt prefills in 3.5 s
(7,700–7,800 t/s) and a 110,081-token prompt in 13.1 s (8,380–8,390 t/s), both on 2026-09-17; long-context
retrieval 5/5 at 26.5k, 105.7k and 158.5k prompt tokens.
