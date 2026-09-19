# R507: grouped MoE prefill (E3) round 1 is ×1.48–1.55 per layer at 2,048 rows and +6–8 % end to end with half the layers eligible

Results directory on the serving host: `results/2026-09-18-r507-prefill-e3`. Raw records: [`2026-09-18-r507-prefill-e3/`](2026-09-18-r507-prefill-e3/). Driver: [`scripts/r507-prefill-e3.sh`](../../scripts/r507-prefill-e3.sh). Date: 2026-09-18.

E3 runs each MoE layer's prefill as one grouped GEMM over all routed experts instead of one GEMM per expert. Round 1 covered only experts quantized at K=3, which is 23 of the 2.50bpw pack's 49 MoE layers. Image `tabbyapi:prefill-e3-r1`, 2.50bpw at 786,432.

Per layer, real weights, same input: card 0 layer 0, 512 rows 2.408 → 1.961 ms (×1.23), 2,048 rows 4.197 → 2.717 ms (**×1.55**); card 1 layer 37, 512 rows ×1.12, 2,048 rows ×1.48. NRMSE against the default path 6.4e-4 to 1.6e-3.

Cold prefill, one salted invocation per context (t/s, OFF / OFF2 against ON / ON2): 30k 7,323–8,047 against 7,338–8,395 (+3 %); 60k 8,263–8,301 against 8,475–8,890 (+5 to +7 %); 120k 8,440–8,622 against 9,020–9,257 (**+6 to +8 %**). Decode unchanged. c1 fingerprint canonical on all four arms; the 30k fingerprint changes under E3 (prefill accumulation order). Needles 5/5. Round 2 ([R513](r513-prefill-e3-r2.md)) covers every K.

Instrument defect found here: `fn_bench --unique` seeded its filler identically on every invocation, so a second "cold" run hit the prefix cache. `fn_bench` now takes `--salt`, and cold prefill is measured one invocation per context with a random salt.
