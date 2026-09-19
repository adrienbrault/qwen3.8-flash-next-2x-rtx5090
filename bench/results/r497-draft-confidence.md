# R497: confidence-gated dynamic draft crashes at c4 with a CUDA-graph out-of-memory

Results directory on the serving host: `results/2026-09-18-r497-draft-confidence`. Raw records: [`2026-09-18-r497-draft-confidence/`](2026-09-18-r497-draft-confidence/). Driver: [`scripts/r497-draft-confidence.sh`](../../scripts/r497-draft-confidence.sh). Date: 2026-09-18.

exllamav3's dynamic draft stops a draft chain early when the draft head's confidence drops below a threshold. Arms FIX / D40 / D60 / D70 / FIX2 / D60b, 3.05bpw pack, 360,448.

Two defects. (1) The image never applied the threshold override: the host's legacy `docker build` drops the body of a `RUN` heredoc, so every dynamic arm ran exllamav3's default 0.4 (same c1 hash and speed at 0.4, 0.6 and 0.7). Since then no image recipe here uses a heredoc; scripts are copied in and run. (2) Dynamic draft crashes deterministically on the second prose c4 run, about 321 tokens into each stream, in all four dynamic arms: `GPU assert: out of memory exllamav3_ext/graph.cu 51` in `cudaGraphInstantiate`. Variable draft lengths per job create new verify shapes, each captured as a new graph, until the 833 MiB left on the card run out; fixed depth only sees batch 1 to 4 at 4 query rows.

Before the crash, dynamic (0.4) against FIX / FIX2, `fn_bench` 2,048 tokens, aggregate: code c1 194–197 / 175–177 against 216–221 t/s; code c4 458–480 against 443–464; prose c1 204–206 then 179–181 against 174–180; prose c4 427–429 against 437–439. Fixed depth stays.
