# R504: the reference-base image is identical in output and speed to the served image

Results directory on the serving host: `results/2026-09-18-r504-refbase`. Raw records: [`2026-09-18-r504-refbase/`](2026-09-18-r504-refbase/). Driver: [`scripts/r504-refbase.sh`](../../scripts/r504-refbase.sh). Date: 2026-09-18.

Later overlays (R499, R501, R507, R513) were written against a reference tree of the served sources. R504 checked that the image built from that tree, with its two unpromoted switches off, serves the same as the daily: DAILY / REF / DAILY2 / REF2, 3.05bpw pack at 360,448. All four arms canonical. `fn_bench` 2,048 × 2, aggregate, DAILY / DAILY2 against REF / REF2: code c1 212.5–221.1 / 216.9–220.6 against 218.2–218.6 / 221.4–221.8; code c4 443–462 against 447–463; prose c1 173–180 against 175–181; prose c4 434–453 against 438–455. Pass.
