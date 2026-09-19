# R462: MoE coop mode 3 is bit-identical to mode 1 and 2–3 % slower

Results directory on the serving host: `results/2026-09-17-r462-moecoop-v3-ab`. Driver: [`scripts/r460-moecoop-v2-ab.sh`](../../scripts/r460-moecoop-v2-ab.sh). Date: 2026-09-17.

Mode 3 (`EXL3_MOE_COOP_V2=3`) routes singleton expert runs through the V1 path inside the V2 kernel. Same c1 and 30k fingerprints and GSM8K c8 0.935 as mode 1, but per-stream decode at c1 / c4 / c8 reads 214 / 115 / 70 t/s against 220 / 118 / 73: the mode-3 kernel is 8–19 % slower than V1 on rows that route to distinct or partly overlapping experts, which is what decode rows look like. Driver: `scripts/r460-moecoop-v2-ab.sh` with `ARMS=ON1|ON3`. Rejected; mode 1 is served.
