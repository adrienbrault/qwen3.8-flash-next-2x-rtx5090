# R347: Stamina

Results directory on the serving host: `results/2026-09-16-r347-soak`. Raw records: [`2026-09-16-r347-soak/`](2026-09-16-r347-soak/). Driver: [`scripts/r347-soak.sh`](../../scripts/r347-soak.sh). Date: 2026-09-16.


Forty rounds of c4, 1,024 forced code tokens each, alone on the box: per-round median decode 63.5–65.5 t/s,
**drift 101.0 % of the start** (first three rounds 64.5, last five 65.1). No decay, no error, no VRAM drift.

That is the figure the first attempt could not produce: the gate suite's soak ran while a native extension was
compiling on the same host and read 40.5 t/s from round six onward, which has the shape of stamina decay and was
not. See `docs/GOTCHAS.md` #8.
