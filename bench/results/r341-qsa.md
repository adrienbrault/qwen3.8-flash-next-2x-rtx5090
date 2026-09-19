# R341: QSA sparse multi-job: the A/B

Results directory on the serving host: `results/2026-09-16-r341-qsa`. Raw records: [`2026-09-16-r341-qsa/`](2026-09-16-r341-qsa/). Driver: [`scripts/r341-qsa-ab.sh`](../../scripts/r341-qsa-ab.sh). Date: 2026-09-16.


Both arms built from the same CUDA devel base, same pip resolution, same native rebuild — the only difference is
`APPLY_QSA`. Deep context (152,761 prompt tokens), 1,024 forced code tokens, greedy.

| arm | c2 decode/stream | c2 aggregate | c4 decode/stream | c4 aggregate |
| --- | --- | --- | --- | --- |
| control (unpatched) | 99.2 | 27.4 | 51.7 | 191.6 |
| **treatment (QSA multi-job)** | **125.8** | 28.2 | **72.5** | **257.9** |

**+27 % at c2 and +40 % per stream at c4**, at the depth where the captured QSA path used to fall back to eager for
bsz>1 — the case the patch exists for. The c2 aggregate is unchanged because those two requests queue on a 262k
pool against 2 × 152,761 tokens of context; the per-stream figures are the ones to read.

The correctness gate passed on all three checks at 74,796-token contexts with bsz=2: the two concurrent greedy
outputs match within each arm, and control == treatment byte for byte (`sha256` `2e0c2a563342…`, 1,365 bytes). An
attention patch that changes what the model says is not a win, and this one does not.
