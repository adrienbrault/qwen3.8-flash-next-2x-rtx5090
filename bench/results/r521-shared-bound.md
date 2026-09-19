# R521: after the side-stream overlap, the shared expert costs 1.9 µs per layer at 4 rows; a fused kernel could save at most about 1 %

Results directory on the serving host: `results/2026-09-19-r521-shared-bound`. Raw records: [`2026-09-19-r521-shared-bound/`](2026-09-19-r521-shared-bound/). Driver: [`scripts/r521-shared-bound.sh`](../../scripts/r521-shared-bound.sh). Date: 2026-09-19.

[R490](r490-shared-overlap.md) moved the shared expert onto a side CUDA stream. A fused shared-plus-routed kernel cannot be bit-identical to the served path, so it is only worth building if the shared expert still sits on the critical path. The probe runs one decoder layer's MoE block in isolation on the served image (layers 0 and 29 on cuda:0, 30 and 47 on cuda:1, a copy of the served tune cache) and times it with the overlap on against the routed experts alone. The difference is the shared expert's residual cost.

| rows per layer (served shape) | residual, mean over layers (max), µs |
| --- | --- |
| 1 | 11.6 (13.2) |
| 2 | 5.6 (6.5) |
| 4 (1 stream, draft 3) | 1.9 (2.5) |
| 8 | 4.3 (9.5) |
| 16 (4 streams, draft 3) | 5.6 (13.1) |

The shared chain alone takes 18.6–23.6 µs; the overlap already hides 10–25 µs of it. Over 48 layers the residual is at most 0.09 ms per step at 1 stream (0.6 % of the [R519](r519-profile-2p50.md) step) and 0.27 ms at 4 streams with draft 3 (1.2 %). A fused kernel is not built; a scheduling change could recover at most about 1 % at 4 streams.
