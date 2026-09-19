# R567: a deeper-draft controller gains too little for what draft depth 4 costs the page pool

Results directory on the serving host: `results/2026-09-19-r567-adaptive-draft-r3-try2`. Raw records: [`2026-09-19-r567-adaptive-draft-r3-try2/`](2026-09-19-r567-adaptive-draft-r3-try2/). Driver: [`scripts/r567-adaptive-draft-r3.sh`](../../scripts/r567-adaptive-draft-r3.sh).

Allowing depth 4 at one decoding job reserves recurrent history for four draft positions on all 8 slots. The pool ladder priced that first: 966,656 → 933,888 tokens, **−32,768 (−3.4 %)**, measured as the highest pool whose free VRAM at boot stays at the served floor.

Four boots at that pool, 24 prompts per cell, paired against the served configuration:

| arm | code 1 stream | prose 1 stream | code 4 | prose 4 |
| --- | --- | --- | --- | --- |
| fixed depth 4 at 1 job | −0.83 % | −2.43 % | +1.84 % | +4.49 % |
| controller (deeper only) | +1.70 % [−2.03, +5.52] | +2.37 % [+0.29, +4.29] | +0.54 % | +3.03 % |
| controller, context split off | +0.79 % | +2.32 % | +3.40 % | +4.71 % |

The controller chose depth 4 in about one round in eleven (mean per-request depth 3.09) and it does avoid fixed depth 4's loss at 1 stream: +4.93 % [+1.50, +8.36] against it on prose. Against the served configuration it needed +2.0 % on code at 1 stream and reached +1.70 %.

Read the 4-stream columns with care: both policies draft the same depth at 4 jobs, yet all three arms at the smaller pool beat the served arm by +0.5 to +4.7 %, so part of every column is a boot or pool difference rather than the controller.

Not served: at most about +2 % at 1 stream for 3.4 % of the page pool is the wrong trade while pool is the scarce resource.
