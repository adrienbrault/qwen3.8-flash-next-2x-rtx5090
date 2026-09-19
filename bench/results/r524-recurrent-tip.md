# R524: recurrent checkpoints at the end of each answer are correct but save about 9k prefill tokens per 120-call agent replay; not served

Results directory on the serving host: `results/2026-09-19-r524-recurrent-tip` (try 8, tip on, and the same-hour tip-off arm). Raw records: [`2026-09-19-r524-recurrent-tip/`](2026-09-19-r524-recurrent-tip/). Drivers: [`scripts/r524-recurrent-tip.sh`](../../scripts/r524-recurrent-tip.sh), [`scripts/r524b-recurrent-tip-off.sh`](../../scripts/r524b-recurrent-tip-off.sh). Date: 2026-09-19.

The stock generator stores a recurrent-state checkpoint every 2,048 tokens. An agent's next call re-reads everything after the last checkpoint, including the previous answer. The overlay (written for this repository by an Opus agent round) also stores a checkpoint at the end of each answer ("tip"), keeps each conversation's newest tip when evicting, and resumes from it.

## Correctness (step 0, in-process harness)

- Six agent turns of 8,017 to 12,057 prompt tokens, 400 generated tokens each. Resuming from tips gives the same output as a cold run on every turn.
- Storing tips never changes the output of the job that stores them.
- The tip at position 9,216 is compared with a checkpoint built by prefill at the same position. Its PLE token ids equal the actual tokens. Its GDN conv windows line up best at zero offset (median rel-L2 0.036, against 0.67 or more one token off). Its float state is 0.159 rel-L2 from the prefill-built one at worst (median 0.036); the stock decode-built checkpoint at 8,192 is itself 0.076 from prefill (median 0.021 to 0.024). Decode and prefill use different kernels (fused recurrent step against the chunked delta rule), so this is expected numerics, not a position error.
- The comparison also found a base ExLlamaV3 bug: stored PLE checkpoints alias the live slot ([R530](r530-promote-plefix.md)).

## Served A/B (same hour, one boot each)

| | tip off | tip on |
| --- | --- | --- |
| agent replay in echo mode, 8 conversations, 120 calls | 95.4 s | 100.7 s |
| prompt tokens served from cache | 86.6 % | 89.9 % |
| new prompt tokens prefilled | 66,020 | 57,096 |
| code decode, 1 stream | 225.4 t/s | 222.0–223.1 t/s |

The tips save about 9k prefill tokens over the replay, about 1 s of prefill. Four tip-off runs of the same replay today took 95 to 108 s, so the saving is below the run-to-run spread. The tip snapshots also hold 4 GB of host RAM. Not served.
