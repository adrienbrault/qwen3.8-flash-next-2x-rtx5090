# R584 and R585: the whole gap between the published number and a real session, accounted for

Results directories on the serving host: `results/2026-09-20-r584-sampling-regime` and `results/2026-09-20-r585-prefill-interference`. Raw records: [`2026-09-20-r584-sampling-regime/`](2026-09-20-r584-sampling-regime/), [`2026-09-20-r585-prefill-interference/`](2026-09-20-r585-prefill-interference/). Drivers: [`scripts/r584-sampling-regime.sh`](../../scripts/r584-sampling-regime.sh), [`scripts/r585-prefill-interference.sh`](../../scripts/r585-prefill-interference.sh).

## The question

A real three-agent session against this server delivered **65.6 tokens/s per stream** where the table in the README says 213. [R583](r583-long-generation.md) cleared the two obvious explanations: generation length costs nothing, and the windowed MTP draft cache does not either. What it left was a list of ways the benchmark differs from a session — greedy instead of sampled, English filler instead of code, and above all every stream starting at the same instant instead of arriving when an agent happens to need something.

These two rounds measure that list.

## R584: sampling costs something, but not much

All at ~9.8k context, 3,000 forced tokens, NVMe tier off, on the served daily.

| content | streams | greedy | temperature 0.6 | ratio |
| --- | --- | --- | --- | --- |
| prose | 1 | 284.1 | 215.6 | 0.76 |
| prose | 3 | 160.0 | 162.0 | 1.01 |
| code | 1 | 223.4 | 212.2 | 0.95 |
| code | 3 | 140.7 | 131.5 | 0.93 |

The engine verifies a draft token by running the request's own sampler at that position and accepting when the result matches, so a sampler that does not always pick the argmax accepts fewer drafts. The cost is real and it is between 0 and 24 %, not a factor of three. (The round also measured a 512-token pair to test whether sampling and length compound. The greedy 512 reading came back at 205.7 against 284.1 for the same arm at 3,000 tokens, which is outside the band every other 512-token measurement here has produced, so that comparison is not reported as a result.)

## R585: arrival shape is the missing factor

One greedy 3,000-token generation at ~9.8k context, measured while different things happen around it. **NVMe tier on**, as production runs it. Background prompts are salted per request, so nothing is served another arm's cached pages.

| arm | decode tokens/s | vs alone | what else was running |
| --- | --- | --- | --- |
| A | 236.4 | 1.00 | nothing |
| B | 154.5 | **0.65** | a fresh ~45k-token prompt arriving every 8 s |
| C | 202.4 | 0.86 | a fresh ~750-token prompt arriving every 8 s |
| D | 141.1 | **0.60** | two more 3,000-token generations, started 20 s apart |

**B against C is the point.** Same arrival rate, same number of extra requests, same 64-token replies — only the prompt size differs. Big prompts cost the long generation 35 %, small ones 14 %. What hurts is not traffic, it is *prefill*: a 45k-token prompt is 22 chunks of 2,048 tokens, and every chunk is a forward pass the decoder does not get. The interleaving is invisible to a benchmark whose streams all prefill at the start and then decode undisturbed.

**D is the other half.** Three long generations that did not start together run at 141 per stream, against the 172–194 [R583](r583-long-generation.md) measured for three that did. Staggering alone costs something, because the batch is never the steady shape the scheduler settles into.

## Putting it together

A real session has both at once, and its requests are sampled:

| | tokens/s |
| --- | --- |
| one greedy stream, idle server, NVMe tier on | 236 |
| × 0.65, big prompts arriving mid-generation | 154 |
| × 0.60, other long generations already in flight | 92 |
| × 0.76…0.95, sampled rather than greedy | 70–88 |

The measured session delivered **65.6**, and a greedy probe fired into that same traffic read **88.9**. Both land inside this range. Multiplying penalties measured separately is an approximation and the true interaction is not exactly a product, but nothing is left over that needs another explanation: the gap is arrival shape first, sampling second, and generation length not at all.

## What this means for the numbers in this repository

They are honestly measured and they describe a steady-state batch on an otherwise idle server. That is a real operating point — it is what a queue of uniform work looks like — but it is not what an agent session looks like. A session's requests interrupt each other, and the interruption costs about 40 % before sampling is accounted for.

One more thing worth recording: arm A here read 236 tokens/s where R583 measured 283 on the same shape with the NVMe tier **off**. The two rounds differ in warm-up as well as in the tier, so this is a candidate rather than a result, but it is the first sign that the tier costs something on the decode path and it deserves its own round.
