# GPU duty cycle — measured under a live agentic load

Date: 2026-09-16.


`nvidia-smi` sampled at 1 Hz for 20 s while eight SWE-bench agents were running against the seat:

| | mean | min | max | mean power | limit |
| --- | --- | --- | --- | --- | --- |
| GPU 0 | **45 %** | 22 % | 84 % | 226 W | 600 W |
| GPU 1 | **38 %** | 20 % | 53 % | 195 W | 575 W |

This is the layer-split signature, and it is *not* a tuning miss: with `tensor_parallel: false` and
`gpu_split: [30, 30]` the two cards take turns over their own layers, so each is busy only while its own layers
execute. Under a real eight-agent load it is lower than the 44–47 % measured during a single request, because agent
turns are short and bursty: the cards are idle waiting for the next request as well as for each other.

What moves it and what does not, from this session's own measurements:

| lever | effect on the duty cycle |
| --- | --- |
| QSA multi-job | none directly; +27 %/+40 % throughput at deep-context concurrency, so more work in the same busy windows |
| concurrency-indexed draft depth | none directly; +35 % aggregate at c4 for the same reason |
| host KV tier | none; measured flat |
| expert parallelism | would put both cards on every layer — assessed as needing engine work, not a patch (`supports_tp` is still False in this tree) |
| tensor parallelism | forbidden for this architecture in this engine |
