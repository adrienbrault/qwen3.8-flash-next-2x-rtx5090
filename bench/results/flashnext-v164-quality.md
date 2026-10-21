# Self-calibrated checkpoint accuracy screen on ExLlamaV3 1.6.4

Measured 2026-10-21, results `2026-10-21-155747-flashnext-v164-quality`. The B/C/D arms use the same candidate image, pinned engine/TabbyAPI sources and configuration described in the [throughput screen](flashnext-v164-screen.md). B is the original r0b0tlab 2.50 bpw pack, C is turboderp self-calibrated 2.18 bpw and D is self-calibrated 2.68 bpw. This comparison changes the checkpoint pack while holding the rebased runtime fixed. The new packs were not tested on the served engine in this run. The daily was restored after the run, with no promotion.

[Score-only records](2026-10-21-flashnext-v164-quality/) contain every question/filter outcome and scenario score, plus needle outcomes and summary. The [validator/exporter](../v164_quality_summary.py) checks full locally generated results before exporting those records. Full tool conversations and host boot logs are not published. Both checkpoint revisions and LFS SHA256 values are pinned in the [candidate packet](../../docker/overlays/rebase-v164/).

| Checkpoint | GSM8K strict /100 | GSM8K flexible /100 | Tool points /138 | Tool reported score /100 |
|---|---:|---:|---:|---:|
| B: original 2.50 | 98 | 98 | 121 | 88 |
| C: self-calibrated 2.18 | 98 | 99 | 119 | 86 |
| D: self-calibrated 2.68 | 98 | 99 | 121 | 88 |

GSM8K uses the same first100 examples, five shots, temperature0, max8192 generated tokens and four concurrent requests through lm-eval0.4.12 local-chat-completions. The evaluation proxy removes server stop strings, which otherwise stop the model's reasoning when it repeats Question:. Both the100 document hashes and prompt hashes match between arms. C and D each gain one strict-match answer and lose one versus B; under flexible extraction each gains one and loses none. A one-question difference in this sample does not establish better accuracy.

Tool-eval-bench2.1.0 uses all69 default scenarios, one trial, parallel8, temperature0.6, top_p0.95, top_k20 and seed20261021. B/D each have57 pass,7 partial and5 fail scenarios; C has56 pass,7 partial and6 fail. This is a concurrent tool-behavior screen: the tool warns that saturation can cause timeouts at parallel8, so it is not a serial, load-independent estimate of model quality. The raw point totals are reported separately from the CLI's rounded final score.

The retrieval client requested approximately131072 tokens but actually sent105680 prompt tokens. All packs have3/3 passphrase substring hits at document fractions0.08/0.55/0.96. All nine responses end at the64-token length limit, and the client combines reasoning and visible content. These are retrieval hits, not completed final-answer checks or a validated131K-context gate.

The screen does not establish an accuracy gain for either new pack. The2.18 pack is a candidate for a smaller-memory tradeoff; this sample does not qualify it as equal or superior on coding tasks. No full coding-agent evaluation or BF16-reference fidelity qualification was run. The throughput regression of the rebased stack remains unresolved, and neither candidate is served.
