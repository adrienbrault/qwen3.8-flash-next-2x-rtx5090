# R529: `tool_choice` "required" and named functions are enforced; tool-eval 88.0, TC-45 passes on every trial

Results directories on the serving host: `results/2026-09-19-r529-promote-tool-choice` (promotion) and `results/2026-09-19-r523-tool-choice` (validation, round 3). Raw records: [`2026-09-19-r529-promote-tool-choice/`](2026-09-19-r529-promote-tool-choice/), [`2026-09-19-r523-tool-choice/`](2026-09-19-r523-tool-choice/). Drivers: [`scripts/r529-promote-tool-choice.sh`](../../scripts/r529-promote-tool-choice.sh), [`scripts/r523-tool-choice.sh`](../../scripts/r523-tool-choice.sh). Date: 2026-09-19.

Unpatched TabbyAPI accepts `tool_choice: "required"` and a named function but does not enforce them: tool-eval's TC-45 ("tool_choice=required Compliance") scored 0 on every trial. The overlay [`docker/overlays/tool-choice-r1/`](../../docker/overlays/tool-choice-r1/) changes only TabbyAPI's Python:

- For `required` and named requests, an llguidance grammar over the tool-call format switches on when reasoning ends (TabbyAPI's existing `filter_trigger`). Tags are matched by token id.
- The model may answer in content first. If the turn then ends without a call, a second job continues after the content with a call-only grammar, so the client sees reasoning, content, the call and one `tool_calls` finish. Blocking the end of turn instead made the model loop on its closing sentence (round 2).
- A forced turn that ends without the call for any reason other than the client's own limits returns a 503.
- `auto`, `none` and requests without `tool_choice` take the original code path unchanged.

## Validation (R523 round 3, 2026-09-19 02:30 UTC)

99 in-image tests pass with the model's tokenizer and chat template. Request matrix on the patched image, 12 trials per cell:

| check | result |
| --- | --- |
| `required`, streaming and not, thinking on and off (48 requests) | 48/48: `finish_reason: tool_calls`, valid JSON arguments, content only before the call |
| named function | 4/4 |
| TC-45's second turn (answer after the tool result, still forced) | 24/24 |
| argument validation, content cap | 4/4, 2/2 |
| 8 concurrent forced requests | 8/8 |
| TC-45 × 12 | 100 (unpatched: 0) |
| server log | 0 `ToolChoiceNotHonoured`, 0 loop detections, 50 content-to-call continuations |

The cold identity check for `auto` / `none` could not be judged from outputs: two fresh boots of the same unpatched image agreed on 3 of 12 cold tool-bearing requests, the same rate as unpatched against patched. The tool schemas make these prompts long enough to go through the E3 grouped MoE prefill, whose atomic accumulation is not run-to-run deterministic: two cold runs of one 8,017-token prompt diverged at generated token 128 with it on and never with it off (R524, 2026-09-19). Identity for `auto` / `none` rests on the unchanged code path and a 120-scenario differential test on CPU (0 differences); the short c1 and 30k fingerprints below are byte-identical.

## Promotion gates (R529, promoted 03:43 UTC)

Built on the served image: `tabbyapi:mtp-pruned-r1-tc1`, launcher unchanged except the image.

| gate | result |
| --- | --- |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| decode (one boot, against the [R522](r522-mtp-pruned.md) figures) | code c1 222.3 vs 221.8 t/s, c4 508.8 vs 508.4; prose c1 198.9, c4 487.2 |
| agentic-edit | 6/6 in four modes |
| needles | 5/5 at 131k and 5/5 at 240k |
| tool-eval 69×4 | **88.0 ± 1.6** (trials 121 / 122 / 124 / 118); TC-45 2/2 points on all 4 trials, against 0 on all 4 in [R528](r528-promote-mtp-pruned.md) (84.8 ± 1.5) |
| GSM8K 5-shot n=500, no stop strings | 0.978 |
