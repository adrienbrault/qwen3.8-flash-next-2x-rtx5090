# R348: Capabilities

Results directory on the serving host: `results/2026-09-16-r348-capabilities`. Raw records: [`2026-09-16-r348-capabilities/`](2026-09-16-r348-capabilities/). Driver: [`scripts/r348-capabilities.sh`](../../scripts/r348-capabilities.sh). Date: 2026-09-16.


| check | result |
| --- | --- |
| JSON-schema structured output | **PASS** — content parses *and* satisfies the schema (`city`, `population`, `coastal`, `climate`), and the server log shows the grammar engaged (`constrained generation … json_schema (req)`), so it is not the model being agreeable |
| tool call parsing | **PASS** — `write_note` with `{"file_path": "/tmp/gate-note.txt", "content": "hello from the gate"}`, arguments JSON-valid and limited to declared parameters |
| vision | **PASS** — a red 32×32 PNG built in-process, answer "Red" |
| reasoning channel | **PASS** — 309 chars in `reasoning_content`, 61 in `content` |
